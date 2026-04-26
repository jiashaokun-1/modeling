"""Bridge: aggregate fine-grained ``OpGraph`` nodes into ``training.ir.Graph``.

This module takes a captured ``OpGraph`` (hundreds of aten ops per layer)
and produces a ``training.ir.Graph`` (~12 semantic ops per layer) whose
shape and cost model are derived from the captured shapes rather than
hand-written formulas.  The resulting ``Graph`` is compatible with all
existing downstream analysis (``flops.py``, ``shard.py``, ``compose/``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zrt.training.ir.graph import Graph, Op, Tensor, Collective
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec

if TYPE_CHECKING:
    from zrt.ir.graph import OpGraph as CaptureOpGraph
    from zrt.ir.node import OpNode as CaptureOpNode


# ── Scope → semantic kind mapping ──────────────────────────────────────────────

# Keywords in scope → Op kind and name suffix
_SCOPE_RULES: list[tuple[list[str], str, str]] = [
    # (scope_keywords, op_kind, name_component)
    # Order matters — first match wins. Put more specific patterns first.

    # Embedding / head (model-level, not layer-level)
    (["embed_tokens"],                                          "embed",   "embed"),
    (["lm_head"],                                               "lm_head", "lm_head"),

    # Norm — must precede "attention" to avoid false match on post_attention_layernorm
    (["final_ln"],                                              "ln",      "final_ln"),
    (["input_layernorm"],                                       "ln",      "ln1"),
    (["post_attention_layernorm"],                              "ln",      "ln2"),
    (["rmsnorm", "rms_norm", "layer_norm", "layernorm"],       "ln",      "ln"),

    # MLA (DeepSeek) attention projections — before generic q_proj
    (["q_a_proj"],                                              "matmul",  "q_a_proj"),
    (["q_b_proj"],                                              "matmul",  "q_b_proj"),
    (["kv_a_proj"],                                             "matmul",  "kv_a_proj"),
    (["kv_b_proj"],                                             "matmul",  "kv_b_proj"),

    # Attention: rotary / rope
    (["rotary_emb", "rotary", "rope"],                          "rope",    "rope"),

    # Attention: standard projections
    (["q_proj", "k_proj", "v_proj", "qkv_proj"],               "matmul",  "qkv"),
    (["o_proj", "out_proj"],                                    "matmul",  "o_proj"),

    # Attention: compute (after projections so o_proj/q_proj don't match attn)
    (["flash_attn", "scaled_dot_product"],                      "attn_core", "attn_core"),
    (["self_attn"],                                             "attn_core", "attn_core"),
    (["attention"],                                             "attn_core", "attn_core"),

    # MoE router — bare "gate" (NOT gate_proj) in MoE context
    (["gate_proj"],                                             "matmul",  "gate_proj"),
    (["up_proj"],                                               "matmul",  "up_proj"),
    (["down_proj"],                                             "matmul",  "down_proj"),

    # MoE dispatch/combine (all_to_all)
    (["all_to_all", "a2a"],                                     "dispatch", "dispatch"),

    # MoE shared expert
    (["shared_expert", "shared_experts"],                       "matmul",  "shared_expert"),

    # MoE routed experts
    (["experts", "expert"],                                     "matmul",  "expert"),

    # MoE gate/router — ".gate" or "MoEGate" (after gate_proj to avoid substring match)
    (["moegate", "moe_gate", ".gate"],                          "router",  "router"),
    (["router"],                                                "router",  "router"),

    # FFN activation
    (["silu", "swiglu", "act_fn"],                              "swiglu",  "swiglu"),

    # Residual add
    (["residual"],                                              "add",     "residual"),
    (["add"],                                                   "add",     "residual"),
]


def _match_scope(scope: str) -> tuple[str, str] | None:
    """Return (op_kind, name_suffix) for a scope string, or None."""
    scope_lower = scope.lower()
    for keywords, kind, suffix in _SCOPE_RULES:
        for kw in keywords:
            if kw in scope_lower:
                return (kind, suffix)
    return None


def _dtype_from_tensor_meta(tmeta) -> Dtype:
    """Extract DType from a captured TensorMeta (or OpNode).

    Handles both ``ir.types.DType`` (str-based Enum) and
    ``training.spec.dtype.Dtype`` (int-based Enum).
    """
    if tmeta is None:
        return Dtype.BF16
    dt = getattr(tmeta, 'dtype', None)
    if dt is None:
        return Dtype.BF16
    # ir.types.DType is a str Enum → map by string name
    if hasattr(dt, 'name'):
        name = dt.name.upper()  # BF16, FP32, FP16, FP8
        if hasattr(Dtype, name):
            return getattr(Dtype, name)
    # int value (training Dtype)
    if hasattr(dt, 'value') and isinstance(dt.value, int):
        return dt
    return Dtype.BF16


def aggregate_to_training_ir(
    capture_graph: "CaptureOpGraph",
    model: ModelSpec,
) -> Graph:
    """Convert a captured OpGraph into a training.ir.Graph.

    Algorithm
    ---------
    1. Group captured nodes by ``.layer``.
    2. Within each layer, bucket by scope-pattern match (see ``_SCOPE_RULES``).
    3. Each bucket becomes one ``training.ir.Op``.
       - For matmul-like buckets: ``m``/``n``/``k`` are extracted from the
         first matmul node's input / output shapes.
       - For memory-bound ops (ln, rope, swiglu, add): ``bytes_fwd`` is
         summed from all member nodes.
    4. Layers with no captured ops (untraced) are filled with the dense
       layer template from ``builders`` as a fallback.

    Parameters
    ----------
    capture_graph : OpGraph
        Captured graph from Path A (with or without shape_template).
    model : ModelSpec
        Model geometry for fallback layer generation.

    Returns
    -------
    Graph
        Compatible with ``shard.py``, ``flops.py``, and ``compose/``.
    """
    # ── Phase 1: group captured nodes by layer ───────────────────────────────
    captured_layers: dict[int, list[CaptureOpNode]] = {}
    for node in capture_graph.nodes.values():
        try:
            lid = int(node.layer) if node.layer else -1
        except (ValueError, TypeError):
            lid = -1
        captured_layers.setdefault(lid, []).append(node)

    # ── Phase 2: build semantic ops from captured nodes ──────────────────────
    all_ops: list[Op] = []
    layer_index: dict[int, tuple[int, int]] = {}

    for lid in sorted(captured_layers.keys()):
        if lid < 0:
            continue  # skip embedding / non-layer ops for now
        layer_ops = _layer_to_semantic_ops(captured_layers[lid], lid, model)
        if not layer_ops:
            continue
        start = len(all_ops)
        all_ops.extend(layer_ops)
        layer_index[lid] = (start, len(all_ops))

    # ── Phase 3: handle embedding / lm_head (layer_id=-1) ────────────────────
    if -1 in captured_layers:
        nonlayer_ops = _nonlayer_ops(captured_layers[-1], model)
        all_ops = nonlayer_ops + all_ops
        # Adjust layer indices
        offset = len(nonlayer_ops)
        layer_index = {k: (s + offset, e + offset) for k, (s, e) in layer_index.items()}

    # ── Phase 4: fill untraced layers with template ──────────────────────────
    for i in range(len(model.layers)):
        if i not in layer_index:
            from zrt.training.ir.builders import dense_block
            template = dense_block(
                hidden=model.hidden, ffn=model.ffn, seq=model.seq_len,
                num_heads=model.num_heads, num_kv_heads=model.num_kv_heads,
                head_dim=model.head_dim, layer_id=i, act_dtype=model.act_dtype,
            )
            start = len(all_ops)
            all_ops.extend(template)
            layer_index[i] = (start, len(all_ops))

    return Graph(ops=all_ops, collectives=[], layer_index=layer_index)


# ── Layer-level aggregation ────────────────────────────────────────────────────


def _layer_to_semantic_ops(
    nodes: list["CaptureOpNode"], lid: int, model: ModelSpec,
) -> list[Op]:
    """Convert one layer's captured nodes into semantic ops."""
    # Classify each node
    buckets: dict[str, list["CaptureOpNode"]] = {}
    for node in nodes:
        match = _match_scope(node.scope)
        if match is None:
            continue
        kind, suffix = match
        # Use suffix to distinguish same-kind ops (ln1 vs ln2, up_proj vs gate_proj)
        bucket_key = f"{kind}.{suffix}"
        buckets.setdefault(bucket_key, []).append(node)

    # Build ops in a deterministic order matching the plan's table.
    # Bucket keys are now "kind.suffix" (e.g. "ln.ln1", "matmul.qkv").
    layer_kind = model.layers[lid] if lid < len(model.layers) else LayerKind.DENSE
    ops: list[Op] = []

    # LN before attention (input_layernorm)
    for ln_key in ("ln.ln1", "ln.ln", "ln.final_ln"):
        if ln_key in buckets:
            ops.append(_build_ln_op(buckets.pop(ln_key), lid, f"L{lid}.ln1", layer_kind))
            break

    # QKV: MLA projections or standard q/k/v/qkv_proj
    qkv_nodes: list["CaptureOpNode"] = []
    for k in list(buckets.keys()):
        if k.startswith("matmul.") and any(
            kw in k for kw in ("qkv", "q_proj", "k_proj", "v_proj",
                               "q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj")
        ):
            qkv_nodes.extend(buckets.pop(k))
    if qkv_nodes:
        ops.append(_build_matmul_op(qkv_nodes, lid, f"L{lid}.qkv", layer_kind))

    # RoPE
    if "rope.rope" in buckets:
        ops.append(_build_memory_bound_op(buckets.pop("rope.rope"), lid, f"L{lid}.rope", "rope", layer_kind))

    # Attention core
    if "attn_core.attn_core" in buckets:
        ops.append(_build_attn_core_op(buckets.pop("attn_core.attn_core"), lid, f"L{lid}.attn_core", model))

    # O projection
    if "matmul.o_proj" in buckets:
        ops.append(_build_matmul_op(buckets.pop("matmul.o_proj"), lid, f"L{lid}.o_proj", layer_kind))

    # Residual after attn
    if "add.residual" in buckets:
        ops.append(_build_memory_bound_op(buckets.pop("add.residual"), lid, f"L{lid}.residual1", "add", layer_kind))

    # LN before FFN (post_attention_layernorm)
    if "ln.ln2" in buckets:
        ops.append(_build_ln_op(buckets.pop("ln.ln2"), lid, f"L{lid}.ln2", layer_kind))

    # MoE router
    for router_key in ("router.router", "router.router"):
        if router_key in buckets:
            ops.append(_build_memory_bound_op(buckets.pop(router_key), lid, f"L{lid}.router", "router", layer_kind))
            break

    # MoE dispatch
    if "dispatch.dispatch" in buckets:
        ops.extend(_build_dispatch_combine(buckets.pop("dispatch.dispatch"), lid, f"L{lid}.dispatch", layer_kind))

    # Shared expert
    for k in list(buckets.keys()):
        if "shared_expert" in k:
            ops.append(_build_matmul_op(buckets.pop(k), lid, f"L{lid}.shared_expert", layer_kind))

    # MoE expert matmuls (grouped expert FFN)
    for k in list(buckets.keys()):
        if "matmul.expert" in k:
            ops.append(_build_matmul_op(buckets.pop(k), lid, f"L{lid}.expert_ffn", layer_kind))

    # FFN up/gate
    for ffn_suffix in ("up_proj", "gate_proj"):
        key = f"matmul.{ffn_suffix}"
        if key in buckets:
            ops.append(_build_matmul_op(buckets.pop(key), lid, f"L{lid}.{ffn_suffix}", layer_kind))

    # SwiGLU
    if "swiglu.swiglu" in buckets:
        ops.append(_build_memory_bound_op(buckets.pop("swiglu.swiglu"), lid, f"L{lid}.swiglu", "swiglu", layer_kind))

    # FFN down
    if "matmul.down_proj" in buckets:
        ops.append(_build_matmul_op(buckets.pop("matmul.down_proj"), lid, f"L{lid}.down_proj", layer_kind))

    # Remaining unmatched nodes → generic fallback
    for k, orphan_nodes in buckets.items():
        ops.extend(_generic_ops(orphan_nodes, lid, layer_kind))

    return ops


# ── Op builders ─────────────────────────────────────────────────────────────────


def _t(name: str, shape: tuple[int, ...], dtype: Dtype) -> Tensor:
    return Tensor(name=name, shape_logical=shape, shape_local=shape,
                  dtype=dtype, is_activation=True)


def _build_matmul_op(
    nodes: list["CaptureOpNode"], lid: int, name: str, layer_kind: LayerKind,
) -> Op:
    """Build a matmul Op from captured aten.mm/addmm nodes."""
    # Use the first matmul (largest output) as representative
    matmul_nodes = [n for n in nodes
                    if "mm" in n.op_type.lower() or "matmul" in n.op_type.lower()
                       or "addmm" in n.op_type.lower() or "linear" in n.op_type.lower()]
    if not matmul_nodes:
        matmul_nodes = nodes

    # Extract m, n, k from the most representative node
    rep = _pick_largest_output(matmul_nodes)
    m, n, k = _extract_mnk(rep)

    act_dtype = _dtype_from_tensor_meta(rep.inputs[0] if rep.inputs else None)
    in_shape = (m, k)
    out_shape = (m, n)

    return Op(
        name=name, kind="matmul",
        inputs=[_t(f"{name}.in", in_shape, act_dtype)],
        outputs=[_t(f"{name}.out", out_shape, act_dtype)],
        meta={"m": m, "n": n, "k": k, "n_local": n, "k_local": k},
        layer_id=lid, layer_kind=layer_kind,
    )


def _build_attn_core_op(
    nodes: list["CaptureOpNode"], lid: int, name: str, model: ModelSpec,
) -> Op:
    """Build an attn_core Op from captured attention nodes."""
    rep = nodes[0]
    # Look for s, heads, head_dim in the rep's meta or from node annotations
    s = model.seq_len  # fallback
    b = 1
    heads = model.num_heads
    head_dim = model.head_dim
    h_kv = model.num_kv_heads * model.head_dim

    # Try to extract from the first node with shape info
    for n in nodes:
        if n.outputs:
            shape = n.outputs[0].shape
            if len(shape) >= 2:
                s = shape[0]
                h_kv = min(shape[1], h_kv)
            break

    act_dtype = _dtype_from_tensor_meta(rep.inputs[0] if rep.inputs else None)
    h_attn = heads * head_dim
    out_shape = (s, h_attn)

    return Op(
        name=name, kind="attn_core",
        inputs=[
            _t(f"{name}.q", (s, h_attn), act_dtype),
            _t(f"{name}.k", (s, h_kv), act_dtype),
            _t(f"{name}.v", (s, h_kv), act_dtype),
        ],
        outputs=[_t(f"{name}.out", out_shape, act_dtype)],
        meta={"b": b, "s": s, "heads": heads, "head_dim": head_dim,
              "causal": True, "h_kv": h_kv},
        layer_id=lid, layer_kind=LayerKind.DENSE,
    )


def _build_ln_op(
    nodes: list["CaptureOpNode"], lid: int, name: str, layer_kind: LayerKind,
) -> Op:
    """Build a layer-norm Op from captured LN/RMSNorm nodes."""
    rep = nodes[0]
    raw_shape = rep.inputs[0].shape if rep.inputs else (model_seq_fallback, 4096)
    # Flatten to 2D: (m, h) — take product of all but last dim as m
    if len(raw_shape) >= 2:
        import math
        m = math.prod(raw_shape[:-1])
        h = raw_shape[-1]
    elif len(raw_shape) == 1:
        m, h = raw_shape[0], 1
    else:
        m, h = model_seq_fallback, 4096
    act_dtype = _dtype_from_tensor_meta(rep.inputs[0] if rep.inputs else None)
    shape = (m, h)
    # Sum bytes_fwd from all member nodes
    total_bytes = sum(
        sum(t.mem_bytes for t in n.inputs) + sum(t.mem_bytes for t in n.outputs)
        for n in nodes
    )
    return Op(
        name=name, kind="ln",
        inputs=[_t(f"{name}.in", shape, act_dtype)],
        outputs=[_t(f"{name}.out", shape, act_dtype)],
        meta={"bytes_fwd": total_bytes},
        layer_id=lid, layer_kind=layer_kind,
    )


def _build_memory_bound_op(
    nodes: list["CaptureOpNode"], lid: int, name: str,
    kind: str, layer_kind: LayerKind,
) -> Op:
    """Build a memory-bound op (rope, swiglu, add, router, etc.)."""
    rep = nodes[0]
    in_shape = rep.inputs[0].shape if rep.inputs else (1, 4096)
    out_shape = rep.outputs[0].shape if rep.outputs else in_shape
    total_bytes = sum(
        sum(t.mem_bytes for t in n.inputs) + sum(t.mem_bytes for t in n.outputs)
        for n in nodes
    )
    act_dtype = _dtype_from_tensor_meta(rep.inputs[0] if rep.inputs else None)
    return Op(
        name=name, kind=kind,
        inputs=[_t(f"{name}.in", in_shape, act_dtype)],
        outputs=[_t(f"{name}.out", out_shape, act_dtype)],
        meta={"bytes_fwd": total_bytes},
        layer_id=lid, layer_kind=layer_kind,
    )


def _build_dispatch_combine(
    nodes: list["CaptureOpNode"], lid: int, name: str, layer_kind: LayerKind,
) -> list[Op]:
    """Build dispatch/combine ops for MoE."""
    ops: list[Op] = []
    for i, node in enumerate(nodes):
        shape = node.inputs[0].shape if node.inputs else (1, 4096)
        act_dtype = _dtype_from_tensor_meta(node.inputs[0] if node.inputs else None)
        total_bytes = sum(t.mem_bytes for t in node.inputs) + sum(t.mem_bytes for t in node.outputs)
        ops.append(Op(
            name=f"{name}_{i}", kind="dispatch",
            inputs=[_t(f"{name}_{i}.in", shape, act_dtype)],
            outputs=[_t(f"{name}_{i}.out", shape, act_dtype)],
            meta={"bytes_fwd": total_bytes},
            layer_id=lid, layer_kind=layer_kind,
        ))
    return ops


def _nonlayer_ops(nodes: list["CaptureOpNode"], model: ModelSpec) -> list[Op]:
    """Handle embedding, lm_head, and final norm from non-layer nodes."""
    ops: list[Op] = []
    embed_nodes = [n for n in nodes if "embed" in n.scope.lower()]
    if embed_nodes:
        ops.append(_build_matmul_op(embed_nodes, -1, "embed", LayerKind.DENSE))
    ln_nodes = [n for n in nodes if "norm" in n.scope.lower() and "final" in n.scope.lower()]
    if ln_nodes:
        ops.append(_build_ln_op(ln_nodes, -1, "final_ln", LayerKind.DENSE))
    lm_nodes = [n for n in nodes if "lm_head" in n.scope.lower()]
    if lm_nodes:
        ops.append(_build_matmul_op(lm_nodes, -1, "lm_head", LayerKind.DENSE))
    return ops


def _generic_ops(
    nodes: list["CaptureOpNode"], lid: int, layer_kind: LayerKind,
) -> list[Op]:
    """Fallback: create generic compute/memory ops from unmatched nodes."""
    ops: list[Op] = []
    for node in nodes:
        shape = node.inputs[0].shape if node.inputs else (1, 4096)
        act_dtype = _dtype_from_tensor_meta(node.inputs[0] if node.inputs else None)
        kind = "matmul" if "mm" in node.op_type.lower() else "ln"
        ops.append(Op(
            name=f"L{lid}.generic_{node.id}", kind=kind,
            inputs=[_t(f"{node.id}.in", shape, act_dtype)],
            outputs=[_t(f"{node.id}.out",
                        node.outputs[0].shape if node.outputs else shape, act_dtype)],
            meta={"m": shape[0], "n": shape[1], "k": shape[1]} if kind == "matmul" else {"bytes_fwd": 0},
            layer_id=lid, layer_kind=layer_kind,
        ))
    return ops


# ── Helpers ─────────────────────────────────────────────────────────────────────


def _pick_largest_output(nodes: list["CaptureOpNode"]) -> "CaptureOpNode":
    """Return the node with the largest total output bytes."""
    best = nodes[0]
    best_bytes = 0
    for n in nodes:
        total = sum(t.mem_bytes for t in n.outputs)
        if total > best_bytes:
            best_bytes = total
            best = n
    return best


def _extract_mnk(node: "CaptureOpNode") -> tuple[int, int, int]:
    """Extract (m, n, k) from an aten matmul node."""
    m = n_val = k = 0
    # Input: (m, k), output: (m, n)
    if node.inputs and len(node.inputs) >= 1:
        shape = node.inputs[0].shape
        if len(shape) >= 2:
            m, k = shape[0], shape[1]
        elif len(shape) == 1:
            m, k = shape[0], 0
    if node.outputs and len(node.outputs) >= 1:
        out_shape = node.outputs[0].shape
        if len(out_shape) >= 2:
            n_val = out_shape[1]
            if m == 0:
                m = out_shape[0]

    # Override with meta info if available (from annotations or extra_args)
    meta_m = _get_meta_int(node, "m") or m
    meta_n = _get_meta_int(node, "n") or n_val
    meta_k = _get_meta_int(node, "k") or k

    return (meta_m or m, meta_n or n_val, meta_k or k)


def _get_meta_int(node: "CaptureOpNode", key: str) -> int | None:
    """Try to read an int from node annotations."""
    val = node.annotations.get(key)
    if isinstance(val, int):
        return val
    val = node.attrs.get(key)
    if isinstance(val, int):
        return val
    return None


# Module-level fallback for shape extraction
model_seq_fallback = 4096
