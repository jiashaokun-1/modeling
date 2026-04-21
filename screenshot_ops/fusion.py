"""Automatic operator fusion using module hierarchy."""
from __future__ import annotations

import inspect
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from screenshot_ops.tracker import ModuleTracker


@dataclass
class FusionSpec:
    """A fusion pattern auto-discovered from dispatch tracing."""
    module_key: str
    aten_op_sequence: List[str]
    num_sub_ops: int
    fusion_level: str
    example_module_path: str
    occurrences: int = 1
    fused_input_shapes: str = ""
    fused_input_dtypes: str = ""
    fused_input_sources: str = ""
    fused_output_shapes: str = ""
    fused_output_dtypes: str = ""
    fused_output_sources: str = ""
    input_map: List[Dict] = field(default_factory=list)
    output_map: List[Dict] = field(default_factory=list)
    parameter_map: List[Dict] = field(default_factory=list)
    constant_map: List[Dict] = field(default_factory=list)


def _split_shape_list(s: str) -> List[str]:
    """Split '[1, 128], [64]' into ['[1, 128]', '[64]']."""
    if not s:
        return []
    result = []
    depth = 0
    current = []
    for ch in s:
        if ch == "[":
            depth += 1
            current.append(ch)
        elif ch == "]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            result.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        result.append("".join(current).strip())
    return result


def _strip_layer_prefix(module_path: str) -> str:
    """Remove the 'model.layers.N.' prefix from a module path.

    'model.layers.0.input_layernorm' → 'input_layernorm'
    'model.layers.0.self_attn.q_a_layernorm' → 'self_attn.q_a_layernorm'
    'model.norm' → 'norm'
    """
    parts = module_path.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                int(parts[i + 1])
                return ".".join(parts[i + 2:]) or module_path
            except ValueError:
                pass
    return module_path


def _parent_path(module_path: str) -> str:
    if "." in module_path:
        return module_path.rsplit(".", 1)[0]
    return ""


def _aten_short(op_name: str) -> str:
    """'aten.mm.default' → 'mm'."""
    parts = op_name.split(".")
    return parts[1] if len(parts) >= 2 else op_name


def _compute_fused_io(ops: List[Dict[str, Any]],
                      graph=None,
                      module_tracker: Optional[ModuleTracker] = None) -> Dict[str, Any]:
    """Identify external inputs/outputs of a fused group.

    Classifies each external input as one of:
      - **input**: a tensor coming from outside the module (e.g. hidden_states)
      - **parameter**: a module weight/bias (nn.Parameter)
      - **constant**: a scalar literal (e.g. epsilon)

    Parameters
    ----------
    ops:
        Raw op records belonging to this fused group.
    graph:
        ``DataFlowGraph`` used to resolve tensor IDs through skip ops.
    module_tracker:
        ``ModuleTracker`` with ``path_to_module`` mapping, used to
        inspect the module's ``forward()`` signature and
        ``named_parameters()`` for input/parameter classification.
    """
    group_indices = {op["idx"] for op in ops}

    module_path = ops[0].get("module_path", "")

    forward_param_names: Set[str] = set()
    module_param_shapes: Dict[str, str] = {}
    module_param_dtypes: Dict[str, str] = {}

    if module_tracker and module_path in module_tracker.path_to_module:
        mod = module_tracker.path_to_module[module_path]
        try:
            sig = inspect.signature(mod.forward)
            forward_param_names = {
                n for n in sig.parameters if n != "self"
            }
        except (ValueError, TypeError):
            pass
        for name, param in mod.named_parameters():
            module_param_shapes[name] = str(list(param.shape))
            module_param_dtypes[name] = str(param.dtype)

    def _is_internal(tid: int) -> bool:
        if graph is None:
            all_produced = {t for op in ops for t in op.get("_output_ids", [])}
            return tid in all_produced
        return graph.is_produced_by_any(tid, group_indices)

    def _all_internal_resolved() -> set:
        if graph is None:
            return {t for op in ops for t in op.get("_input_ids", [])}
        return {graph.resolve_id(t) for op in ops for t in op.get("_input_ids", [])}

    # ── External inputs ────────────────────────────────────────────────────
    seen_canonical_in: set = set()
    ext_inputs: List[Tuple[int, Dict, int]] = []

    for op in ops:
        for slot, tid in enumerate(op.get("_input_ids", [])):
            canonical = graph.resolve_id(tid) if graph else tid
            if canonical in seen_canonical_in:
                continue
            if not _is_internal(tid):
                seen_canonical_in.add(canonical)
                ext_inputs.append((tid, op, slot))

    # ── External outputs ───────────────────────────────────────────────────
    internal_consumed_resolved = _all_internal_resolved()
    seen_canonical_out: set = set()
    ext_outputs: List[Tuple[int, Dict, int]] = []

    for op in reversed(ops):
        for slot, tid in enumerate(op.get("_output_ids", [])):
            canonical = graph.resolve_id(tid) if graph else tid
            if canonical in seen_canonical_out:
                continue
            if canonical not in internal_consumed_resolved:
                seen_canonical_out.add(canonical)
                ext_outputs.insert(0, (tid, op, slot))

    # ── Classify external inputs ───────────────────────────────────────────
    input_shapes, input_dtypes, input_sources = [], [], []
    input_map: List[Dict] = []
    parameter_map: List[Dict] = []
    constant_map: List[Dict] = []

    max_inputs = len(forward_param_names) if forward_param_names else float("inf")
    seen_input_shapes: set = set()

    for sub_idx, (tid, op, slot) in enumerate(ext_inputs):
        shapes = _split_shape_list(op["input_shapes"])
        dtypes = op["input_dtypes"].split(", ")
        shape = shapes[slot] if slot < len(shapes) else "?"
        dtype = dtypes[slot] if slot < len(dtypes) else "?"
        short = _aten_short(op["aten_op"])

        kind = _classify_input(
            op, slot, shape, forward_param_names, module_param_shapes)

        if kind == "parameter":
            param_name = _guess_param_name(op, slot, module_param_shapes)
            if param_name not in [p["name"] for p in parameter_map]:
                parameter_map.append({
                    "name": param_name,
                    "kind": "parameter",
                    "sub_op": op["aten_op"],
                    "sub_op_seq_idx": ops.index(op),
                    "arg_slot": slot,
                    "shape": shape,
                    "dtype": dtype,
                })
            input_sources.append(f"{short}[{slot}](param:{param_name})")
        elif kind == "constant":
            const_key = (shape, dtype)
            if const_key not in seen_input_shapes:
                seen_input_shapes.add(const_key)
                constant_map.append({
                    "kind": "constant",
                    "sub_op": op["aten_op"],
                    "sub_op_seq_idx": ops.index(op),
                    "arg_slot": slot,
                    "shape": shape,
                    "dtype": dtype,
                })
            input_sources.append(f"{short}[{slot}](const)")
        else:
            input_key = shape
            if input_key not in seen_input_shapes and len(input_map) < max_inputs:
                seen_input_shapes.add(input_key)
                input_map.append({
                    "name": _guess_input_name(op, slot, forward_param_names),
                    "kind": "input",
                    "sub_op": op["aten_op"],
                    "sub_op_seq_idx": ops.index(op),
                    "arg_slot": slot,
                    "shape": shape,
                    "dtype": dtype,
                })
            input_sources.append(f"{short}[{slot}]")

        input_shapes.append(shape)
        input_dtypes.append(dtype)

    # ── Build output maps ──────────────────────────────────────────────────
    output_shapes, output_dtypes, output_sources = [], [], []
    output_map: List[Dict] = []

    for sub_idx, (tid, op, slot) in enumerate(ext_outputs):
        shapes = _split_shape_list(op["output_shapes"])
        dtypes = op["output_dtypes"].split(", ")
        shape = shapes[slot] if slot < len(shapes) else "?"
        dtype = dtypes[slot] if slot < len(dtypes) else "?"
        short = _aten_short(op["aten_op"])
        output_shapes.append(shape)
        output_dtypes.append(dtype)
        output_sources.append(f"{short}[{slot}]")
        output_map.append({
            "sub_op": op["aten_op"],
            "sub_op_seq_idx": ops.index(op),
            "out_slot": slot,
            "shape": shape,
            "dtype": dtype,
        })

    return {
        "fused_input_shapes": ", ".join(input_shapes),
        "fused_input_dtypes": ", ".join(input_dtypes),
        "fused_input_sources": " | ".join(input_sources),
        "fused_output_shapes": ", ".join(output_shapes),
        "fused_output_dtypes": ", ".join(output_dtypes),
        "fused_output_sources": " | ".join(output_sources),
        "_input_map": input_map,
        "_output_map": output_map,
        "_parameter_map": parameter_map,
        "_constant_map": constant_map,
    }


def _classify_input(
    op: Dict[str, Any],
    slot: int,
    shape: str,
    forward_param_names: Set[str],
    module_param_shapes: Dict[str, str],
) -> str:
    """Classify an external input as 'input', 'parameter', or 'constant'."""
    if shape in module_param_shapes.values():
        return "parameter"
    if shape == "[]" or shape == "[1]" or shape == "1":
        return "constant"
    if "Scalar" in op["aten_op"] and slot > 0:
        return "constant"
    return "input"


def _guess_param_name(
    op: Dict[str, Any],
    slot: int,
    module_param_shapes: Dict[str, str],
) -> str:
    shapes = _split_shape_list(op["input_shapes"])
    shape = shapes[slot] if slot < len(shapes) else "?"
    for name, pshape in module_param_shapes.items():
        if pshape == shape:
            return name
    return "weight"


def _guess_input_name(
    op: Dict[str, Any],
    slot: int,
    forward_param_names: Set[str],
) -> str:
    if forward_param_names:
        return next(iter(forward_param_names))
    return "hidden_states"


def _make_fused_entry(ops: List[Dict[str, Any]], tracker: ModuleTracker,
                      fusion_level: str = "leaf",
                      graph=None) -> Dict[str, Any]:
    first, last = ops[0], ops[-1]
    path = first["module_path"]
    module_key = _strip_layer_prefix(path) if path else ""

    aten_ops = list(dict.fromkeys(r["aten_op"] for r in ops))

    if module_key and len(ops) > 1:
        label = module_key
    elif module_key:
        label = module_key
    else:
        fn_parts = first["aten_op"].split(".")
        label = fn_parts[1] if len(fn_parts) >= 2 else first["aten_op"]

    io = _compute_fused_io(ops, graph, module_tracker=tracker)

    return {
        "fused_op": label,
        "module_path": path,
        "module_key": module_key,
        "fusion_level": fusion_level,
        "aten_ops": " \u2192 ".join(aten_ops),
        "num_sub_ops": len(ops),
        "layer": first["layer"],
        "input_shapes": first["input_shapes"],
        "input_dtypes": first["input_dtypes"],
        "output_shapes": last["output_shapes"],
        "output_dtypes": last["output_dtypes"],
        **io,
        "_children": ops,
    }


class FusionEngine:
    """Automatic fusion using module hierarchy."""

    def __init__(self, tracker: ModuleTracker, graph=None):
        self._tracker = tracker
        self._graph = graph

    def fuse(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups = self._pass1_leaf(records)
        for i, g in enumerate(groups):
            g["idx"] = i
            g.pop("_children", None)
        return groups

    def extract_specs(self, fused: List[Dict[str, Any]]) -> List[FusionSpec]:
        specs_by_key: Dict[Tuple[str, str], FusionSpec] = {}
        for g in fused:
            if g["num_sub_ops"] <= 1:
                continue
            key = (g["module_key"], g["fusion_level"])
            if key in specs_by_key:
                specs_by_key[key].occurrences += 1
            else:
                specs_by_key[key] = FusionSpec(
                    module_key=g["module_key"],
                    aten_op_sequence=g["aten_ops"].split(" \u2192 "),
                    num_sub_ops=g["num_sub_ops"],
                    fusion_level=g["fusion_level"],
                    example_module_path=g["module_path"],
                    occurrences=1,
                    fused_input_shapes=g.get("fused_input_shapes", ""),
                    fused_input_dtypes=g.get("fused_input_dtypes", ""),
                    fused_input_sources=g.get("fused_input_sources", ""),
                    fused_output_shapes=g.get("fused_output_shapes", ""),
                    fused_output_dtypes=g.get("fused_output_dtypes", ""),
                    fused_output_sources=g.get("fused_output_sources", ""),
                    input_map=g.get("_input_map", []),
                    output_map=g.get("_output_map", []),
                    parameter_map=g.get("_parameter_map", []),
                    constant_map=g.get("_constant_map", []),
                )
        return sorted(specs_by_key.values(), key=lambda s: -s.occurrences)

    def _pass1_leaf(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not records:
            return []
        groups = []
        current_group = [records[0]]
        for rec in records[1:]:
            same_path = rec["module_path"] == current_group[0]["module_path"]
            same_layer = rec["layer"] == current_group[0]["layer"]
            has_path = rec["module_path"] != ""
            if same_path and same_layer and has_path:
                current_group.append(rec)
            else:
                groups.append(_make_fused_entry(
                    current_group, self._tracker, "leaf", self._graph))
                current_group = [rec]
        groups.append(_make_fused_entry(
            current_group, self._tracker, "leaf", self._graph))
        return groups
