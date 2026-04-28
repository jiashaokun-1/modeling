"""Model compatibility patches for FakeTensorMode tracing.

Contains the minimal set of patches needed to run arbitrary HuggingFace
causal LMs through FakeTensorMode without allocating real memory.

Patch inventory
---------------
apply_compat_patches()
    Adds deprecated transformers attributes expected by older model code
    (is_torch_fx_available, is_torch_greater_or_equal_than_1_13).

patch_moe_for_fake(model)
    Replaces MoE module forwards with a simplified version that avoids
    .cpu().numpy() and torch.bincount() calls on routing indices, which
    crash on fake tensors.  Only applied when the standard heuristic
    identifies a module as MoE (has nn.ModuleList experts, not already
    patched).

patch_indexer_for_fake(model)
    Patches DeepSeek-V3.2 Indexer modules whose original forward contains
    a 3-D tensor .transpose(2,3) that is invalid under FakeTensorMode.
    The original modeling files from HF are kept untouched; this patch
    supplies a corrected forward at runtime only.

What is intentionally NOT patched
----------------------------------
* Autocast / dtype casting — FakeTensorMode handles these transparently.
* Meta-device specific hacks — superseded by FakeTensorMode.
"""
from __future__ import annotations

import inspect
import logging
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── Transformers compatibility ────────────────────────────────────────────────

def apply_compat_patches() -> None:
    """Apply all compatibility patches needed before model loading.

    Call order matters: version shims must be injected *before* any model file
    is imported, so that ``from transformers.xxx import yyy`` at module level
    finds the stub symbols even when the installed transformers version removed them.
    """
    # 1. Version shims (missing symbols injected into transformers sub-modules)
    from python.zrt.graph.compat import apply_version_shims
    apply_version_shims()

    # 2. Legacy attrs still expected by some older model files
    try:
        import transformers.utils.import_utils as _iu
        if not hasattr(_iu, "is_torch_fx_available"):
            _iu.is_torch_fx_available = lambda: True
    except ImportError:
        pass
    try:
        import transformers.pytorch_utils as _pu
        if not hasattr(_pu, "is_torch_greater_or_equal_than_1_13"):
            _pu.is_torch_greater_or_equal_than_1_13 = True
    except ImportError:
        pass


# ── MoE patch ─────────────────────────────────────────────────────────────────
# Many MoE implementations call .cpu().numpy() or torch.bincount() on routing
# indices, which crash on fake tensors.  This simplified forward exercises the
# gate + one expert + shared experts — enough to capture the full op pattern.

def is_moe_module(module: nn.Module) -> bool:
    """True if module looks like a MoE layer that needs patching."""
    experts = getattr(module, "experts", None)
    return (
        isinstance(experts, nn.ModuleList)
        and any(e is not None for e in experts)
        and not getattr(module, "_fake_patched", False)
    )


def _returns_router_tuple(mod: nn.Module) -> bool:
    """True if this MoE forward returns (hidden, router_logits) tuple."""
    try:
        src = inspect.getsource(type(mod).forward)
        if any(pat in src for pat in ("router_logits", "aux_loss",
                                      "return hidden_states,")):
            return True
    except Exception:
        pass
    return hasattr(mod, "router") and not hasattr(mod, "gate")


def _make_fake_moe_forward(mod: nn.Module):
    _tuple_return = _returns_router_tuple(mod)

    def _forward(hidden_states: torch.Tensor, *args: Any, **kwargs: Any):
        try:
            result = _impl(hidden_states)
            return (result, None) if _tuple_return else result
        except Exception as exc:
            logger.debug("Fake MoE forward error (%s) — returning identity.", exc)
            return (hidden_states, None) if _tuple_return else hidden_states

    def _impl(hidden_states: torch.Tensor) -> torch.Tensor:
        orig = hidden_states
        bs, seq, h = orig.shape
        flat = orig.reshape(bs * seq, h)

        gate_weight: Optional[torch.Tensor] = None
        gate = getattr(mod, "gate", None)
        if gate is not None and callable(gate):
            try:
                gate_out = gate(orig)
                if isinstance(gate_out, (tuple, list)):
                    gate_weight = gate_out[1]
                else:
                    gate_weight = torch.softmax(gate_out.float(), dim=-1)[:, :1]
            except Exception:
                try:
                    gate_out = gate(flat)
                    if isinstance(gate_out, (tuple, list)):
                        gate_weight = gate_out[1]
                    else:
                        gate_weight = torch.softmax(gate_out.float(), dim=-1)[:, :1]
                except Exception as exc:
                    logger.debug("Gate forward failed (%s).", exc)

        first_expert = next((e for e in mod.experts if e is not None), None)
        if first_expert is None:
            return orig

        try:
            expert_out = first_expert(flat)
        except Exception:
            expert_out = first_expert(orig).reshape(bs * seq, -1)

        y = (expert_out * gate_weight[:, :1]
             if gate_weight is not None else expert_out)
        try:
            y = y.reshape(bs, seq, -1)
        except Exception:
            y = orig

        for attr in ("shared_experts", "shared_expert"):
            shared = getattr(mod, attr, None)
            if shared is not None and callable(shared):
                try:
                    y = y + shared(orig)
                except Exception as exc:
                    logger.debug("Shared expert failed (%s).", exc)
                break

        return y

    return _forward


def patch_moe_for_fake(model: nn.Module) -> None:
    """Replace MoE forwards with a fake-tensor-safe simplified version."""
    patched = 0
    for _, module in model.named_modules():
        if not is_moe_module(module):
            continue
        module._fake_patched = True
        module.forward = _make_fake_moe_forward(module)
        patched += 1
    if patched:
        logger.info("Applied fake-tensor MoE patch to %d module(s).", patched)


# ── DeepSeek-V3.2 Indexer patch ──────────────────────────────────────────────
# The Indexer forward in the original HF modeling file (kept unmodified) uses
# k_nope.transpose(1, 2).transpose(2, 3) on a 3-D tensor, which is invalid.
# We supply a corrected forward at runtime without touching the model files.

def _make_indexer_forward_fake(IndexerClass: type) -> None:
    """Replace Indexer.forward with a FakeTensorMode-compatible version."""

    def _forward(self, x, qr, position_ids=None, attention_mask=None):
        import torch.nn.functional as F
        bsz, seqlen, _ = x.size()
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.index_n_heads, self.index_head_dim)
        # Split into RoPE and NoPE parts (mirrors the original design intent)
        q_pe, q_nope = torch.split(
            q, [self.rope_head_dim, self.index_head_dim - self.rope_head_dim], dim=-1)
        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_head_dim, self.index_head_dim - self.rope_head_dim], dim=-1)
        # q_nope: (bsz, seqlen, n_heads, nope_dim) -> (bsz, n_heads, seqlen, nope_dim)
        q_nope = q_nope.transpose(1, 2)
        # k_nope: (bsz, seqlen, nope_dim) -> broadcast across heads
        k_nope = k_nope.unsqueeze(1).expand(-1, self.index_n_heads, -1, -1)
        scores = torch.matmul(q_nope, k_nope.transpose(2, 3))
        if attention_mask is not None:
            scores = scores + attention_mask
        scores = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        topk_indices = scores.topk(min(self.index_topk, seqlen), dim=-1)[1]
        return topk_indices

    IndexerClass.forward = _forward


def patch_indexer_for_fake(model: nn.Module) -> None:
    """Patch DeepSeek-V3.2 Indexer modules for FakeTensorMode compatibility.

    The original HF modeling files are not modified.  The fix is applied
    in-memory at runtime: any module whose class name contains 'Indexer'
    and which has the expected MLA attributes (wq_b, index_n_heads) gets
    a corrected forward method.
    """
    patched_classes: set = set()
    for _, module in model.named_modules():
        cls = type(module)
        if cls in patched_classes:
            continue
        if "Indexer" in cls.__name__ and hasattr(module, "wq_b") and hasattr(module, "index_n_heads"):
            _make_indexer_forward_fake(cls)
            patched_classes.add(cls)
            logger.debug("Patched Indexer class: %s", cls.__name__)
    if patched_classes:
        logger.info("Applied Indexer patch to: %s",
                    ", ".join(c.__name__ for c in patched_classes))


# ── DeepSeek-V4 Hyper-Connections capture patch ──────────────────────────────
# Block.hc_pre / Block.hc_post are *methods* on the inference Block class.  The
# graph-capture pipeline relies on ``ModuleTracker`` to delimit semantic units
# by ``nn.Module`` boundaries, so methods are invisible to it.  We do not edit
# the upstream inference/model.py; instead, at load time we attach 4 wrapper
# nn.Modules to each Block (and 1 to ParallelHead for the MTP head path) and
# rebind .forward on the *class* to dispatch through them.  After this patch,
# ModuleTracker reports paths like ``transformer.layers.0.hc_pre_attn``.


class _HCBoundMethodModule(nn.Module):
    """Base wrapper whose forward delegates to a method bound on a parent module.

    Subclasses (HCPreAttn / HCPostAttn / HCPreFfn / HCPostFfn / HCHead) carry
    distinct class names so ``fusion_rules.SEMANTIC_LABELS`` can match them
    individually.  The parent is held by ``weakref`` so this child does not
    double-register the parent's params in ``state_dict()``.
    """

    def __init__(self, parent: nn.Module, method_name: str, *param_names: str):
        super().__init__()
        import weakref
        self._parent_ref = weakref.ref(parent)
        self._method_name = method_name
        self._param_names = param_names

    def forward(self, *runtime_args):
        parent = self._parent_ref()
        if parent is None:
            raise RuntimeError(f"HC wrapper lost reference to parent before {self._method_name}")
        method = getattr(type(parent), self._method_name)
        bound_params = tuple(getattr(parent, n) for n in self._param_names)
        return method(parent, *runtime_args, *bound_params)


class HCPreAttn(_HCBoundMethodModule): pass
class HCPostAttn(_HCBoundMethodModule): pass
class HCPreFfn(_HCBoundMethodModule): pass
class HCPostFfn(_HCBoundMethodModule): pass
class HCHead(_HCBoundMethodModule): pass


def _block_forward_with_hc_modules(self, x, start_pos, input_ids):
    """Replacement Block.forward routing HC through child nn.Modules."""
    residual = x
    x, post, comb = self.hc_pre_attn(x)
    x = self.attn_norm(x)
    x = self.attn(x, start_pos)
    x = self.hc_post_attn(x, residual, post, comb)

    residual = x
    x, post, comb = self.hc_pre_ffn(x)
    x = self.ffn_norm(x)
    x = self.ffn(x, input_ids)
    x = self.hc_post_ffn(x, residual, post, comb)
    return x


def _head_forward_with_hc_module(self, x, hc_fn, hc_scale, hc_base, norm):
    """Replacement ParallelHead.forward routing hc_head through a child Module.

    Cannot bind hc_fn / hc_scale / hc_base on the head itself: a single
    ParallelHead is shared between the main Transformer (uses
    ``transformer.hc_head_*``) and each MTPBlock (uses ``mtp[i].hc_head_*``).
    The wrapper module therefore takes them as runtime args.
    """
    import torch.distributed as dist
    x = self.hc_head_module(x, hc_fn, hc_scale, hc_base)
    logits = self.get_logits(norm(x))
    if dist.is_initialized() and dist.get_world_size() > 1:
        all_logits = [torch.empty_like(logits) for _ in range(dist.get_world_size())]
        dist.all_gather(all_logits, logits)
        logits = torch.cat(all_logits, dim=-1)
    return logits


def _attach_hc_modules_to_block(block: nn.Module) -> None:
    """Attach 4 HC wrapper modules to a Block (covers MTPBlock via inheritance)."""
    if getattr(block, "_hc_patched", False):
        return
    # hc_pre takes only x at runtime; hc_*_fn / hc_*_scale / hc_*_base are bound
    # to the parent block.
    block.hc_pre_attn = HCPreAttn(
        block, "hc_pre", "hc_attn_fn", "hc_attn_scale", "hc_attn_base"
    )
    block.hc_pre_ffn = HCPreFfn(
        block, "hc_pre", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base"
    )
    # hc_post takes (x, residual, post, comb) — all runtime.
    block.hc_post_attn = HCPostAttn(block, "hc_post")
    block.hc_post_ffn = HCPostFfn(block, "hc_post")
    block._hc_patched = True


def _attach_hc_module_to_head(head: nn.Module) -> None:
    """Attach 1 HC wrapper module to a ParallelHead."""
    if getattr(head, "_hc_patched", False):
        return
    # hc_head takes (x, hc_fn, hc_scale, hc_base) — all runtime; the head
    # is shared between Transformer and MTPBlocks, each supplying their own.
    head.hc_head_module = HCHead(head, "hc_head")
    head._hc_patched = True


def patch_hc_for_capture(model: nn.Module) -> None:
    """Inject HC wrapper nn.Modules so ModuleTracker sees HC boundaries.

    Each Block / MTPBlock gets 4 child modules:
      - ``hc_pre_attn`` / ``hc_post_attn`` (around the attn sub-layer)
      - ``hc_pre_ffn``  / ``hc_post_ffn``  (around the ffn  sub-layer)

    Each ParallelHead gets 1 child module ``hc_head_module``, used by both the
    main Transformer's final head call and any MTPBlock's head call.

    ``Block.forward`` and ``ParallelHead.forward`` are rebound on the *class*
    so the change carries through the inheritance chain (MTPBlock inherits
    Block.forward via ``super().forward`` in its own forward).
    """
    block_classes: set[type] = set()
    head_classes: set[type] = set()

    for _, module in model.named_modules():
        cls = type(module)
        cls_name = cls.__name__
        if cls_name in ("Block", "MTPBlock") and hasattr(module, "hc_attn_fn"):
            _attach_hc_modules_to_block(module)
            # Only rebind the *defining* class for hc_pre/hc_post forward.
            # MTPBlock keeps its own forward (which calls super().forward).
            if cls_name == "Block":
                block_classes.add(cls)
        elif cls_name == "ParallelHead" and hasattr(module, "hc_head"):
            _attach_hc_module_to_head(module)
            head_classes.add(cls)

    for cls in block_classes:
        cls.forward = _block_forward_with_hc_modules
    for cls in head_classes:
        cls.forward = _head_forward_with_hc_module

    if block_classes or head_classes:
        logger.info(
            "Applied HC capture patch — Block classes: %s; Head classes: %s",
            ", ".join(c.__name__ for c in block_classes) or "(none)",
            ", ".join(c.__name__ for c in head_classes) or "(none)",
        )


# Backward-compatible aliases
patch_moe_for_meta = patch_moe_for_fake
_is_moe_module = is_moe_module
