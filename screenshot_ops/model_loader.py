"""Load any HuggingFace causal LM for op-sequence tracing.

Supports:
  - HF Hub model IDs  (``deepseek-ai/DeepSeek-V3``, ``Qwen/Qwen3-8B``, …)
  - Local directories (``./hf_models/deepseek_v3``, …)
    * Standard model types (llama, qwen2, …): config.json only required
    * Custom architectures: uses local modeling files via importlib fallback
      (handles deepseek_v3, deepseek_v32 which have no auto_map)

Compatibility patches applied here:
  - Deprecated transformers internals (is_torch_fx_available, etc.)
  - torch.autocast with device_type='meta' (transformers 4.50+ RoPE issue)

No MoE patches needed — FakeTensorMode handles all ops natively.
Indexer patch still required — V3.2 modeling file has a shape bug.
MoE patch still required — ``.cpu().numpy()`` is not supported on FakeTensors.
"""
from __future__ import annotations

import importlib
import inspect
import json
import logging
import sys
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Compatibility patches ──────────────────────────────────────────────────────

_KNOWN_AUTOCAST_DEVICES = frozenset({"cpu", "cuda", "xpu", "hpu", "mps", "xla"})


def apply_compat_patches() -> None:
    """Add deprecated transformers attrs expected by older model code.

    Also patches ``torch.autocast`` to accept ``device_type='meta'``.
    transformers 4.50+ passes the tensor's device type to autocast inside
    RoPE; meta tensors surface as ``'meta'``, which torch 2.x rejects.
    We remap unknown device types to ``'cpu'`` (no-op for meta tensors).
    """
    try:
        import transformers.utils.import_utils as _iu
        if not hasattr(_iu, "is_torch_fx_available"):
            _iu.is_torch_fx_available = lambda: True
    except ImportError:
        pass
    try:
        import transformers.utils as _tu
        if not hasattr(_tu, "is_torch_fx_available"):
            _tu.is_torch_fx_available = lambda: True
    except ImportError:
        pass
    try:
        import transformers.pytorch_utils as _pu
        if not hasattr(_pu, "is_torch_greater_or_equal_than_1_13"):
            _pu.is_torch_greater_or_equal_than_1_13 = True
    except ImportError:
        pass

    if not getattr(torch.amp.autocast, "_meta_device_safe", False):
        _orig_init = torch.amp.autocast.__init__

        def _safe_init(self, device_type: str, *args: Any, **kwargs: Any) -> None:
            if device_type not in _KNOWN_AUTOCAST_DEVICES:
                device_type = "cpu"
            _orig_init(self, device_type, *args, **kwargs)

        torch.amp.autocast.__init__ = _safe_init  # type: ignore[method-assign]
        torch.amp.autocast._meta_device_safe = True  # type: ignore[attr-defined]


# ── Config normalization ───────────────────────────────────────────────────────

def _normalize_config(config: Any) -> None:
    """Apply generic compatibility fixes to a PretrainedConfig in-place."""
    rs = getattr(config, "rope_scaling", None)
    if isinstance(rs, dict) and "rope_type" in rs and "type" not in rs:
        rs["type"] = rs["rope_type"]
    config._attn_implementation = "eager"


# ── Generic MoE patch ──────────────────────────────────────────────────────────

def _is_moe_module(module: nn.Module) -> bool:
    experts = getattr(module, "experts", None)
    return (
        isinstance(experts, nn.ModuleList)
        and any(e is not None for e in experts)
        and not getattr(module, "_fake_patched", False)
    )


def _returns_router_tuple(mod: nn.Module) -> bool:
    try:
        src = inspect.getsource(type(mod).forward)
        if any(pat in src for pat in ("router_logits", "aux_loss",
                                      "return hidden_states,")):
            return True
    except Exception:
        pass
    return hasattr(mod, "router") and not hasattr(mod, "gate")


def patch_moe_for_fake(model: nn.Module) -> None:
    """Replace MoE forwards with a FakeTensor-safe simplified version.

    The original MoE forward calls ``.cpu().numpy()`` on routing indices,
    which is not supported on FakeTensors.  We replace it with a simplified
    version that only runs 1 expert + shared expert.
    """
    patched = 0
    for _, module in model.named_modules():
        if not _is_moe_module(module):
            continue
        module._fake_patched = True
        module.forward = _make_fake_moe_forward(module)
        patched += 1
    if patched:
        logger.info("Applied FakeTensor MoE patch to %d module(s).", patched)


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


# ── DeepSeek-V3.2 Indexer patch ───────────────────────────────────────────────

def _patch_indexer_forward(IndexerClass: type) -> None:
    """Patch DeepSeek-V3.2 Indexer to fix a shape bug in the modeling file.

    The V3.2 modeling file's own simplified forward contains a shape bug
    (``k_nope.transpose(1,2).transpose(2,3)`` on a 3D tensor fails).
    We always replace it with this correct implementation.
    """
    def _indexer_forward(self, x, qr, position_ids=None, attention_mask=None):
        bsz, seqlen, _ = x.size()
        nope_dim = self.index_head_dim - self.rope_head_dim

        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.index_n_heads, self.index_head_dim)
        _q_pe, q_nope = torch.split(q, [self.rope_head_dim, nope_dim], dim=-1)
        q_nope = q_nope.transpose(1, 2)

        k = self.wk(x)
        k = self.k_norm(k)
        _k_pe, k_nope = torch.split(k, [self.rope_head_dim, nope_dim], dim=-1)
        k_nope = k_nope.unsqueeze(1).expand(-1, self.index_n_heads, -1, -1)
        k_nope = k_nope.transpose(2, 3)

        scores = torch.matmul(q_nope, k_nope)
        if attention_mask is not None:
            scores = scores + attention_mask
        scores = torch.nn.functional.softmax(
            scores, dim=-1, dtype=torch.float32).to(x.dtype)
        topk_indices = scores.topk(min(self.index_topk, seqlen), dim=-1)[1]
        return topk_indices

    IndexerClass.forward = _indexer_forward


# ── Local custom-architecture loader ──────────────────────────────────────────

def _load_local_custom(model_dir: Path, num_hidden_layers: int):
    """Load a local model whose model_type is unknown to transformers.

    Used for deepseek_v3 / deepseek_v32 which ship their own modeling files
    but have no ``auto_map`` in config.json.

    Strategy:
    1. Import the local ``configuration_deepseek`` and ``modeling_deepseek``
       modules via importlib (temporarily adds the parent dir to sys.path).
    2. Instantiate the config class directly from config.json.
    3. Instantiate the first causal-LM class from the modeling module.
    """
    parent = str(model_dir.parent)
    pkg_name = model_dir.name

    init_path = model_dir / "__init__.py"
    created = not init_path.exists()
    if created:
        init_path.write_text("")

    if parent not in sys.path:
        sys.path.insert(0, parent)
        _added = True
    else:
        _added = False

    try:
        for mod_key in list(sys.modules.keys()):
            if mod_key.startswith(pkg_name + "."):
                del sys.modules[mod_key]
        if pkg_name in sys.modules:
            del sys.modules[pkg_name]

        cfg_mod = importlib.import_module(f"{pkg_name}.configuration_deepseek")
        mdl_mod = importlib.import_module(f"{pkg_name}.modeling_deepseek")
    finally:
        if created:
            init_path.unlink(missing_ok=True)
        if _added and parent in sys.path:
            sys.path.remove(parent)

    from transformers import PretrainedConfig
    ConfigClass = None
    for name in dir(cfg_mod):
        obj = getattr(cfg_mod, name)
        if (isinstance(obj, type) and issubclass(obj, PretrainedConfig)
                and obj is not PretrainedConfig):
            ConfigClass = obj
            break
    if ConfigClass is None:
        raise RuntimeError(f"No PretrainedConfig subclass found in {pkg_name}.configuration_deepseek")

    CausalLMClass = None
    for name in dir(mdl_mod):
        if "ForCausalLM" in name:
            obj = getattr(mdl_mod, name)
            if isinstance(obj, type) and issubclass(obj, nn.Module):
                CausalLMClass = obj
                break
    if CausalLMClass is None:
        raise RuntimeError(f"No ForCausalLM class found in {pkg_name}.modeling_deepseek")

    raw = json.loads((model_dir / "config.json").read_text())

    rs = raw.get("rope_scaling")
    if isinstance(rs, dict) and "rope_type" in rs and "type" not in rs:
        rs["type"] = rs["rope_type"]
    original_rope_scaling = raw.get("rope_scaling")

    default_keys = set(ConfigClass().__dict__.keys())
    config = ConfigClass(**{
        k: v for k, v in raw.items()
        if k in default_keys or k.startswith("_")
    })
    config._attn_implementation = "eager"
    if original_rope_scaling is not None:
        config.rope_scaling = dict(original_rope_scaling)

    config._full_num_hidden_layers = getattr(config, "num_hidden_layers", None)
    config.num_hidden_layers = num_hidden_layers

    logger.info("Instantiating %s on CPU (%d layers) …",
                type(config).__name__, num_hidden_layers)
    model = CausalLMClass(config)
    model.eval()
    return model, config


# ── Public API ────────────────────────────────────────────────────────────────

def load_model(
    model_id: str,
    num_hidden_layers: int = 4,
) -> Tuple[nn.Module, Any]:
    """Load any HF causal LM for op-sequence tracing.

    The model is instantiated on CPU.  During tracing, FakeTensorMode
    converts all tensors to FakeTensors (no real computation, no GPU needed).

    Parameters
    ----------
    model_id:
        HF Hub ID (``"deepseek-ai/DeepSeek-V3"``) **or** a local directory
        (``"./hf_models/deepseek_v3"``).
    num_hidden_layers:
        Number of transformer blocks to instantiate (2–4 is enough to see all
        distinct op patterns including dense + MoE layers).

    Returns
    -------
    (model, config)
        model  — on CPU, eval mode.
        config — ``config._full_num_hidden_layers`` stores the original depth.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    apply_compat_patches()

    logger.info("Loading config from %s …", model_id)

    model_dir = Path(model_id)
    config = None
    model = None

    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except (ValueError, KeyError) as exc:
        if model_dir.is_dir() and (model_dir / "modeling_deepseek.py").exists():
            logger.info("AutoConfig failed (%s) — using local modeling files.", exc)
            model, config = _load_local_custom(model_dir, num_hidden_layers)
        else:
            raise

    if model is None:
        config._full_num_hidden_layers = getattr(config, "num_hidden_layers", None)
        config.num_hidden_layers = num_hidden_layers
        _normalize_config(config)

        logger.info("Instantiating %s on CPU (%d layers) …",
                    type(config).__name__, num_hidden_layers)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model.eval()

    _patch_indexer(model)
    patch_moe_for_fake(model)

    return model, config


def _patch_indexer(model: nn.Module) -> None:
    for _, module in model.named_modules():
        cls_name = type(module).__name__
        if "Indexer" in cls_name and hasattr(module, "wq_b"):
            _patch_indexer_forward(type(module))
            logger.debug("Patched Indexer forward: %s", cls_name)
            break
