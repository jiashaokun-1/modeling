"""Load DeepSeek models onto meta device with compatibility patches."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn as nn


def load_model(model_dir: Path, num_hidden_layers: int = 2) -> Tuple[nn.Module, Any]:
    """Load DeepSeek model from local config onto meta device."""
    _patch_transformers_compat()

    config_mod, modeling_mod = _import_model_package(model_dir)
    
    config_path = model_dir / "config.json"
    raw = json.loads(config_path.read_text())
    model_type = raw.get("model_type", "deepseek_v3")
    
    if model_type == "deepseek_v32":
        return _load_v32(config_mod, modeling_mod, model_dir, num_hidden_layers)
    else:
        return _load_v3(config_mod, modeling_mod, model_dir, num_hidden_layers)


def _load_v32(config_mod, modeling_mod, model_dir: Path, num_hidden_layers: int):
    """Load DeepSeek-V3.2 model."""
    DeepseekV32Config = config_mod.DeepseekV32Config
    DeepseekV32ForCausalLM = modeling_mod.DeepseekV32ForCausalLM

    config = _build_config(DeepseekV32Config, model_dir, num_hidden_layers)

    with torch.device("meta"):
        model = DeepseekV32ForCausalLM(config)
    model.eval()

    _patch_moe_forward(modeling_mod.DeepseekV32MoE)
    _patch_indexer_forward(modeling_mod.DeepseekV32Indexer)

    return model, config


def _load_v3(config_mod, modeling_mod, model_dir: Path, num_hidden_layers: int):
    """Load DeepSeek-V3 model."""
    DeepseekV3Config = config_mod.DeepseekV3Config
    DeepseekV3ForCausalLM = modeling_mod.DeepseekV3ForCausalLM

    config = _build_config(DeepseekV3Config, model_dir, num_hidden_layers)

    with torch.device("meta"):
        model = DeepseekV3ForCausalLM(config)
    model.eval()

    _patch_moe_forward(modeling_mod.DeepseekV3MoE)

    return model, config


def _patch_transformers_compat():
    import transformers.utils.import_utils as _iu
    if not hasattr(_iu, "is_torch_fx_available"):
        _iu.is_torch_fx_available = lambda: True
    import transformers.utils as _tu
    if not hasattr(_tu, "is_torch_fx_available"):
        _tu.is_torch_fx_available = lambda: True
    import transformers.pytorch_utils as _pu
    if not hasattr(_pu, "is_torch_greater_or_equal_than_1_13"):
        _pu.is_torch_greater_or_equal_than_1_13 = True


def _import_model_package(model_dir: Path):
    sys.path.insert(0, str(model_dir.parent))
    pkg_name = model_dir.name
    init_path = model_dir / "__init__.py"
    created_init = False
    if not init_path.exists():
        init_path.write_text("")
        created_init = True
    try:
        config_mod = importlib.import_module(f"{pkg_name}.configuration_deepseek")
        modeling_mod = importlib.import_module(f"{pkg_name}.modeling_deepseek")
    finally:
        if created_init:
            init_path.unlink(missing_ok=True)
    return config_mod, modeling_mod


def _build_config(ConfigClass, model_dir: Path, num_hidden_layers: int):
    config_path = model_dir / "config.json"
    raw = json.loads(config_path.read_text())
    raw["num_hidden_layers"] = num_hidden_layers
    raw["_attn_implementation"] = "eager"

    if "rope_scaling" in raw and isinstance(raw["rope_scaling"], dict):
        rs = raw["rope_scaling"]
        if "rope_type" in rs and "type" not in rs:
            rs["type"] = rs["rope_type"]

    original_rope_scaling = raw.get("rope_scaling")

    default_keys = ConfigClass().__dict__
    config = ConfigClass(**{
        k: v for k, v in raw.items()
        if k in default_keys or k.startswith("_")
    })
    config._attn_implementation = "eager"

    if original_rope_scaling is not None:
        config.rope_scaling = dict(original_rope_scaling)

    return config


def _patch_moe_forward(MoEClass):
    def _moe_forward_meta(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        bsz, seq_len, h = orig_shape

        topk_idx, topk_weight = self.gate(hidden_states)

        flat_hidden = hidden_states.view(-1, h)
        expert_out = self.experts[0](flat_hidden)

        y = expert_out * topk_weight[:, :1]
        y = y.view(*orig_shape)

        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    MoEClass.forward = _moe_forward_meta


def _patch_indexer_forward(IndexerClass):
    """Patch Indexer to work on meta device (simplified forward)."""
    def _indexer_forward_meta(self, x, qr, position_ids=None, attention_mask=None):
        bsz, seqlen, _ = x.size()
        
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.index_n_heads, self.index_head_dim)
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.index_head_dim - self.rope_head_dim], dim=-1)
        
        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.index_head_dim - self.rope_head_dim], dim=-1)
        
        weights = self.weights_proj(x)
        
        # q_nope: [bsz, seqlen, n_heads, head_dim] -> [bsz, n_heads, seqlen, head_dim]
        q_nope = q_nope.transpose(1, 2)
        # k_nope: [bsz, seqlen, head_dim] -> [bsz, 1, seqlen, head_dim] -> broadcast to heads
        k_nope = k_nope.unsqueeze(1).expand(-1, self.index_n_heads, -1, -1)
        
        scores = torch.matmul(q_nope, k_nope.transpose(2, 3))
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        scores = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        
        topk_indices = scores.topk(min(self.index_topk, seqlen), dim=-1)[1]
        return topk_indices

    IndexerClass.forward = _indexer_forward_meta
