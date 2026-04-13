"""Capture the operator sequence of DeepSeek-V3 via dispatch-level tracing
and write the results to an Excel file.

Inspired by xpu_simulator/frontend/dispatch_extractor.py — we use
TorchDispatchMode to intercept every aten op that fires during a forward
pass on meta tensors, then dump the ordered sequence to Excel.
"""
from __future__ import annotations

import sys
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "hf_models" / "deepseek_v3"
OUTPUT_FILE = Path(__file__).parent / "deepseek_v3_ops.xlsx"

# ── Helpers ──────────────────────────────────────────────────────────────────

def _shape_str(t: torch.Tensor) -> str:
    return str(list(t.shape))


def _collect_tensors(args: tuple, kwargs: dict) -> List[torch.Tensor]:
    tensors = []
    for a in args:
        if isinstance(a, torch.Tensor):
            tensors.append(a)
        elif isinstance(a, (list, tuple)):
            for item in a:
                if isinstance(item, torch.Tensor):
                    tensors.append(item)
    for v in kwargs.values():
        if isinstance(v, torch.Tensor):
            tensors.append(v)
        elif isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, torch.Tensor):
                    tensors.append(item)
    return tensors


def _collect_output_tensors(out: Any) -> List[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        return [item for item in out if isinstance(item, torch.Tensor)]
    return []


# ── Zero-cost ops to skip (from DispatchExtractor) ──────────────────────────

_SKIP_OPS = {
    "aten.detach.default", "aten.alias.default", "aten.t.default",
    "aten.view.default", "aten._unsafe_view.default",
    "aten.expand.default", "aten.contiguous.default",
    "aten.slice.Tensor", "aten.select.int",
    "aten.unsqueeze.default", "aten.squeeze.dim",
    "aten.split.Tensor", "aten.split_with_sizes.default",
    "aten.permute.default", "aten.reshape.default",
    "aten.clone.default",
    "aten.arange.default", "aten.arange.start",
    "aten.ones.default", "aten.zeros.default",
    "aten.full.default", "aten.scalar_tensor.default",
    "aten.tril.default", "aten.triu.default",
    "aten.empty_like.default", "aten.zeros_like.default",
    "aten.index_put_.default", "aten.index_put.default",
    "aten.scatter_.value", "aten.scatter_.src",
    "aten.histc.default", "aten.cumsum.default",
    "aten.bitwise_not.default",
    "aten.sort.default", "aten.sort.stable",
}


# ── Module tracker ──────────────────────────────────────────────────────────

class _ModuleTracker:
    """Track which nn.Module is currently executing via forward hooks."""

    def __init__(self, root: nn.Module):
        self._stack: List[str] = []
        self._handles: List[Any] = []
        self._install(root, "")

    def _install(self, module: nn.Module, prefix: str):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            def _pre_hook(m, inp, _fn=full_name):
                self._stack.append(_fn)

            def _post_hook(m, inp, out, _fn=full_name):
                if self._stack and self._stack[-1] == _fn:
                    self._stack.pop()

            h1 = child.register_forward_pre_hook(_pre_hook)
            h2 = child.register_forward_hook(_post_hook)
            self._handles.extend([h1, h2])
            self._install(child, full_name)

    @property
    def current_module(self) -> str:
        return self._stack[-1] if self._stack else ""

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ── Recording dispatch ──────────────────────────────────────────────────────

class _RecordingDispatch(TorchDispatchMode):
    """Intercept every aten op and record its metadata."""

    def __init__(self, module_tracker: Optional[_ModuleTracker] = None,
                 skip_reshapes: bool = True):
        super().__init__()
        self.records: List[Dict[str, Any]] = []
        self._module_tracker = module_tracker
        self._skip_reshapes = skip_reshapes

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)

        func_name = str(func.overloadpacket) + "." + func._overloadname

        if self._skip_reshapes and func_name in _SKIP_OPS:
            return out

        input_tensors = _collect_tensors(args, kwargs)
        output_tensors = _collect_output_tensors(out)

        module_path = ""
        if self._module_tracker:
            module_path = self._module_tracker.current_module

        input_shapes = [_shape_str(t) for t in input_tensors]
        input_dtypes = [str(t.dtype) for t in input_tensors]
        output_shapes = [_shape_str(t) for t in output_tensors]
        output_dtypes = [str(t.dtype) for t in output_tensors]

        self.records.append({
            "idx": len(self.records),
            "aten_op": func_name,
            "module_path": module_path,
            "layer": _extract_layer_idx(module_path),
            "component": _classify_component(module_path, func_name),
            "input_shapes": ", ".join(input_shapes),
            "input_dtypes": ", ".join(input_dtypes),
            "output_shapes": ", ".join(output_shapes),
            "output_dtypes": ", ".join(output_dtypes),
            "num_inputs": len(input_tensors),
            "num_outputs": len(output_tensors),
        })

        return out


# ── Naming / classification helpers ─────────────────────────────────────────

def _extract_layer_idx(module_path: str) -> str:
    parts = module_path.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return str(int(parts[i + 1]))
            except ValueError:
                pass
    return ""


def _classify_component(module_path: str, func_name: str) -> str:
    """Map a module path to a human-readable component category."""
    s = module_path.lower()

    # Aten op short name
    fn_parts = func_name.split(".")
    op_short = fn_parts[1] if len(fn_parts) >= 2 else func_name

    if "input_layernorm" in s:
        return "attn_norm"
    if "post_attention_layernorm" in s:
        return "ffn_norm"

    # MLA projections
    if "q_a_proj" in s:
        return "attn.q_a_proj"
    if "q_a_layernorm" in s:
        return "attn.q_norm"
    if "q_b_proj" in s:
        return "attn.q_b_proj"
    if "q_proj" in s:
        return "attn.q_proj"
    if "kv_a_proj" in s:
        return "attn.kv_a_proj"
    if "kv_a_layernorm" in s:
        return "attn.kv_norm"
    if "kv_b_proj" in s:
        return "attn.kv_b_proj"
    if "o_proj" in s:
        return "attn.o_proj"

    # Attention compute
    if "self_attn" in s or "attn" in s:
        if "rotary" in s:
            return "attn.rope"
        if op_short in ("matmul", "mm", "bmm"):
            return "attn.score"
        if "softmax" in op_short or "safe_softmax" in op_short:
            return "attn.softmax"
        return f"attn.{op_short}"

    # MoE
    if "gate" in s and "experts" not in s and "up" not in s and ("moe" in s or "mlp" in s):
        return f"moe.gate.{op_short}"
    if "shared_expert" in s or ("shared" in s and "mlp" in s):
        if "gate_proj" in s:
            return "moe.shared.gate_proj"
        if "up_proj" in s:
            return "moe.shared.up_proj"
        if "down_proj" in s:
            return "moe.shared.down_proj"
        return f"moe.shared.{op_short}"
    if "experts" in s:
        return f"moe.experts.{op_short}"

    # Dense FFN
    if "mlp" in s or "moe" in s:
        if "gate_proj" in s:
            return "ffn.gate_proj"
        if "up_proj" in s:
            return "ffn.up_proj"
        if "down_proj" in s:
            return "ffn.down_proj"
        if op_short == "silu":
            return "ffn.silu"
        if op_short == "mul":
            return "ffn.mul"
        return f"ffn.{op_short}"

    # Embedding / final norm / lm_head
    if "embed" in s:
        return "embedding"
    if "norm" in s:
        return "final_norm"
    if "lm_head" in s:
        return "lm_head"

    return op_short


# ── Model loading ───────────────────────────────────────────────────────────

def load_model(model_dir: Path, num_hidden_layers: int = 2) -> nn.Module:
    """Load DeepSeek-V3 from local config onto meta device.

    Uses a small layer count by default for fast extraction.
    """
    # Patch away removed/renamed helpers so the older modeling file can import
    import transformers.utils.import_utils as _iu
    if not hasattr(_iu, "is_torch_fx_available"):
        _iu.is_torch_fx_available = lambda: True
    import transformers.utils as _tu
    if not hasattr(_tu, "is_torch_fx_available"):
        _tu.is_torch_fx_available = lambda: True
    import transformers.pytorch_utils as _pu
    if not hasattr(_pu, "is_torch_greater_or_equal_than_1_13"):
        _pu.is_torch_greater_or_equal_than_1_13 = True
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter as _AMC
    if not hasattr(_AMC, "_prepare_4d_causal_attention_mask"):
        pass  # already available via function import

    # Import as a proper package so relative imports work
    sys.path.insert(0, str(model_dir.parent))
    import importlib
    pkg_name = model_dir.name
    # Ensure the directory is treated as a package
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
    DeepseekV3Config = config_mod.DeepseekV3Config
    DeepseekV3ForCausalLM = modeling_mod.DeepseekV3ForCausalLM

    config_path = model_dir / "config.json"
    raw = json.loads(config_path.read_text())
    raw["num_hidden_layers"] = num_hidden_layers
    # Force eager attention (no flash_attn dependency needed on meta device)
    raw["_attn_implementation"] = "eager"

    # Newer transformers renames rope_scaling["type"] to ["rope_type"];
    # restore the "type" key that the modeling code expects.
    if "rope_scaling" in raw and isinstance(raw["rope_scaling"], dict):
        rs = raw["rope_scaling"]
        if "rope_type" in rs and "type" not in rs:
            rs["type"] = rs["rope_type"]

    # Save rope_scaling before PretrainedConfig.__init__ can mangle it
    original_rope_scaling = raw.get("rope_scaling")

    default_keys = DeepseekV3Config().__dict__
    config = DeepseekV3Config(**{
        k: v for k, v in raw.items()
        if k in default_keys or k.startswith("_")
    })
    config._attn_implementation = "eager"

    # Restore the original rope_scaling dict (PretrainedConfig may have overwritten it)
    if original_rope_scaling is not None:
        config.rope_scaling = dict(original_rope_scaling)

    with torch.device("meta"):
        model = DeepseekV3ForCausalLM(config)
    model.eval()

    # Monkey-patch MoE forward: the real moe_infer calls .cpu().numpy()
    # which fails on meta tensors.  We replace it with a simplified version
    # that runs one representative expert + shared expert, which captures the
    # correct operator sequence without needing real routing data.
    DeepseekV3MoE = modeling_mod.DeepseekV3MoE

    def _moe_forward_meta(self, hidden_states):
        """Simplified MoE forward that works on meta tensors."""
        identity = hidden_states
        orig_shape = hidden_states.shape
        bsz, seq_len, h = orig_shape

        # Run the gate to capture gate ops (linear + sigmoid + topk)
        topk_idx, topk_weight = self.gate(hidden_states)
        # topk_weight: (bsz*seq_len, top_k), topk_idx: (bsz*seq_len, top_k)

        # Run one representative expert to capture expert MLP ops
        flat_hidden = hidden_states.view(-1, h)
        expert_out = self.experts[0](flat_hidden)

        # Simulate weighted sum: mul
        y = expert_out * topk_weight[:, :1]

        y = y.view(*orig_shape)

        # Shared expert
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    DeepseekV3MoE.forward = _moe_forward_meta

    return model, config


# ── Excel writing ───────────────────────────────────────────────────────────

# Color palette for component categories
_FILL_COLORS = {
    "attn_norm":    "E8F5E9",  # light green
    "ffn_norm":     "E8F5E9",
    "final_norm":   "E8F5E9",
    "attn.":        "E3F2FD",  # light blue
    "moe.gate":     "FFF3E0",  # light orange
    "moe.shared":   "FFF8E1",  # light yellow
    "moe.experts":  "FCE4EC",  # light pink
    "ffn.":         "F3E5F5",  # light purple
    "embedding":    "ECEFF1",  # light grey
    "lm_head":      "ECEFF1",
}


def _get_fill(component: str) -> Optional[PatternFill]:
    for prefix, color in _FILL_COLORS.items():
        if component.startswith(prefix):
            return PatternFill(start_color=color, end_color=color, fill_type="solid")
    return None


def write_excel(records: List[Dict[str, Any]], output_path: Path,
                config_summary: Dict[str, Any]):
    """Write operator records to a formatted Excel workbook."""
    wb = openpyxl.Workbook()

    # ── Sheet 1: Model Config ─────────────────────────────────────────────
    ws_cfg = wb.active
    ws_cfg.title = "Model Config"
    header_font = Font(bold=True, size=12)
    ws_cfg.append(["Parameter", "Value"])
    ws_cfg["A1"].font = header_font
    ws_cfg["B1"].font = header_font
    for key, val in config_summary.items():
        ws_cfg.append([key, str(val)])
    ws_cfg.column_dimensions["A"].width = 30
    ws_cfg.column_dimensions["B"].width = 40

    # ── Sheet 2: Operator Sequence ────────────────────────────────────────
    ws = wb.create_sheet("Operator Sequence")

    columns = [
        ("Index", 8),
        ("Aten Op", 35),
        ("Module Path", 55),
        ("Layer", 7),
        ("Component", 25),
        ("Input Shapes", 50),
        ("Input Dtypes", 30),
        ("Output Shapes", 50),
        ("Output Dtypes", 30),
    ]

    header_fill = PatternFill(start_color="263238", end_color="263238", fill_type="solid")
    header_font_white = Font(bold=True, color="FFFFFF", size=11)
    thin_border = Border(
        bottom=Side(style="thin", color="BDBDBD"),
    )

    for col_idx, (col_name, width) in enumerate(columns, 1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font = header_font_white
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
        ws.column_dimensions[get_column_letter(col_idx)].width = width

    for row_idx, rec in enumerate(records, 2):
        values = [
            rec["idx"],
            rec["aten_op"],
            rec["module_path"],
            rec["layer"],
            rec["component"],
            rec["input_shapes"],
            rec["input_dtypes"],
            rec["output_shapes"],
            rec["output_dtypes"],
        ]
        fill = _get_fill(rec["component"])
        for col_idx, val in enumerate(values, 1):
            cell = ws.cell(row=row_idx, column=col_idx, value=val)
            cell.border = thin_border
            if fill:
                cell.fill = fill
            if col_idx == 1:
                cell.alignment = Alignment(horizontal="center")

    ws.auto_filter.ref = f"A1:{get_column_letter(len(columns))}{len(records) + 1}"
    ws.freeze_panes = "A2"

    # ── Sheet 3: Summary by Component ─────────────────────────────────────
    ws_sum = wb.create_sheet("Summary")
    ws_sum.append(["Component", "Count"])
    ws_sum["A1"].font = header_font
    ws_sum["B1"].font = header_font

    from collections import Counter
    component_counts = Counter(r["component"] for r in records)
    for comp, count in sorted(component_counts.items()):
        ws_sum.append([comp, count])
    ws_sum.column_dimensions["A"].width = 30
    ws_sum.column_dimensions["B"].width = 10

    # ── Sheet 4: Summary by Layer ─────────────────────────────────────────
    ws_layer = wb.create_sheet("By Layer")
    ws_layer.append(["Layer", "Op Count", "Components"])
    ws_layer["A1"].font = header_font
    ws_layer["B1"].font = header_font
    ws_layer["C1"].font = header_font

    from collections import defaultdict
    layer_info = defaultdict(lambda: {"count": 0, "components": set()})
    for r in records:
        layer = r["layer"] or "non-layer"
        layer_info[layer]["count"] += 1
        layer_info[layer]["components"].add(r["component"])

    for layer in sorted(layer_info.keys(), key=lambda x: (x == "non-layer", x)):
        info = layer_info[layer]
        ws_layer.append([layer, info["count"], ", ".join(sorted(info["components"]))])
    ws_layer.column_dimensions["A"].width = 10
    ws_layer.column_dimensions["B"].width = 10
    ws_layer.column_dimensions["C"].width = 80

    wb.save(output_path)
    logger.info("Saved %d ops to %s", len(records), output_path)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    # Use 2 layers to capture both a dense layer (layer 0,1,2 are dense) and
    # an MoE layer. With first_k_dense_replace=3, layers 0..2 are dense and
    # layers 3+ are MoE. To capture both patterns we need at least 4 layers.
    num_layers = 4
    batch_size = 1
    seq_len = 128

    logger.info("Loading DeepSeek-V3 model from %s (%d layers)...", MODEL_DIR, num_layers)
    model, config = load_model(MODEL_DIR, num_hidden_layers=num_layers)

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="meta")
    position_ids = torch.arange(seq_len, device="meta").unsqueeze(0)

    # Build the 4D causal attention mask on meta device
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device="meta")
    mask = torch.triu(mask, diagonal=1)

    logger.info("Tracing forward pass (batch=%d, seq=%d)...", batch_size, seq_len)
    tracker = _ModuleTracker(model)
    recorder = _RecordingDispatch(module_tracker=tracker, skip_reshapes=True)
    try:
        with recorder, torch.no_grad():
            model(input_ids=input_ids, attention_mask=mask, position_ids=position_ids,
                  use_cache=False)
    finally:
        tracker.remove()

    logger.info("Captured %d ops (after filtering zero-cost reshapes)", len(recorder.records))

    config_summary = {
        "model_type": "deepseek_v3",
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "moe_intermediate_size": config.moe_intermediate_size,
        "num_hidden_layers (full)": 61,
        "num_hidden_layers (traced)": num_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "q_lora_rank": config.q_lora_rank,
        "kv_lora_rank": config.kv_lora_rank,
        "qk_nope_head_dim": config.qk_nope_head_dim,
        "qk_rope_head_dim": config.qk_rope_head_dim,
        "v_head_dim": config.v_head_dim,
        "n_routed_experts": config.n_routed_experts,
        "n_shared_experts": config.n_shared_experts,
        "num_experts_per_tok": config.num_experts_per_tok,
        "first_k_dense_replace": config.first_k_dense_replace,
        "vocab_size": config.vocab_size,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }

    write_excel(recorder.records, OUTPUT_FILE, config_summary)


if __name__ == "__main__":
    main()
