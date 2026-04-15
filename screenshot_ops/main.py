"""Entry point: load model, trace forward, write Excel + JSON.

Public API::

    from screenshot_ops import run_trace, build_config_summary, load_model

    output_path, records = run_trace(
        model_id="deepseek-ai/DeepSeek-V3-0324",
        num_layers=4,
        batch_size=1,
        seq_len=128,
        output_path="output.xlsx",   # optional
    )
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from screenshot_ops.dispatch import RecordingDispatch, TensorTracker
from screenshot_ops.excel_writer import ExcelWriter
from screenshot_ops.fx_graph_adapter import FXGraphAdapter
from screenshot_ops.fx_tracer import FXTracer
from screenshot_ops.graph import build_graph
from screenshot_ops.graph_builder import build_compute_graph
from screenshot_ops.fusion_pass import (
    FusionRule, FusionPass, FusionRuleDiscoverer, export_fusion_rules_json,
)
from screenshot_ops.model_loader import load_model
from screenshot_ops.tracker import ModuleTracker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

logger = logging.getLogger(__name__)

# Backward-compat map for --model v3 / v3.2 shorthand
_MODEL_DIRS = {
    "v3":  "deepseek_v3",
    "v3.2": "deepseek_v3_2",
}


# ── Public helpers ─────────────────────────────────────────────────────────────

def build_config_summary(
    model_id: str,
    config: Any,
    num_hidden_layers: int,
    batch_size: int,
    seq_len: int,
) -> Dict[str, Any]:
    """Return a config dict suitable for the Excel Model Config sheet.

    Always includes ``model_id``, ``model_type``, ``hidden_size``,
    ``num_attention_heads``, and ``vocab_size``.  Architecture-specific
    fields (MLA ranks, MoE counts, etc.) are included when present.
    """
    def _get(attr: str) -> Any:
        return getattr(config, attr, None)

    summary: Dict[str, Any] = {
        "model_id": model_id,
        "model_type": _get("model_type") or "unknown",
        "hidden_size": _get("hidden_size"),
        "intermediate_size": _get("intermediate_size"),
        "num_hidden_layers (full)": getattr(
            config, "_full_num_hidden_layers", _get("num_hidden_layers")),
        "num_hidden_layers (traced)": num_hidden_layers,
        "num_attention_heads": _get("num_attention_heads"),
        "num_key_value_heads": _get("num_key_value_heads"),
        "vocab_size": _get("vocab_size"),
        "batch_size": batch_size,
        "seq_len": seq_len,
    }

    # Optional architecture-specific fields
    for field in (
        "moe_intermediate_size",
        "q_lora_rank", "kv_lora_rank",
        "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
        "n_routed_experts", "n_shared_experts", "num_experts_per_tok",
        "first_k_dense_replace",
        "num_local_experts",
        "sliding_window", "head_dim", "rope_theta",
        "index_head_dim", "index_n_heads", "index_topk",
    ):
        val = _get(field)
        if val is not None:
            summary[field] = val

    return {k: v for k, v in summary.items() if v is not None}


def run_trace(
    model_id: str,
    num_layers: int = 4,
    batch_size: int = 1,
    seq_len: int = 128,
    output_path: Optional[Any] = None,
) -> Tuple[Path, List[Dict[str, Any]]]:
    """Load *model_id*, trace one forward pass, write Excel, return results.

    Parameters
    ----------
    model_id:
        HF Hub ID or local directory path.
    num_layers:
        Number of transformer blocks to instantiate (2–4 covers all op patterns).
    batch_size:
        Dummy input batch size.
    seq_len:
        Dummy input sequence length.
    output_path:
        Path for the output ``.xlsx`` file.  Defaults to
        ``<model_slug>_ops.xlsx`` in the current directory.

    Returns
    -------
    (output_path, records)
        output_path — ``Path`` to the written Excel file.
        records     — list of op-record dicts (``aten_op``, ``component``,
                      ``layer``, ``module_path``, shape/dtype fields, …).
    """
    logger.info("Loading model %s (%d layers) …", model_id, num_layers)
    model, config = load_model(model_id, num_hidden_layers=num_layers)

    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), device="meta")
    position_ids = torch.arange(seq_len, device="meta").unsqueeze(0)
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device="meta")
    mask = torch.triu(mask, diagonal=1)

    logger.info("Tracing forward pass (batch=%d, seq=%d) …", batch_size, seq_len)
    tensor_tracker = TensorTracker()
    tracker = ModuleTracker(model)
    recorder = RecordingDispatch(
        tensor_tracker=tensor_tracker,
        module_tracker=tracker,
        skip_reshapes=True,
    )
    try:
        with recorder, torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=mask,
                position_ids=position_ids,
                use_cache=False,
            )
    finally:
        tracker.remove()

    logger.info("Captured %d ops.", len(recorder.records))

    # Build lightweight data-flow graph for correct fusion I/O resolution
    graph = build_graph(recorder.records, tensor_tracker.passthroughs)
    logger.info("Built data-flow graph (%d producers, %d passthroughs).",
                len(graph.tensor_producer), len(graph.passthroughs))

    # Build NetworkX ComputeGraph for graph-based fusion (legacy path)
    compute_graph = build_compute_graph(recorder.records, tensor_tracker.passthroughs)
    logger.info("Built compute graph (%s).", compute_graph)

    config_summary = build_config_summary(
        model_id, config, num_layers, batch_size, seq_len)

    if output_path is None:
        slug = re.sub(r"[^\w]+", "_", Path(model_id).name)
        output_path = Path(f"{slug}_ops.xlsx")
    output_path = Path(output_path)

    writer = ExcelWriter(tracker, graph)
    writer.write(recorder.records, output_path, config_summary)

    # ── Legacy fusion (primary path) ──────────────────────────────────────
    from screenshot_ops.fusion import FusionEngine
    fusion_engine = FusionEngine(tracker, graph)
    fused = fusion_engine.fuse(recorder.records)
    specs = fusion_engine.extract_specs(fused)
    fusion_rules = FusionRule.from_specs(specs)

    fused_graph, fusion_result = FusionPass(fusion_rules, mode="module_class").apply(compute_graph)
    logger.info("Legacy graph fusion: %s", fusion_result.summary())

    json_path = export_fusion_rules_json(fusion_rules, output_path)
    logger.info("Fusion rules exported to %s", json_path)

    # ── FX-based tracing (experimental, does not override JSON) ───────────
    try:
        fx_tracer = FXTracer()
        gm = fx_tracer.trace(model, input_ids, mask, position_ids)

        adapter = FXGraphAdapter()
        fx_graph = adapter.convert(gm)
        logger.info("Built FX compute graph (%s).", fx_graph)

        discoverer = FusionRuleDiscoverer(fx_graph, adapter)
        fx_rules = discoverer.discover()
        logger.info("Discovered %d FX fusion rules (experimental).", len(fx_rules))

        fused_graph, fusion_result = FusionPass(fx_rules, mode="fx").apply(fx_graph)
        logger.info("FX graph fusion: %s", fusion_result.summary())
    except Exception as exc:
        logger.warning("FX tracing failed (%s).", exc)

    logger.info("Output saved to %s", output_path)

    return output_path, recorder.records


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace LLM operator sequences and write Excel report.")
    parser.add_argument(
        "model_id", nargs="?",
        help="HF Hub model ID or local directory (e.g. deepseek-ai/DeepSeek-V3-0324)")
    parser.add_argument(
        "--model", choices=_MODEL_DIRS.keys(),
        help="Shorthand for local DeepSeek model: v3 or v3.2 (backward compat)")
    parser.add_argument("--layers", type=int, default=4,
                        help="Number of transformer layers to trace (default: 4)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Dummy input batch size (default: 1)")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Dummy input sequence length (default: 128)")
    parser.add_argument("--output", "-o",
                        help="Output .xlsx path (default: <model_slug>_ops.xlsx)")
    args = parser.parse_args()

    # Resolve model_id: positional takes precedence over --model shorthand
    if args.model_id:
        model_id = args.model_id
    elif args.model:
        model_dir_name = _MODEL_DIRS[args.model]
        model_id = str(
            Path(__file__).parent.parent / "hf_models" / model_dir_name)
    else:
        parser.error("Provide a model_id argument or --model v3/v3.2")

    output_path = Path(args.output) if args.output else None

    run_trace(
        model_id=model_id,
        num_layers=args.layers,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
