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

from screenshot_ops.excel_writer import ExcelWriter
from screenshot_ops.fused_discovery import (
    CaptureResult,
    capture_ops,
    discover_fusion_rules,
    write_fusion_rules,
)
from screenshot_ops.model_loader import load_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

logger = logging.getLogger(__name__)

_MODEL_DIRS = {
    "v3":  "deepseek_v3",
    "v3.2": "deepseek_v3_2",
}


def build_config_summary(
    model_id: str,
    config: Any,
    num_hidden_layers: int,
    batch_size: int,
    seq_len: int,
) -> Dict[str, Any]:
    """Return a config dict suitable for the Excel Model Config sheet."""
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
    """Load *model_id*, trace one forward pass, write Excel, return results."""
    logger.info("Loading model %s (%d layers) …", model_id, num_layers)
    model, config = load_model(model_id, num_hidden_layers=num_layers)

    capture = capture_ops(
        model, config,
        batch_size=batch_size, seq_len=seq_len,
        model_id=model_id, num_layers=num_layers,
    )

    config_summary = build_config_summary(
        model_id, config, num_layers, batch_size, seq_len)

    if output_path is None:
        slug = re.sub(r"[^\w]+", "_", Path(model_id).name)
        output_path = Path(f"{slug}_ops.xlsx")
    output_path = Path(output_path)

    writer = ExcelWriter(capture.tracker, capture.graph)
    writer.write(capture.records, output_path, config_summary)

    fusion_rules, fused_graph, fusion_result = discover_fusion_rules(capture)
    json_path = write_fusion_rules(fusion_rules, output_path)

    logger.info("Output saved to %s", output_path)

    return output_path, capture.records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace LLM operator sequences and write Excel report.")
    parser.add_argument(
        "model_id", nargs="?",
        help="HF Hub model ID or local directory")
    parser.add_argument(
        "--model", choices=_MODEL_DIRS.keys(),
        help="Shorthand for local DeepSeek model: v3 or v3.2")
    parser.add_argument("--layers", type=int, default=4,
                        help="Number of transformer layers to trace (default: 4)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Dummy input batch size (default: 1)")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Dummy input sequence length (default: 128)")
    parser.add_argument("--output", "-o",
                        help="Output .xlsx path")
    args = parser.parse_args()

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
