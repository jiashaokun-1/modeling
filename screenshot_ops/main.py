"""Entry point: load model, trace forward, write Excel + JSON."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from screenshot_ops.dispatch import RecordingDispatch, TensorTracker
from screenshot_ops.excel_writer import ExcelWriter
from screenshot_ops.model_loader import load_model
from screenshot_ops.tracker import ModuleTracker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MODEL_DIRS = {
    "v3": "deepseek_v3",
    "v3.2": "deepseek_v3_2",
}

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Trace LLM operator sequences")
    parser.add_argument("--model", choices=MODEL_DIRS.keys(), default="v3",
                        help="Model version to trace (default: v3)")
    parser.add_argument("--layers", type=int, default=4,
                        help="Number of layers to trace (default: 4)")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Sequence length (default: 128)")
    args = parser.parse_args()

    model_name = args.model
    num_layers = args.layers
    batch_size = 1
    seq_len = args.seq_len

    model_dir_name = MODEL_DIRS[model_name]
    model_dir = Path(__file__).parent.parent / "hf_models" / model_dir_name
    output_file = Path(__file__).parent.parent / f"deepseek_{model_name.replace('.', '_')}_ops.xlsx"

    logger.info("Loading DeepSeek-%s model from %s (%d layers)...", model_name.upper(), model_dir, num_layers)
    model, config = load_model(model_dir, num_hidden_layers=num_layers)

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="meta")
    position_ids = torch.arange(seq_len, device="meta").unsqueeze(0)

    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device="meta")
    mask = torch.triu(mask, diagonal=1)

    logger.info("Tracing forward pass (batch=%d, seq=%d)...", batch_size, seq_len)
    tensor_tracker = TensorTracker()
    tracker = ModuleTracker(model)
    recorder = RecordingDispatch(
        tensor_tracker=tensor_tracker,
        module_tracker=tracker,
        skip_reshapes=True,
    )
    try:
        with recorder, torch.no_grad():
            model(input_ids=input_ids, attention_mask=mask, position_ids=position_ids,
                  use_cache=False)
    finally:
        tracker.remove()

    logger.info("Captured %d ops (after filtering zero-cost reshapes)", len(recorder.records))

    model_type = getattr(config, "model_type", f"deepseek_{model_name.replace('.', '_')}")
    config_summary = {
        "model_type": model_type,
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
    
    if hasattr(config, "index_head_dim"):
        config_summary["index_head_dim"] = config.index_head_dim
        config_summary["index_n_heads"] = config.index_n_heads
        config_summary["index_topk"] = config.index_topk

    writer = ExcelWriter(tracker)
    writer.write(recorder.records, output_file, config_summary)
    logger.info("Output saved to %s", output_file)


if __name__ == "__main__":
    main()
