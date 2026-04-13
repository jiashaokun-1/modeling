"""Entry point: load model, trace forward, write Excel + JSON."""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from screenshot_ops.dispatch import RecordingDispatch, TensorTracker
from screenshot_ops.excel_writer import ExcelWriter
from screenshot_ops.model_loader import load_model
from screenshot_ops.tracker import ModuleTracker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MODEL_DIR = Path(__file__).parent.parent / "hf_models" / "deepseek_v3"
OUTPUT_FILE = Path(__file__).parent.parent / "deepseek_v3_ops.xlsx"

logger = logging.getLogger(__name__)


def main():
    num_layers = 4
    batch_size = 1
    seq_len = 128

    logger.info("Loading DeepSeek-V3 model from %s (%d layers)...", MODEL_DIR, num_layers)
    model, config = load_model(MODEL_DIR, num_hidden_layers=num_layers)

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

    writer = ExcelWriter(tracker)
    writer.write(recorder.records, OUTPUT_FILE, config_summary)
