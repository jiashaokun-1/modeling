"""Standalone fusion discovery pipeline: capture → discover → write.

Public API::

    from screenshot_ops.fused_discovery import run_fused_discovery

    result = run_fused_discovery(
        model_id="hf_models/deepseek_v3",
        num_layers=4,
        batch_size=1,
        seq_len=128,
        output_path="output.xlsx",
    )
    print(result.json_path)
    print(result.fusion_result.summary())

Or use the individual steps::

    from screenshot_ops.fused_discovery import capture_ops, discover_fusion_rules, write_fusion_rules

    capture = capture_ops(model, config, batch_size=1, seq_len=128)
    rules = discover_fusion_rules(capture)
    json_path = write_fusion_rules(rules, output_path)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from screenshot_ops.compute_graph import ComputeGraph
from screenshot_ops.dispatch import RecordingDispatch, TensorTracker
from screenshot_ops.fusion import FusionEngine
from screenshot_ops.fusion_pass import (
    FusionResult,
    FusionRule,
    FusionPass,
    export_fusion_rules_json,
)
from screenshot_ops.graph import DataFlowGraph, build_graph
from screenshot_ops.graph_builder import build_compute_graph
from screenshot_ops.model_loader import load_model
from screenshot_ops.tracker import ModuleTracker

logger = logging.getLogger(__name__)


@dataclass
class CaptureResult:
    """Result of operator capture (forward-pass tracing)."""
    records: List[Dict[str, Any]]
    tracker: ModuleTracker
    graph: DataFlowGraph
    compute_graph: ComputeGraph
    model_id: str
    config: Any
    num_layers: int
    batch_size: int
    seq_len: int


@dataclass
class FusedDiscoveryResult:
    """Complete result of the fusion discovery pipeline."""
    capture: CaptureResult
    fusion_rules: List[FusionRule]
    fused_graph: ComputeGraph
    fusion_result: FusionResult
    json_path: Path


def capture_ops(
    model: torch.nn.Module,
    config: Any,
    batch_size: int = 1,
    seq_len: int = 128,
    model_id: str = "",
    num_layers: int = 4,
) -> CaptureResult:
    """Capture operator sequence by running a forward pass with FakeTensors.

    The model is on CPU; FakeTensorMode converts all tensors (inputs and
    parameters) to FakeTensors so no real computation occurs.

    Parameters
    ----------
    model:
        The nn.Module to trace (on CPU).
    config:
        Model config (used for vocab_size).
    batch_size:
        Dummy input batch size.
    seq_len:
        Dummy input sequence length.
    model_id:
        Model identifier (for logging / metadata).
    num_layers:
        Number of transformer layers (for metadata).

    Returns
    -------
    CaptureResult with records, tracker, graph, and compute_graph.
    """
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0)
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1)

    input_ids = fake_mode.from_tensor(input_ids)
    position_ids = fake_mode.from_tensor(position_ids)
    mask = fake_mode.from_tensor(mask)

    logger.info("Tracing forward pass (batch=%d, seq=%d) with FakeTensorMode …",
                batch_size, seq_len)

    tensor_tracker = TensorTracker()
    tracker = ModuleTracker(model)
    recorder = RecordingDispatch(
        tensor_tracker=tensor_tracker,
        module_tracker=tracker,
        skip_reshapes=True,
    )
    try:
        with fake_mode, recorder, torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=mask,
                position_ids=position_ids,
                use_cache=False,
            )
    finally:
        tracker.remove()

    logger.info("Captured %d ops.", len(recorder.records))

    graph = build_graph(recorder.records, tensor_tracker.passthroughs)
    logger.info("Built data-flow graph (%d producers, %d passthroughs).",
                len(graph.tensor_producer), len(graph.passthroughs))

    compute_graph = build_compute_graph(
        recorder.records, tensor_tracker.passthroughs)
    logger.info("Built compute graph (%s).", compute_graph)

    return CaptureResult(
        records=recorder.records,
        tracker=tracker,
        graph=graph,
        compute_graph=compute_graph,
        model_id=model_id,
        config=config,
        num_layers=num_layers,
        batch_size=batch_size,
        seq_len=seq_len,
    )


def discover_fusion_rules(capture: CaptureResult) -> Tuple[List[FusionRule], ComputeGraph, FusionResult]:
    """Discover fusion rules from captured operator sequence.

    Parameters
    ----------
    capture:
        Result from ``capture_ops()``.

    Returns
    -------
    (fusion_rules, fused_graph, fusion_result)
    """
    fusion_engine = FusionEngine(capture.tracker, capture.graph)
    fused = fusion_engine.fuse(capture.records)
    specs = fusion_engine.extract_specs(fused)
    fusion_rules = FusionRule.from_specs(specs)

    fused_graph, fusion_result = FusionPass(
        fusion_rules, mode="module_key").apply(capture.compute_graph)
    logger.info("Fusion: %s", fusion_result.summary())

    return fusion_rules, fused_graph, fusion_result


def write_fusion_rules(
    fusion_rules: List[FusionRule],
    output_path: Path,
) -> Path:
    """Write fusion rules to a JSON file.

    Parameters
    ----------
    fusion_rules:
        List of FusionRule to export.
    output_path:
        Path to the Excel output file.  The JSON file will be named
        ``<stem>_fusion_rules.json`` alongside the Excel file.

    Returns
    -------
    Path to the written JSON file.
    """
    json_path = export_fusion_rules_json(fusion_rules, Path(output_path))
    logger.info("Fusion rules exported to %s", json_path)
    return json_path


def run_fused_discovery(
    model_id: str,
    num_layers: int = 4,
    batch_size: int = 1,
    seq_len: int = 128,
    output_path: Optional[Any] = None,
) -> FusedDiscoveryResult:
    """One-shot pipeline: load model → capture → discover → write.

    Parameters
    ----------
    model_id:
        HF Hub ID or local directory path.
    num_layers:
        Number of transformer blocks to instantiate.
    batch_size:
        Dummy input batch size.
    seq_len:
        Dummy input sequence length.
    output_path:
        Path for the output ``.xlsx`` file.  Defaults to
        ``<model_slug>_ops.xlsx``.

    Returns
    -------
    FusedDiscoveryResult with all intermediate and final results.
    """
    logger.info("Loading model %s (%d layers) …", model_id, num_layers)
    model, config = load_model(model_id, num_hidden_layers=num_layers)

    capture = capture_ops(
        model, config,
        batch_size=batch_size, seq_len=seq_len,
        model_id=model_id, num_layers=num_layers,
    )

    if output_path is None:
        slug = re.sub(r"[^\w]+", "_", Path(model_id).name)
        output_path = Path(f"{slug}_ops.xlsx")
    output_path = Path(output_path)

    fusion_rules, fused_graph, fusion_result = discover_fusion_rules(capture)

    json_path = write_fusion_rules(fusion_rules, output_path)

    return FusedDiscoveryResult(
        capture=capture,
        fusion_rules=fusion_rules,
        fused_graph=fused_graph,
        fusion_result=fusion_result,
        json_path=json_path,
    )


_MODEL_DIRS = {
    "v3": "deepseek_v3",
    "v3.2": "deepseek_v3_2",
}


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Standalone fusion discovery pipeline")
    parser.add_argument(
        "--model", choices=_MODEL_DIRS.keys(), default="v3",
        help="Model shorthand: v3 or v3.2 (default: v3)")
    parser.add_argument("--layers", type=int, default=4,
                        help="Number of transformer layers (default: 4)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Dummy input batch size (default: 1)")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Dummy input sequence length (default: 128)")
    parser.add_argument("--output", "-o",
                        help="Output .xlsx path")
    args = parser.parse_args()

    model_dir_name = _MODEL_DIRS[args.model]
    model_id = str(Path(__file__).parent.parent / "hf_models" / model_dir_name)
    output_path = Path(args.output) if args.output else None

    result = run_fused_discovery(
        model_id=model_id,
        num_layers=args.layers,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        output_path=output_path,
    )

    print(f"\nJSON: {result.json_path}")
    print(f"Fusion rules: {len(result.fusion_rules)}")
    print(result.fusion_result.summary())
