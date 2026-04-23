"""Top-level CLI entry point for ZRT-Sim.

Usage::

    python -m python.zrt <model_id> [options]
    python -m python.zrt.graph.main <model_id> [options]  # backward compat

Examples::

    python -m python.zrt Qwen/Qwen2.5-7B-Instruct --layers 4
    python -m python.zrt deepseek-ai/DeepSeek-V3-0324 --layers 4 --hw nvidia_h100_sxm --tp 8
    python -m python.zrt hf_models/llama3_8b --train --layers 2
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from python.zrt.graph.main import (
    run_trace_phases,
    _make_model_slug,
    _MODEL_DIRS,
    _PHASE_ALIASES,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace LLM operator sequences and write Excel + computation graph.")
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
                        help="Prefill sequence length (default: 128)")
    parser.add_argument("--output-dir", "-o",
                        help="Output directory (default: output/graph/<model_slug>)")
    parser.add_argument(
        "--phases", nargs="+", default=["prefill", "decode"],
        choices=["prefill", "decode", "forward",
                 "train_forward", "train_backward", "train"],
        metavar="PHASE",
        help="Phases to trace (default: prefill decode). "
             "Inference: prefill, decode. Training: train_forward, train_backward. "
             "'forward'/'train' are aliases for 'prefill'/'train_forward'.")
    parser.add_argument(
        "--phase", default=None,
        help="(legacy) Trace a single phase. Overrides --phases when set.")
    parser.add_argument(
        "--train", action="store_true", default=False,
        help="Trace training phases (train_forward + train_backward). "
             "Equivalent to --phases train_forward train_backward.")
    parser.add_argument(
        "--platform",
        default="generic",
        choices=["cuda", "ascend_npu", "cpu", "generic"],
        help="Target inference platform for fusion labelling (default: generic).",
    )
    parser.add_argument(
        "--graph-mode",
        action="store_true",
        default=False,
        help="Use torch.compile graph capture instead of TorchDispatchMode eager tracing.",
    )
    parser.add_argument(
        "--hw",
        metavar="HW",
        default=None,
        help="Hardware spec name for performance report (e.g. nvidia_h100_sxm). "
             f"Available: {', '.join(__import__('python.zrt.hardware.registry', fromlist=['list_available']).list_available())}",
    )
    parser.add_argument(
        "--tp", type=int, default=1,
        help="Tensor-parallel degree used when --hw is set (default: 1).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=False,
        help="Enable activation checkpointing during training phases.",
    )

    # --- Training modelling flags (used with --train --hw) ---
    parser.add_argument(
        "--pp", type=int, default=1,
        help="Pipeline-parallel degree (training, default: 1).",
    )
    parser.add_argument(
        "--ep", type=int, default=1,
        help="Expert-parallel degree (training, default: 1).",
    )
    parser.add_argument(
        "--cp", type=int, default=1,
        help="Context-parallel degree (training, default: 1).",
    )
    parser.add_argument(
        "--dp", type=int, default=1,
        help="Data-parallel degree (training, default: 1).",
    )
    parser.add_argument(
        "--zero-stage", type=int, default=1,
        help="ZeRO optimization stage 0-3 (training, default: 1).",
    )
    parser.add_argument(
        "--optimizer", default="adam",
        choices=["adam", "adamw", "muon"],
        help="Optimizer for training estimation (default: adam).",
    )
    parser.add_argument(
        "--micro-batch", type=int, default=1,
        help="Micro-batch size per GPU (training, default: 1).",
    )
    parser.add_argument(
        "--global-batch", type=int, default=32,
        help="Global batch size across DP ranks (training, default: 32).",
    )
    parser.add_argument(
        "--total-params", type=float, default=None,
        help="Full model param count, e.g. 671e9 (for scaling traced layers).",
    )
    parser.add_argument(
        "--hidden", type=int, default=7168,
        help="Hidden dimension for memory estimation (default: 7168).",
    )
    parser.add_argument(
        "--num-layers-full", type=int, default=None,
        help="Total layers in full model (defaults to --layers if not set).",
    )

    _layer_group = parser.add_mutually_exclusive_group()
    _layer_group.add_argument(
        "--target-layers",
        metavar="IDX",
        help="Comma-separated layer indices to trace, e.g. '0,3'.",
    )
    _layer_group.add_argument(
        "--auto-layers",
        action="store_true",
        default=False,
        help="Automatically select the first dense and first sparse (MoE) layer.",
    )
    args = parser.parse_args()

    # Resolve model_id
    if args.model_id:
        model_id = args.model_id
    elif args.model:
        model_dir_name = _MODEL_DIRS[args.model]
        model_id = str(
            Path(__file__).parent.parent.parent / "hf_models" / model_dir_name)
    else:
        parser.error("Provide a model_id argument or --model v3/v3.2")

    output_dir = Path(args.output_dir) if args.output_dir else None

    # Phase resolution: --train > --phase (legacy) > --phases
    if args.train:
        phases = ["train_forward", "train_backward"]
    elif args.phase is not None:
        phases = [args.phase]
    else:
        phases = args.phases

    target_layers: Optional[List[int]] = None
    if args.target_layers:
        try:
            target_layers = [int(x.strip()) for x in args.target_layers.split(",")]
        except ValueError:
            parser.error(
                f"--target-layers must be comma-separated integers, "
                f"got: {args.target_layers!r}"
            )

    effective_auto_layers = args.auto_layers or (target_layers is None)

    result = run_trace_phases(
        model_id=model_id,
        num_layers=args.layers,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        output_dir=output_dir,
        phases=tuple(phases),
        target_layers=target_layers,
        auto_layers=effective_auto_layers,
        platform=args.platform,
        graph_mode=args.graph_mode,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    if args.hw:
        import python.zrt.hardware.registry as hw_registry
        hw = hw_registry.load(args.hw)

        if args.train:
            _run_training_modelling(args, model_id, hw, result)
        else:
            _run_inference_pipeline(args, model_id, hw, result)


def _run_inference_pipeline(args, model_id: str, hw, result) -> None:
    """Run the inference transform + simulate + report pipeline."""
    from python.zrt.transform import (
        build_default_pipeline, TransformContext,
        ParallelConfig, StreamConfig,
    )
    from python.zrt.executor import DAGScheduler
    from python.zrt.simulator import SimulatorHub
    from python.zrt.report import build_summary
    from python.zrt.graph.excel_writer import append_perf_summary

    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=args.tp),
        stream_config=StreamConfig(num_compute_streams=1, num_comm_streams=1),
    )
    pipe = build_default_pipeline()
    hub = SimulatorHub.default()
    scheduler = DAGScheduler(hw_spec=hw)

    for phase, (raw_graph, _) in result.graphs.items():
        g = pipe.run(raw_graph, ctx)
        tl = scheduler.schedule(g)
        sim_results = hub.simulate_graph(g, hw)
        summary = build_summary(
            model=model_id,
            hardware=args.hw,
            phase=phase,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            graph=g,
            sim_results=sim_results,
            timeline=tl,
            hw_spec=hw,
            parallel_desc=f"TP{args.tp}",
        )
        try:
            print(f"\n{summary}")
        except UnicodeEncodeError:
            logger.info("Performance summary: %s", summary)

        slug = _make_model_slug(model_id)
        xlsx_path = result.output_dir / f"{slug}_{phase}_ops.xlsx"
        if xlsx_path.exists():
            append_perf_summary(xlsx_path, summary)
            logger.info("Performance summary written to %s", xlsx_path)


def _run_training_modelling(args, model_id: str, hw, result) -> None:
    """Run the full training pipeline: transform → schedule → simulate → report → export."""
    from python.zrt.transform import (
        build_default_pipeline, TransformContext,
        ParallelConfig, StreamConfig,
        TrainingMemoryPass, export_training_graphs,
    )
    from python.zrt.transform.context import TrainingConfig
    from python.zrt.executor import DAGScheduler
    from python.zrt.simulator import SimulatorHub
    from python.zrt.report import build_training_summary

    fwd_pair = result.graphs.get("train_forward")
    if not fwd_pair:
        logger.error("--train --hw requires train_forward phase but none was captured.")
        return

    raw_fwd = fwd_pair[0]
    bwd_pair = result.graphs.get("train_backward")
    raw_bwd = bwd_pair[0] if bwd_pair else None

    cp = getattr(args, "cp", 1)
    parallel_parts = []
    if args.tp > 1: parallel_parts.append(f"TP{args.tp}")
    if cp > 1:      parallel_parts.append(f"CP{cp}")
    if args.ep > 1: parallel_parts.append(f"EP{args.ep}")
    if args.pp > 1: parallel_parts.append(f"PP{args.pp}")
    if args.dp > 1: parallel_parts.append(f"DP{args.dp}")
    parallel_desc = "-".join(parallel_parts) if parallel_parts else "single"

    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=args.tp, cp=cp, ep=args.ep, pp=args.pp, dp=args.dp),
        stream_config=StreamConfig(num_compute_streams=1, num_comm_streams=1),
        training=TrainingConfig(
            optimizer=args.optimizer,
            zero_stage=args.zero_stage,
            micro_batch=args.micro_batch,
            global_batch=args.global_batch,
        ),
    )
    pipe = build_default_pipeline()
    scheduler = DAGScheduler(hw_spec=hw)
    hub = SimulatorHub.default()

    fwd_transformed = pipe.run(raw_fwd, ctx)
    fwd_tl = scheduler.schedule(fwd_transformed)
    fwd_sim = hub.simulate_graph(fwd_transformed, hw)
    print(f"  fwd: {raw_fwd.num_nodes()} -> {fwd_transformed.num_nodes()} nodes"
          f"  total={fwd_tl.total_latency_ms:.3f} ms")

    if raw_bwd is not None:
        bwd_transformed = pipe.run(raw_bwd, ctx)
        bwd_tl = scheduler.schedule(bwd_transformed)
        bwd_sim = hub.simulate_graph(bwd_transformed, hw)
        print(f"  bwd: {raw_bwd.num_nodes()} -> {bwd_transformed.num_nodes()} nodes"
              f"  total={bwd_tl.total_latency_ms:.3f} ms")
    else:
        logger.warning("No train_backward graph captured; backward metrics will be zero.")
        bwd_transformed = fwd_transformed
        bwd_tl = fwd_tl
        bwd_sim = {}

    # Memory estimation
    mem_ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=args.tp, cp=cp, ep=args.ep, pp=args.pp, dp=args.dp),
        stream_config=StreamConfig(num_compute_streams=1, num_comm_streams=1),
        training=TrainingConfig(
            optimizer=args.optimizer,
            zero_stage=args.zero_stage,
            micro_batch=args.micro_batch,
            global_batch=args.global_batch,
        ),
    )
    fwd_with_mem = TrainingMemoryPass().run(fwd_transformed, mem_ctx)
    mem_bd = fwd_with_mem.metadata.get("memory_breakdown")

    summary = build_training_summary(
        model            = model_id,
        hardware         = args.hw,
        batch_size       = args.batch_size,
        seq_len          = args.seq_len,
        fwd_graph        = fwd_transformed,
        bwd_graph        = bwd_transformed,
        fwd_results      = fwd_sim,
        bwd_results      = bwd_sim,
        fwd_timeline     = fwd_tl,
        bwd_timeline     = bwd_tl,
        hw_spec          = hw,
        parallel_desc    = parallel_desc,
        memory_breakdown = mem_bd,
    )

    try:
        print(f"\n{summary}")
    except UnicodeEncodeError:
        summary_str = str(summary).replace("µ", "u")
        logger.info("Training summary:\n%s", summary_str)

    # Export Excel + JSON
    output_dir = result.output_dir / "transformed" if result.output_dir else None
    if output_dir:
        try:
            export_result = export_training_graphs(
                fwd_graph        = fwd_transformed,
                bwd_graph        = bwd_transformed,
                ctx              = ctx,
                output_dir       = output_dir,
                training_summary = summary,
            )
            print(f"\n  Excel:    {export_result['excel']}")
            print(f"  JSON fwd: {export_result['json_fwd']}")
            print(f"  JSON bwd: {export_result['json_bwd']}")
        except Exception as exc:
            logger.warning("Export failed: %s", exc)


if __name__ == "__main__":
    main()
