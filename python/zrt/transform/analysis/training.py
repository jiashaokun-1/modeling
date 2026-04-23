"""Training analysis passes: FLOPs (6P rule), Memory (ZeRO), Pipeline (1F1B)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from python.zrt.ir.param_count import count_params
from python.zrt.transform.base import GraphPass

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext


# ── TrainingFlopsPass ───────────────────────────────────────────────────────────

class TrainingFlopsPass(GraphPass):
    """Annotate graph with training FLOPs using the 6P rule.

    Training FLOPs = 6 * total_params * tokens (for dense transformers)
    Forward: 2 * params * tokens, Backward: 4 * params * tokens

    Adds to graph.metadata:
      "training_flops": float
      "forward_flops": float
      "backward_flops": float
      "total_params": int
    """

    name = "training_flops"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        g = graph.clone()

        # Check if total_params was provided as metadata override (Tier-1)
        has_param_override = g.metadata.get("total_params", 0) > 0

        total_params = count_params(g)

        # Layer scaling: when only a subset of layers is traced, scale
        # graph-counted params to full model. Tier-1 override is already
        # the full-model count, so it does NOT need scaling.
        num_layers = g.metadata.get("num_layers", 0)
        num_layers_traced = g.metadata.get("num_layers_traced", num_layers)
        layer_scale = num_layers / num_layers_traced if num_layers_traced > 0 and num_layers != num_layers_traced else 1.0

        if not has_param_override and layer_scale != 1.0:
            total_params = int(total_params * layer_scale)

        # Get sequence length and batch size from metadata
        seq_len = g.metadata.get("seq_len", 2048)
        batch_size = ctx.training.micro_batch if ctx.training else 1
        tokens = seq_len * batch_size

        # 6P rule: 2 params * tokens (forward) + 4 params * tokens (backward)
        forward_flops = 2 * total_params * tokens
        backward_flops = 4 * total_params * tokens
        training_flops = forward_flops + backward_flops

        g.metadata["training_flops"] = training_flops
        g.metadata["forward_flops"] = forward_flops
        g.metadata["backward_flops"] = backward_flops
        g.metadata["total_params"] = total_params
        g.metadata["layer_scale"] = layer_scale

        return g


# ── TrainingMemoryPass ──────────────────────────────────────────────────────────

@dataclass
class TrainingMemoryBreakdown:
    """Training memory breakdown per GPU."""
    weights: float = 0.0      # Model weights (bytes)
    grads: float = 0.0        # Gradients (bytes)
    opt_state: float = 0.0    # Optimizer state (bytes)
    activations: float = 0.0  # Activations (bytes)
    comm_buffers: float = 0.0 # Communication buffers (bytes)

    @property
    def total(self) -> float:
        return self.weights + self.grads + self.opt_state + self.activations + self.comm_buffers

    def to_dict(self) -> dict[str, float]:
        return {
            "weights": self.weights,
            "grads": self.grads,
            "opt_state": self.opt_state,
            "activations": self.activations,
            "comm_buffers": self.comm_buffers,
            "total": self.total,
        }


class TrainingMemoryPass(GraphPass):
    """Annotate graph with training memory breakdown using ZeRO sharding.

    ZeRO stages:
      - 0: No sharding (all replicas store full weights, grads, opt_state)
      - 1: Optimizer state sharded across DP
      - 2: Gradients + optimizer state sharded across DP
      - 3: Weights + gradients + optimizer state sharded across DP

    Adds to graph.metadata:
      "memory_breakdown": TrainingMemoryBreakdown
    """

    name = "training_memory"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        g = graph.clone()

        # Training always uses BF16 parameters; mixed-precision keeps FP32 master copy
        param_dtype = 2  # BF16

        total_params = count_params(g)

        # Get parallel config
        dp = ctx.parallel.dp if ctx.parallel else 1
        tp = ctx.parallel.tp if ctx.parallel else 1
        zero_stage = ctx.training.zero_stage if ctx.training else 0

        # Weights: sharded by TP, optionally by ZeRO-3
        weights_sharding = tp
        if zero_stage >= 3:
            weights_sharding *= dp
        weights_bytes = (total_params * param_dtype) / weights_sharding

        # Gradients: same as weights, optionally sharded by ZeRO-2/3
        grads_sharding = weights_sharding
        if zero_stage >= 2 and zero_stage < 3:
            grads_sharding = dp  # ZeRO-2 shards grads but not weights
        grads_bytes = (total_params * param_dtype) / grads_sharding

        # Optimizer state: Adam/AdamW under mixed precision uses FP32 master + m + v = 12 B/P
        optimizer = ctx.training.optimizer if ctx.training else "adam"
        opt_bytes_per_param = 12 if optimizer in ("adam", "adamw") else 8  # Muon: master + momentum
        opt_sharding = 1
        if zero_stage >= 1:
            opt_sharding = dp
        opt_bytes = (total_params * opt_bytes_per_param) / opt_sharding

        # Activations: estimate using Korthikanti formula
        seq_len = g.metadata.get("seq_len", 2048)
        hidden = g.metadata.get("hidden", 4096)
        num_layers = g.metadata.get("num_layers", 32)
        batch_size = ctx.training.micro_batch if ctx.training else 1

        # Korthikanti formula: 10 * hidden * seq_len * num_layers * batch_size / tp
        # Factor of 10 accounts for attention + FFN activations
        activations_bytes = (10 * hidden * seq_len * num_layers * batch_size) / tp

        # Communication buffers: AG/RS buffers
        # Approx: 2 * hidden * seq_len / tp per layer
        comm_bytes = (2 * hidden * seq_len * num_layers) / tp

        breakdown = TrainingMemoryBreakdown(
            weights=weights_bytes,
            grads=grads_bytes,
            opt_state=opt_bytes,
            activations=activations_bytes,
            comm_buffers=comm_bytes,
        )

        g.metadata["memory_breakdown"] = breakdown

        return g


# ── TrainingPipelinePass ────────────────────────────────────────────────────────

@dataclass
class PipelineStepMetrics:
    """Pipeline step metrics for 1F1B schedule."""
    step_time_ms: float = 0.0
    per_stage_ms: float = 0.0
    warmup_steps: int = 0
    cooldown_steps: int = 0
    steady_steps: int = 0
    bubble_fraction: float = 0.0
    mfu: float = 0.0  # Model FLOPs Utilization

    def to_dict(self) -> dict[str, float]:
        return {
            "step_time_ms": self.step_time_ms,
            "per_stage_ms": self.per_stage_ms,
            "warmup_steps": self.warmup_steps,
            "cooldown_steps": self.cooldown_steps,
            "steady_steps": self.steady_steps,
            "bubble_fraction": self.bubble_fraction,
            "mfu": self.mfu,
        }


class TrainingPipelinePass(GraphPass):
    """Annotate graph with pipeline schedule metrics (1F1B).

    1F1B schedule:
      - Warmup: PP-1 steps (ramp up microbatches)
      - Steady: M - PP + 1 steps (full pipeline utilization)
      - Cooldown: PP-1 steps (drain microbatches)

    Bubble fraction = (warmup + cooldown) / total_steps

    Adds to graph.metadata:
      "pipeline_metrics": PipelineStepMetrics
    """

    name = "training_pipeline"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        g = graph.clone()

        pp = ctx.parallel.pp if ctx.parallel else 1
        num_microbatches = ctx.training.num_microbatches if ctx.training else 1
        hw = ctx.hw_spec
        layer_scale = g.metadata.get("layer_scale", 1.0)

        from python.zrt.executor.scheduler import DAGScheduler
        sched = DAGScheduler(hw)

        if pp > 1 and any("stage_id" in n.annotations for n in g.nodes.values()):
            # Per-stage scheduling: schedule each stage's subgraph independently
            stage_node_sets: dict[int, set[str]] = {}
            for node in g.nodes.values():
                sid = node.annotations.get("stage_id", 0)
                stage_node_sets.setdefault(sid, set()).add(node.id)

            stage_latencies: dict[int, float] = {}
            for s_id in range(pp):
                node_ids = stage_node_sets.get(s_id, set())
                if not node_ids:
                    stage_latencies[s_id] = 0.0
                    continue
                sub = g.subgraph(node_ids)
                tl = sched.schedule(sub)
                lat = tl.total_latency_us
                if layer_scale != 1.0:
                    lat *= layer_scale
                stage_latencies[s_id] = lat

            g.metadata["stage_timelines_fwd"] = dict(stage_latencies)

            # Bottleneck stage (max per-stage fwd latency)
            per_stage_us = max(stage_latencies.values(), default=0.0)
        else:
            # pp=1 (or stage_id not yet annotated): whole graph is one stage
            tl = sched.schedule(g)
            per_stage_us = tl.total_latency_us
            if layer_scale != 1.0:
                per_stage_us *= layer_scale

        warmup_steps = max(0, pp - 1)
        cooldown_steps = max(0, pp - 1)
        steady_steps = max(0, num_microbatches - pp + 1)
        total_steps = warmup_steps + num_microbatches + cooldown_steps

        # 1F1B step time: warmup + M microbatches through bottleneck + cooldown
        step_time_us = per_stage_us * total_steps
        step_time_ms = step_time_us / 1000.0
        per_stage_ms = per_stage_us / 1000.0

        bubble_fraction = (
            (warmup_steps + cooldown_steps) / total_steps if total_steps > 0 else 0.0
        )

        training_flops = g.metadata.get("training_flops", 0.0)
        world_size = ctx.parallel.total_devices if ctx.parallel else 1

        from python.zrt.ir.types import DType
        peak_flops_per_gpu = hw.peak_flops(DType.BF16)

        step_time_sec = step_time_us / 1e6
        achieved_flops = training_flops / step_time_sec if step_time_sec > 0 else 0.0
        peak_flops_total = world_size * peak_flops_per_gpu
        mfu = achieved_flops / peak_flops_total if peak_flops_total > 0 else 0.0

        metrics = PipelineStepMetrics(
            step_time_ms=step_time_ms,
            per_stage_ms=per_stage_ms,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
            steady_steps=steady_steps,
            bubble_fraction=bubble_fraction,
            mfu=mfu,
        )

        g.metadata["pipeline_metrics"] = metrics
        return g
