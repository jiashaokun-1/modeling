"""Shared training report types.

This module provides unified report types used by both spec-based (Stack A)
and graph-native (Stack B) training estimation paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from zrt.training.models.memory import MemBreakdown


@dataclass
class TrainingReport:
    """Unified training performance estimation report.

    Used by both spec-based estimation (Stack A) and graph-native estimation
    (Stack B) to return consistent, comparable results.

    Core timing metrics
    ------------------
    step_time_ms: Total step time in milliseconds
    per_stage: Per-stage timing breakdown (list of StageTime from compose/stage.py) [Stack A]
    per_stage_ms: Average per-stage time in milliseconds [Stack B]

    Efficiency metrics
    -----------------
    mfu: Model FLOPs Utilization (excludes recompute overhead)
    hfu: Hardware FLOPs Utilization (includes recompute overhead)

    FLOPs breakdown
    --------------
    total_flops: Total training FLOPs (forward + backward + recompute) [Stack A]
    forward_flops: Forward pass FLOPs [Stack B]
    backward_flops: Backward pass FLOPs [Stack B]
    training_flops: Alias for total_flops [Stack B]

    Memory metrics
    -------------
    memory: MemBreakdown with weights, grads, opt_state, activations, comm_buffers [Stack A]
    memory_breakdown: Dict representation of memory breakdown [Stack B]

    Pipeline metrics
    ---------------
    bubble_fraction: Fraction of step time spent in pipeline bubble
    schedule_name: Pipeline schedule name ("1f1b", "i1f1b", "dualpipe", etc.)
    warmup_steps: Number of warmup microbatches [Stack B]
    cooldown_steps: Number of cooldown microbatches [Stack B]
    steady_steps: Number of steady-state microbatches [Stack B]

    Config info
    ----------
    config_summary: Human-readable config summary (str) [Stack B] or detailed dict [Stack A]
    warnings: Validation warnings [Stack A]

    Model info
    ---------
    total_params: Total model parameters [Stack B]
    """

    # Core timing metrics
    step_time_ms: float = 0.0
    per_stage: list = field(default_factory=list)  # List[StageTime] [Stack A]
    per_stage_ms: float = 0.0  # [Stack B]

    # Efficiency metrics
    mfu: float = 0.0
    hfu: float = 0.0

    # FLOPs breakdown
    total_flops: float = 0.0  # [Stack A]
    forward_flops: float = 0.0  # [Stack B]
    backward_flops: float = 0.0  # [Stack B]
    training_flops: float = 0.0  # [Stack B]

    # Memory metrics
    memory: MemBreakdown | None = None  # [Stack A]
    memory_breakdown: dict = field(default_factory=dict)  # [Stack B]

    # Pipeline metrics
    bubble_fraction: float = 0.0
    schedule_name: str = "1f1b"
    warmup_steps: int = 0  # [Stack B]
    cooldown_steps: int = 0  # [Stack B]
    steady_steps: int = 0  # [Stack B]

    # Config info
    config_summary: str | dict = ""  # [Stack B] uses str, [Stack A] uses dict
    warnings: list[str] = field(default_factory=list)  # [Stack A]

    # Model info
    total_params: int = 0  # [Stack B]

    def __post_init__(self) -> None:
        """Sync total_flops with training_flops for unified contract.

        When training_flops is set but total_flops is not (defaults to 0.0),
        copy training_flops to total_flops. This ensures both Stack A and Stack B
        callers see consistent FLOPs data regardless of which field they access.
        """
        # If total_flops is default (0.0) but training_flops is set, sync them
        if self.total_flops == 0.0 and self.training_flops > 0:
            self.total_flops = self.training_flops

    def to_dict(self) -> dict:
        """Convert report to JSON-serializable dict."""
        result = {
            "step_time_ms": self.step_time_ms,
            "mfu": self.mfu,
            "hfu": self.hfu,
            "total_flops": self.total_flops,
            "bubble_fraction": self.bubble_fraction,
            "schedule_name": self.schedule_name,
        }

        # Add optional fields if present
        if self.per_stage_ms > 0:
            result["per_stage_ms"] = self.per_stage_ms
        if self.forward_flops > 0:
            result["forward_flops"] = self.forward_flops
        if self.backward_flops > 0:
            result["backward_flops"] = self.backward_flops
        if self.training_flops > 0:
            result["training_flops"] = self.training_flops
        if self.memory:
            result["memory_gb"] = self.memory.to_gb()
        if self.memory_breakdown:
            result["memory_breakdown_gb"] = {
                k: v / 1e9 for k, v in self.memory_breakdown.items()
            }
        if self.warmup_steps > 0:
            result["warmup_steps"] = self.warmup_steps
        if self.cooldown_steps > 0:
            result["cooldown_steps"] = self.cooldown_steps
        if self.steady_steps > 0:
            result["steady_steps"] = self.steady_steps
        if self.total_params > 0:
            result["total_params"] = self.total_params
        if self.warnings:
            result["warnings"] = self.warnings

        # Config summary (handle both str and dict)
        if isinstance(self.config_summary, str):
            result["config_summary"] = self.config_summary
        elif isinstance(self.config_summary, dict):
            result["config_summary"] = self.config_summary

        return result

    def summary(self) -> str:
        """Return a human-readable summary (Stack B format)."""
        # For Stack A reports with dict config_summary, convert to string
        config_str = self.config_summary
        if isinstance(self.config_summary, dict):
            config_str = ", ".join(f"{k}={v}" for k, v in self.config_summary.items())

        lines = [
            "Training Estimation Report",
            "=" * 40,
            f"Config: {config_str}",
            "",
            "Timing:",
            f"  Step time: {self.step_time_ms:.2f} ms",
            f"  Per-stage: {self.per_stage_ms:.2f} ms" if self.per_stage_ms > 0 else "",
            "",
            "Efficiency:",
            f"  MFU: {self.mfu:.1%}",
            f"  HFU: {self.hfu:.1%}",
            "",
            "FLOPs:",
        ]

        if self.training_flops > 0:
            lines.append(f"  Training: {self.training_flops/1e12:.2f} TFLOPs")
        if self.forward_flops > 0:
            lines.append(f"  Forward: {self.forward_flops/1e12:.2f} TFLOPs")
        if self.backward_flops > 0:
            lines.append(f"  Backward: {self.backward_flops/1e12:.2f} TFLOPs")

        lines.extend([
            "",
            "Memory (per GPU):",
        ])

        if self.memory:
            for k, v in self.memory.to_gb().items():
                lines.append(f"  {k}: {v:.2f} GB")
        elif self.memory_breakdown:
            for k, v in self.memory_breakdown.items():
                lines.append(f"  {k}: {v/1e9:.2f} GB")

        lines.extend([
            "",
            "Pipeline:",
            f"  Warmup steps: {self.warmup_steps}" if self.warmup_steps > 0 else "",
            f"  Steady steps: {self.steady_steps}" if self.steady_steps > 0 else "",
            f"  Cooldown steps: {self.cooldown_steps}" if self.cooldown_steps > 0 else "",
            f"  Bubble fraction: {self.bubble_fraction:.1%}",
            "",
            f"Total params: {self.total_params/1e9:.2f}B" if self.total_params > 0 else "",
        ])

        # Filter out empty lines
        return "\n".join(line for line in lines if line)
