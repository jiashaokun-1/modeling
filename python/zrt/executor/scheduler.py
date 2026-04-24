"""DAGScheduler: list-scheduling simulation over a transformed OpGraph.

Algorithm
---------
Iterate nodes in topological order.  For each node:
    ready_time  = max finish time of all predecessor nodes  (data dependency)
    stream_free = earliest time the assigned stream is idle (resource constraint)
    start_us    = max(ready_time, stream_free)
    end_us      = start_us + latency_us

latency_us is read from node.annotations["latency_us"] when present.
If missing and hw_spec is provided, the Roofline backend estimates it.
If neither is available, 1 µs is used as a conservative placeholder.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.ir.node import OpNode
    from python.zrt.hardware.spec import HardwareSpec


# ── output types ──────────────────────────────────────────────────────────────

@dataclass
class ScheduledOp:
    """Timing record for a single scheduled operator."""
    node_id:     str
    stream_id:   int
    stream_type: str    # "compute" | "comm"
    start_us:    float
    end_us:      float
    latency_us:  float
    op_type:     str
    category:    str    # "compute" | "communication" | "memory"
    phase:       str = ""  # "fwd" | "bwd" | ""

    def __repr__(self) -> str:
        return (
            f"ScheduledOp({self.node_id}, stream={self.stream_id}, "
            f"{self.start_us:.2f}→{self.end_us:.2f}µs)"
        )


@dataclass
class Timeline:
    """Complete scheduling result for one graph phase.

    Summary statistics
    ------------------
    total_latency_us : wall-clock time from first op start to last op end
    compute_time_us  : sum of individual compute-op latencies (serial upper bound)
    comm_time_us     : sum of individual comm-op latencies    (serial upper bound)
    overlap_us       : comm time hidden behind compute
                       = compute_time + comm_time - total_latency  (clamped ≥ 0)
    """
    scheduled_ops: list[ScheduledOp] = field(default_factory=list)
    graph_name:    str = ""
    phase:         str = ""

    # ── summary ───────────────────────────────────────────────────────────────

    @property
    def total_latency_us(self) -> float:
        return max((op.end_us for op in self.scheduled_ops), default=0.0)

    @property
    def total_latency_ms(self) -> float:
        return self.total_latency_us / 1_000.0

    @property
    def compute_time_us(self) -> float:
        return sum(op.latency_us for op in self.scheduled_ops
                   if op.stream_type == "compute")

    @property
    def comm_time_us(self) -> float:
        return sum(op.latency_us for op in self.scheduled_ops
                   if op.stream_type == "comm")

    @property
    def overlap_us(self) -> float:
        return max(0.0, self.compute_time_us + self.comm_time_us
                   - self.total_latency_us)

    # ── queries ───────────────────────────────────────────────────────────────

    def phase_latency(self, phase: str) -> float:
        """Wall-clock latency span of ops matching *phase*."""
        ops = [op for op in self.scheduled_ops if op.phase == phase]
        if not ops:
            return 0.0
        return max(op.end_us for op in ops) - min(op.start_us for op in ops)

    def ops_on_stream(self, stream_id: int) -> list[ScheduledOp]:
        return sorted(
            [op for op in self.scheduled_ops if op.stream_id == stream_id],
            key=lambda o: o.start_us,
        )

    def compute_ops(self) -> list[ScheduledOp]:
        return [op for op in self.scheduled_ops if op.stream_type == "compute"]

    def comm_ops(self) -> list[ScheduledOp]:
        return [op for op in self.scheduled_ops if op.stream_type == "comm"]

    def __repr__(self) -> str:
        return (
            f"Timeline('{self.graph_name}', {len(self.scheduled_ops)} ops, "
            f"total={self.total_latency_us:.2f}µs, "
            f"overlap={self.overlap_us:.2f}µs)"
        )


# ── scheduler ─────────────────────────────────────────────────────────────────

class DAGScheduler:
    """List scheduler for annotated OpGraphs.

    Parameters
    ----------
    hw_spec : HardwareSpec | None
        Used to estimate latency when a node lacks a ``latency_us`` annotation.
        If None and annotation is absent, 1 µs is used as a placeholder.
    """

    def __init__(self, hw_spec: "HardwareSpec | None" = None) -> None:
        self._hw = hw_spec
        self._roofline = None   # lazy-initialised if needed

    def schedule(self, graph: "OpGraph") -> Timeline:
        """Schedule all nodes and return a Timeline.

        Nodes must already carry ``stream_id`` annotations (from StreamAssignPass).
        ``latency_us`` annotations are used when present; otherwise estimated.
        """
        finish:       dict[str, float] = {}   # node_id → end_us
        stream_avail: dict[int, float] = {}   # stream_id → next_free_us
        scheduled:    list[ScheduledOp] = []

        for node in graph.topo_sort():
            stream_id   = node.annotations.get("stream_id", 0)
            stream_type = node.annotations.get("stream_type", "compute")
            lat         = self._latency(node)

            # data dependency: wait for all predecessors
            pred_done = max(
                (finish[p] for p in graph.predecessors(node.id) if p in finish),
                default=0.0,
            )
            # resource constraint: wait for stream to be free
            s_free = stream_avail.get(stream_id, 0.0)

            start = max(pred_done, s_free)
            end   = start + lat

            finish[node.id]         = end
            stream_avail[stream_id] = end

            scheduled.append(ScheduledOp(
                node_id     = node.id,
                stream_id   = stream_id,
                stream_type = stream_type,
                start_us    = start,
                end_us      = end,
                latency_us  = lat,
                op_type     = node.op_type,
                category    = node.category,
                phase       = node.annotations.get("phase", ""),
            ))

        return Timeline(
            scheduled_ops = scheduled,
            graph_name    = graph.name,
            phase         = graph.phase,
        )

    # ── internal ──────────────────────────────────────────────────────────────

    def _latency(self, node: "OpNode") -> float:
        if "latency_us" in node.annotations:
            return float(node.annotations["latency_us"])
        if self._hw is not None:
            return self._roofline_estimate(node)
        return 1.0

    def _roofline_estimate(self, node: "OpNode") -> float:
        if self._roofline is None:
            from python.zrt.simulator.backends.roofline import RooflineSimulator
            self._roofline = RooflineSimulator()
        return self._roofline.simulate(node, self._hw).latency_us
