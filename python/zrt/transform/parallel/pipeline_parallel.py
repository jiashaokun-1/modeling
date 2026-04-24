"""Pipeline Parallel pass: stage assignment + P2P boundary insertion.

For each node annotates ``node.annotations["stage_id"] = int`` (0-indexed).
Inserts ``comm.send_recv`` nodes at stage boundaries to model activation
transfer latency between adjacent pipeline stages.

Layer partitioning uses greedy bin-packing by accumulated per-layer compute
load (``compute_us`` → ``latency_us`` → ``flops`` → 1.0 fallback), which
approximates load-balanced stage assignment without requiring a pre-pass.
An explicit ``TrainingConfig.pp_layer_assignment`` list overrides this.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, TYPE_CHECKING

from python.zrt.ir.edge import Edge
from python.zrt.ir.node import OpNode
from python.zrt.ir.types import TensorMeta, DType
from python.zrt.transform.base import GraphPass

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext

logger = logging.getLogger(__name__)


@dataclass
class LayerGroup:
    """Layers assigned to a single pipeline stage."""
    stage_id: int
    layer_ids: List[int] = field(default_factory=list)
    node_ids: Set[str] = field(default_factory=set)
    total_compute_us: float = 0.0


class PipelineParallelPass(GraphPass):
    """Annotate pipeline stage IDs and insert P2P comm nodes at boundaries.

    Pass order: runs in the ``split`` stage, after TP/EP but before Fusion.
    Requires ``ctx.parallel.pp > 1`` to do anything.

    Annotations written
    -------------------
    ``node.annotations["stage_id"]`` : int  — 0-indexed pipeline stage.
    """

    name = "pipeline_parallel"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        from python.zrt.ir.graph import OpGraph  # local to avoid circular

        pp = ctx.parallel.pp if ctx.parallel else 1

        if pp <= 1:
            # Annotate everything as stage 0 for consistency with downstream passes
            g = graph.clone()
            for node in g.nodes.values():
                node.annotations.setdefault("stage_id", 0)
            return g

        g = graph.clone()

        pp_layer_assignment: Optional[List[int]] = (
            getattr(ctx.training, "pp_layer_assignment", None)
            if ctx.training else None
        )

        # 1. Build layer_id → {node_ids} and per-layer compute load
        layer_nodes: Dict[int, Set[str]] = {}
        layer_load:  Dict[int, float]    = {}

        for node in g.nodes.values():
            try:
                layer_idx = int(node.layer) if node.layer else -1
            except (ValueError, TypeError):
                layer_idx = -1

            layer_nodes.setdefault(layer_idx, set()).add(node.id)

            load = (
                node.annotations.get("compute_us")
                or node.annotations.get("latency_us")
                or node.annotations.get("flops", 0) / 1e12
                or 1.0
            )
            layer_load[layer_idx] = layer_load.get(layer_idx, 0.0) + load

        sorted_layers = sorted(k for k in layer_nodes if k >= 0)

        # 2. Partition layers → pp stages
        stages = self._partition(sorted_layers, layer_nodes, layer_load,
                                 pp, pp_layer_assignment)

        # 3. Build lookup: layer_id → stage_id
        layer_to_stage: Dict[int, int] = {
            lid: s.stage_id
            for s in stages
            for lid in s.layer_ids
        }

        # 4. Annotate stage_id on every node
        for node in g.nodes.values():
            try:
                layer_idx = int(node.layer) if node.layer else -1
            except (ValueError, TypeError):
                layer_idx = -1
            node.annotations["stage_id"] = layer_to_stage.get(layer_idx, 0)

        # 5. Insert P2P send_recv at each stage boundary
        self._insert_p2p_nodes(g, stages)

        # 6. Warn if imbalanced
        self._check_balance(stages)

        return g

    # ── partitioning ──────────────────────────────────────────────────────────

    def _partition(
        self,
        sorted_layers: List[int],
        layer_nodes: Dict[int, Set[str]],
        layer_load: Dict[int, float],
        pp: int,
        explicit: Optional[List[int]],
    ) -> List[LayerGroup]:
        stages = [LayerGroup(stage_id=i) for i in range(pp)]

        if not sorted_layers:
            return stages

        if explicit and len(explicit) == len(sorted_layers):
            # User-specified assignment: explicit[i] is the stage for sorted_layers[i]
            for idx, layer_id in enumerate(sorted_layers):
                s_idx = max(0, min(explicit[idx], pp - 1))
                stages[s_idx].layer_ids.append(layer_id)
                stages[s_idx].node_ids.update(layer_nodes[layer_id])
                stages[s_idx].total_compute_us += layer_load.get(layer_id, 0.0)
        else:
            # Greedy bin-packing: always assign to the lightest stage
            stage_load = [0.0] * pp
            for layer_id in sorted_layers:
                min_s = int(min(range(pp), key=lambda i: stage_load[i]))
                stages[min_s].layer_ids.append(layer_id)
                stages[min_s].node_ids.update(layer_nodes[layer_id])
                load = layer_load.get(layer_id, 0.0)
                stages[min_s].total_compute_us += load
                stage_load[min_s] += load

        return stages

    # ── P2P insertion ─────────────────────────────────────────────────────────

    def _insert_p2p_nodes(self, graph: "OpGraph",
                          stages: List[LayerGroup]) -> None:
        """Insert comm.send_recv at stage boundaries and rewire receiver edges.

        Detects edges that cross stage boundaries (both fwd s→s+1 and bwd
        s+1→s), inserts one comm node per crossing edge, and rewires
        receiver-side edges so the comm node is on the real dependency path.
        """
        node_stage = {nid: node.annotations.get("stage_id", 0)
                      for nid, node in graph.nodes.items()}

        # Collect crossing edges (source and dest in different stages)
        crossing_edges: List[Edge] = []
        for edge in list(graph.edges):
            ss = node_stage.get(edge.src, -1)
            ds = node_stage.get(edge.dst, -1)
            if ss >= 0 and ds >= 0 and ss != ds:
                crossing_edges.append(edge)

        p2p_idx = 0
        for edge in crossing_edges:
            ss = node_stage[edge.src]
            ds = node_stage[edge.dst]
            direction = "fwd" if ds > ss else "bwd"

            tensor = edge.tensor
            act_bytes = tensor.mem_bytes if tensor else 4

            if tensor:
                recv_tensor = TensorMeta(
                    id=f"p2p_{direction}_{ss}_{ds}_{p2p_idx}",
                    shape=tensor.shape, dtype=tensor.dtype,
                    mem_bytes=tensor.mem_bytes,
                )
            else:
                recv_tensor = TensorMeta.from_shape_dtype(
                    f"p2p_{direction}_{ss}_{ds}_{p2p_idx}", (1,), DType.BF16
                )

            p2p_id = f"comm_p2p_{direction}_{ss}_{ds}_{p2p_idx}"
            p2p_idx += 1

            p2p_node = OpNode(
                id=p2p_id,
                op_type="comm.send_recv",
                inputs=[tensor] if tensor else [],
                outputs=[recv_tensor],
                attrs={
                    "src_stage": ss,
                    "dst_stage": ds,
                    "message_size_bytes": act_bytes,
                },
                scope=f"pipeline.p2p.{direction}.stage{ss}_to_{ds}",
                category="communication",
            )
            p2p_node.annotations["stage_id"] = ss
            p2p_node.annotations["phase"] = direction

            # Insert after source node
            graph.insert_after(edge.src, p2p_node, [Edge(
                src=edge.src, src_idx=0,
                dst=p2p_id, dst_idx=0,
                tensor=tensor,
            )])

            # Rewire: remove original crossing edge, add p2p→dst edge
            if edge in graph.edges:
                graph.edges.remove(edge)
            graph.edges.append(Edge(
                src=p2p_id, src_idx=0,
                dst=edge.dst, dst_idx=edge.dst_idx,
                tensor=recv_tensor,
            ))
            graph._rebuild_adjacency()

    # ── balance check ─────────────────────────────────────────────────────────

    def _check_balance(self, stages: List[LayerGroup]) -> None:
        loads = [s.total_compute_us for s in stages if s.total_compute_us > 0]
        if len(loads) < 2:
            return
        ratio = max(loads) / min(loads)
        if ratio > 1.5:
            logger.warning(
                "PipelineParallelPass: stage imbalance %.2fx (max/min). "
                "Consider setting --pp-layer-assignment.",
                ratio,
            )
