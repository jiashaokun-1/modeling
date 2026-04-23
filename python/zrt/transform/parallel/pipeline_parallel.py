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
        """Insert one comm.send_recv node per stage boundary."""
        topo = graph.topo_sort()
        topo_rank = {n.id: i for i, n in enumerate(topo)}

        for i in range(len(stages) - 1):
            src_stage = stages[i]
            if not src_stage.node_ids:
                continue

            # Last node in the sending stage (highest topo rank)
            valid_src = [nid for nid in src_stage.node_ids if nid in topo_rank]
            if not valid_src:
                continue
            last_src_id = max(valid_src, key=lambda nid: topo_rank[nid])
            last_src = graph.nodes[last_src_id]

            # Size of the activation crossing the boundary
            act_bytes = sum(t.mem_bytes for t in last_src.outputs)
            if act_bytes == 0:
                act_bytes = 4  # sentinel for scalar activation

            # Build output TensorMeta for the received activation
            if last_src.outputs:
                ref_out = last_src.outputs[0]
                recv_tensor = TensorMeta(
                    id=f"p2p_act_{i}_{i+1}_0",
                    shape=ref_out.shape,
                    dtype=ref_out.dtype,
                    mem_bytes=ref_out.mem_bytes,
                )
            else:
                recv_tensor = TensorMeta.from_shape_dtype(
                    f"p2p_act_{i}_{i+1}_0", (1,), DType.BF16
                )

            p2p_id = f"comm_p2p_{i}_{i+1}"
            p2p_node = OpNode(
                id=p2p_id,
                op_type="comm.send_recv",
                inputs=list(last_src.outputs),
                outputs=[recv_tensor],
                attrs={
                    "src_stage": i,
                    "dst_stage": i + 1,
                    "message_size_bytes": act_bytes,
                },
                scope=f"pipeline.p2p.stage{i}_to_{i+1}",
                category="communication",
            )
            p2p_node.annotations["stage_id"] = i  # belongs to the sending stage

            # Wire: last_src_id → p2p_node
            p2p_edge = Edge(
                src=last_src_id,
                src_idx=0,
                dst=p2p_id,
                dst_idx=0,
                tensor=last_src.outputs[0] if last_src.outputs else None,
            )
            graph.insert_after(last_src_id, p2p_node, [p2p_edge])

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
