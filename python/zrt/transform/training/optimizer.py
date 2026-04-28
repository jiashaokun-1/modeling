from __future__ import annotations

import logging
from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.ir.edge import Edge
from python.zrt.ir.param_count import count_params
from python.zrt.transform.base import GraphPass
from python.zrt.transform.context import TransformContext

logger = logging.getLogger(__name__)


class OptimizerPass(GraphPass):
    """OptimizerPass for optimizer step annotation.

    Adds an ``optimizer_step`` node to the backward graph to represent
    the optimizer update cost. This node is annotated with:
      - state_bytes: Optimizer state memory per rank
      - step_flops: Optimizer step FLOPs

    Runs only on backward-phase graphs (when metadata["phase"] indicates
    backward or when individual nodes have backward phase annotations).
    """
    name = "optimizer"

    def run(self, graph: OpGraph, ctx: TransformContext) -> OpGraph:
        """Run OptimizerPass on the graph.

        Args:
            graph: Input OpGraph
            ctx: TransformContext with training config

        Returns:
            New OpGraph with optimizer step node
        """
        g = graph.clone()
        if not ctx.training or not ctx.is_training:
            return g

        # Check graph-level phase (for unified stitched graphs)
        graph_phase = g.metadata.get("phase", "")

        # For stitched graphs, only add optimizer to backward
        if graph_phase and graph_phase not in ("train_backward", "backward"):
            return g

        # For non-stitched graphs, check if any node is a backward node
        has_backward = any(
            n.annotations.get("phase") in ("bwd", "backward", "train_backward")
            for n in g.nodes.values()
        )
        if not has_backward and not graph_phase:
            return g

        # Calculate total parameters on rank using graph parameter count
        total_params = count_params(g)
        tp = ctx.parallel.tp if ctx.parallel else 1
        dp = ctx.parallel.dp if ctx.parallel else 1
        pp = ctx.parallel.pp if ctx.parallel else 1
        cp = getattr(ctx.parallel, "cp", 1) if ctx.parallel else 1

        # Apply PP sharding: optimizer runs in the last stage, so only 1/pp of parameters
        if pp > 1:
            params = int(total_params / pp)
        else:
            params = total_params

        # Apply TP/DP sharding
        zero_stage = ctx.training.zero_stage if ctx.training else 0
        if zero_stage >= 3:
            params //= (tp * dp)
        else:
            params //= tp

        opt = ctx.training.optimizer if ctx.training else "adam"

        # Optimizer runs in the last stage (after all backward ops complete)
        optimizer_stage_id = max(0, pp - 1)

        # Create optimizer step node
        step_node = OpNode(
            id="optimizer_step",
            op_type=f"optimizer.{opt}",
            inputs=[],
            outputs=[],
            attrs={
                "optimizer": opt,
                "params": params,
                "state_bytes": self._opt_state_bytes(opt, params),
                "step_flops": self._opt_step_flops(opt, params),
            },
            scope="optimizer.step",
            category="compute",
        )
        step_node.annotations["phase"] = "bwd"
        step_node.annotations["stage_id"] = optimizer_stage_id
        step_node.annotations["optimizer_step"] = True

        # Append the optimizer step node at the end of the graph
        self._append_at_end(g, step_node)

        return g

    def _opt_state_bytes(self, optimizer: str, params: int, master_bytes: int = 4) -> int:
        """Calculate optimizer state bytes.

        Args:
            optimizer: Optimizer name
            params: Number of parameters
            master_bytes: Bytes per parameter in master dtype (default FP32=4)

        Returns:
            Optimizer state bytes.
            - Adam: master copy + momentum (m) + variance (v) = 3 × P × master_dtype
            - Muon: master copy + momentum matrix ≈ 2 × P × master_dtype
        """
        if optimizer in ("adam", "adamw"):
            # Adam: master copy + m + v = 3 × P × master_dtype
            return params * master_bytes * 3
        elif optimizer == "muon":
            # Muon: master copy + momentum matrix ≈ 2.1 × P × master_dtype
            return int(params * master_bytes * 2.1)
        else:
            # Default: 3 × P × master_dtype
            return params * master_bytes * 3

    def _opt_step_flops(self, optimizer: str, params: int) -> int:
        """Calculate optimizer step FLOPs.

        Args:
            optimizer: Optimizer name
            params: Number of parameters

        Returns:
            Optimizer step FLOPs
        """
        if optimizer in ("adam", "adamw"):
            # Adam: ~12 FLOPs per parameter (updates + momentum)
            return params * 12
        elif optimizer == "muon":
            # Muon: ~6 FLOPs per parameter (simpler update)
            return params * 6
        else:
            # Default: ~8 FLOPs per parameter
            return params * 8

    def _append_at_end(self, graph: OpGraph, new_node: OpNode) -> None:
        """Append a node at the end of the graph.

        Finds all sink nodes (no outgoing edges) and connects them
        to the new node, then adds the new node to the graph.

        Args:
            graph: OpGraph to modify
            new_node: OpNode to append
        """
        # Build adjacency to find sink nodes
        has_out_edge = set()
        for edge in graph.edges:
            has_out_edge.add(edge.src)

        sink_nodes = [
            graph.nodes[nid] for nid in graph.nodes
            if nid not in has_out_edge
        ]

        # Add the new node first
        graph.nodes[new_node.id] = new_node
        if new_node.id not in graph._pred:
            graph._pred[new_node.id] = []
        if new_node.id not in graph._succ:
            graph._succ[new_node.id] = []

        # Connect all sink nodes to the new node
        for sink_node in sink_nodes:
            if sink_node.outputs:
                graph.edges.append(Edge(
                    src=sink_node.id,
                    src_idx=0,
                    dst=new_node.id,
                    dst_idx=0,
                    tensor=sink_node.outputs[0],
                ))
                # Update adjacency structures
                graph._succ[sink_node.id].append(new_node.id)
                graph._pred[new_node.id].append(sink_node.id)
