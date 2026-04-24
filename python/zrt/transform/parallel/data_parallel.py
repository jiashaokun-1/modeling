from __future__ import annotations

from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.transform.base import GraphPass
from python.zrt.transform.context import TransformContext


class DataParallelPass(GraphPass):
    """Data Parallel pass for gradient reduction."""
    name = "data_parallel"

    def run(self, graph: OpGraph, ctx: TransformContext) -> OpGraph:
        """Run data parallel pass on the graph.
        
        Args:
            graph: Input OpGraph
            ctx: TransformContext with parallel config and training config
            
        Returns:
            New OpGraph with gradient reduction nodes
        """
        if not ctx.is_training or ctx.parallel.dp <= 1 or graph.phase != "train_backward":
            return graph
        
        g = graph.clone()
        dp = ctx.parallel.dp
        zero_stage = ctx.training.zero_stage if ctx.training else 0

        # 1. Collect gradient bytes from the graph
        grad_bytes = self._collect_grad_bytes(g)
        
        # 2. Determine collective type based on Zero stage
        if zero_stage == 0:
            collective = "all_reduce"
        else:
            collective = "reduce_scatter"
        
        # 3. Create gradient reduction communication node
        comm_node = OpNode(
            id="comm_grad_reduce",
            op_type=f"comm.{collective}",
            inputs=[],
            outputs=[],
            attrs={
                "group_size": dp,
                "collective": collective,
                "role": "dp_grad_reduce",
                "bucket_bytes": grad_bytes
            },
            scope="data_parallel.grad_reduce",
            category="communication"
        )
        
        comm_node.annotations["inserted_by"] = "data_parallel_pass"
        comm_node.annotations["dp_comm"] = True
        dp_overlap = getattr(ctx.training, "dp_overlap_in_bubble", True) if ctx.training else False
        if dp_overlap:
            comm_node.annotations["overlap_in_bubble"] = True
        
        # 4. Append the communication node at the end of the graph
        self._append_at_end(g, comm_node)
        
        return g

    def _collect_grad_bytes(self, graph: OpGraph) -> int:
        """Collect gradient bytes from the graph.
        
        Args:
            graph: OpGraph to collect gradients from
            
        Returns:
            Total gradient bytes
        """
        grad_bytes = 0
        
        for node in graph.topo_sort():
            # Check if node is a gradient operation
            if "grad" in node.op_type.lower() or "backward" in node.op_type.lower():
                # Add output tensor sizes
                for output in node.outputs:
                    grad_bytes += output.mem_bytes
        
        return grad_bytes

    def _append_at_end(self, graph: OpGraph, new_node: OpNode):
        """Append a node at the end of the graph.
        
        Args:
            graph: OpGraph to modify
            new_node: OpNode to append
        """
        # Find all nodes with no successors
        end_nodes = []
        for node_id, node in graph.nodes.items():
            has_successors = False
            for edge in graph.edges:
                if edge.src == node_id:
                    has_successors = True
                    break
            if not has_successors:
                end_nodes.append(node)
        
        # If there are end nodes, connect them to the new node
        if end_nodes:
            # Create edges from all end nodes to the new node
            for end_node in end_nodes:
                # Use the first output of the end node as input to the new node
                if end_node.outputs:
                    # Create a new edge
                    from python.zrt.ir.edge import Edge
                    edge = Edge(
                        src=end_node.id,
                        src_idx=0,
                        dst=new_node.id,
                        dst_idx=0,
                        tensor=end_node.outputs[0]
                    )
                    graph.edges.append(edge)
        
        # Add the new node to the graph
        graph.nodes[new_node.id] = new_node
