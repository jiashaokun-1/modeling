"""Build a NetworkX ComputeGraph from RecordingDispatch records."""
from __future__ import annotations

from typing import Any, Dict, List

from screenshot_ops.compute_graph import ComputeGraph


def build_compute_graph(
    records: List[Dict[str, Any]],
    passthroughs: Dict[int, int],
) -> ComputeGraph:
    """Build a ComputeGraph from dispatch records and passthrough mapping.

    Each recorded op becomes a node; edges represent tensor data dependencies
    resolved through the passthrough chain (skipped reshape/view ops).

    Parameters
    ----------
    records:
        Op records from ``RecordingDispatch.records``.
    passthroughs:
        ``TensorTracker.passthroughs`` mapping skip-op output_id → input_id.
    """
    graph = ComputeGraph("model")

    idx_to_node: Dict[int, int] = {}
    tensor_to_node: Dict[int, int] = {}

    def _resolve_tensor(tid: int) -> int:
        visited: set = set()
        while tid in passthroughs:
            if tid in visited:
                break
            visited.add(tid)
            tid = passthroughs[tid]
        return tid

    for rec in records:
        aten_short = rec["aten_op"].split(".")[1] if "." in rec["aten_op"] else rec["aten_op"]
        node_name = f"{rec['component']}.{aten_short}_{rec['idx']}"

        node_id = graph.add_node(
            op_name=rec["aten_op"],
            name=node_name,
            attrs={
                "record_idx": rec["idx"],
                "module_path": rec["module_path"],
                "layer": rec["layer"],
                "component": rec["component"],
                "input_shapes": rec["input_shapes"],
                "input_dtypes": rec["input_dtypes"],
                "output_shapes": rec["output_shapes"],
                "output_dtypes": rec["output_dtypes"],
                "_input_ids": rec["_input_ids"],
                "_output_ids": rec["_output_ids"],
            },
        )
        idx_to_node[rec["idx"]] = node_id

        for tid in rec["_output_ids"]:
            tensor_to_node[tid] = node_id

        for tid in rec["_input_ids"]:
            resolved = _resolve_tensor(tid)
            src_node = tensor_to_node.get(resolved)
            if src_node is not None and src_node != node_id:
                graph.add_edge(src_node, node_id, tensor_id=resolved)

    return graph
