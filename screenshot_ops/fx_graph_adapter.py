"""Convert a torch.fx.GraphModule into a ComputeGraph (NetworkX DiGraph).

Preserves FX node provenance so that downstream fusion can distinguish:

- **inputs**  (placeholder → external function argument)
- **parameters** (get_attr → model weight/bias)
- **ops**     (call_function → aten operator)
- **constants** (literal scalars in args)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.fx

from screenshot_ops.compute_graph import ComputeGraph


class FXGraphAdapter:
    """Convert torch.fx.GraphModule → ComputeGraph."""

    def convert(self, gm: torch.fx.GraphModule, name: str = "model") -> ComputeGraph:
        graph = ComputeGraph(name)
        node_map: Dict[torch.fx.Node, int] = {}

        for fx_node in gm.graph.nodes:
            if fx_node.op == "placeholder":
                nid = graph.add_node(
                    op_name="placeholder",
                    name=fx_node.name,
                    attrs={
                        "kind": "input",
                        "fx_target": str(fx_node.target),
                    },
                )
                node_map[fx_node] = nid

            elif fx_node.op == "get_attr":
                attr_val = self._fetch_attr(gm, fx_node.target)
                shape_str = str(list(attr_val.shape)) if isinstance(attr_val, torch.Tensor) else "?"
                dtype_str = str(attr_val.dtype) if isinstance(attr_val, torch.Tensor) else "?"
                nid = graph.add_node(
                    op_name="get_attr",
                    name=fx_node.name,
                    attrs={
                        "kind": "parameter",
                        "fx_target": str(fx_node.target),
                        "shape": shape_str,
                        "dtype": dtype_str,
                    },
                )
                node_map[fx_node] = nid

            elif fx_node.op == "call_function":
                input_kinds = self._classify_args(fx_node, node_map, graph)
                nid = graph.add_node(
                    op_name=str(fx_node.target),
                    name=fx_node.name,
                    attrs={
                        "kind": "op",
                        "fx_target": str(fx_node.target),
                        "input_kinds": input_kinds,
                    },
                )
                node_map[fx_node] = nid

                for arg in fx_node.args:
                    if isinstance(arg, torch.fx.Node) and arg in node_map:
                        graph.add_edge(node_map[arg], nid)

            elif fx_node.op == "output":
                pass

        self._gm = gm
        self._fx_node_map = node_map
        return graph

    def extract_io_map(
        self,
        node_ids: List[int],
        graph: ComputeGraph,
    ) -> Dict[str, Any]:
        """For a set of ComputeGraph node IDs (a fused group), classify
        each external input as 'input', 'parameter', or 'constant'.

        Returns dict with keys: input_map, parameter_map, constant_map, output_map.
        """
        group_set = set(node_ids)

        input_map: List[Dict[str, Any]] = []
        parameter_map: List[Dict[str, Any]] = []
        constant_map: List[Dict[str, Any]] = []
        output_map: List[Dict[str, Any]] = []

        seen_inputs: set = set()
        seen_params: set = set()
        seen_outputs: set = set()

        for nid in node_ids:
            attrs = graph.node_attrs(nid)
            if attrs.get("attrs", {}).get("kind") != "op":
                continue

            for pred_id in graph.predecessors(nid):
                if pred_id in group_set:
                    continue
                pred_attrs = graph.node_attrs(pred_id)
                pred_kind = pred_attrs.get("attrs", {}).get("kind", "")

                if pred_kind == "input" and pred_id not in seen_inputs:
                    seen_inputs.add(pred_id)
                    input_map.append({
                        "name": pred_attrs.get("attrs", {}).get("fx_target", pred_attrs.get("name", "")),
                        "kind": "input",
                        "shape": pred_attrs.get("attrs", {}).get("shape", ""),
                        "dtype": pred_attrs.get("attrs", {}).get("dtype", ""),
                        "consumer_op": attrs.get("attrs", {}).get("fx_target", ""),
                    })
                elif pred_kind == "parameter" and pred_id not in seen_params:
                    seen_params.add(pred_id)
                    parameter_map.append({
                        "name": pred_attrs.get("attrs", {}).get("fx_target", pred_attrs.get("name", "")),
                        "kind": "parameter",
                        "shape": pred_attrs.get("attrs", {}).get("shape", ""),
                        "dtype": pred_attrs.get("attrs", {}).get("dtype", ""),
                        "consumer_op": attrs.get("attrs", {}).get("fx_target", ""),
                    })

            input_kinds = attrs.get("attrs", {}).get("input_kinds", [])
            for ik in input_kinds:
                if ik.get("kind") == "constant":
                    val = ik.get("value", "")
                    if val not in [c["value"] for c in constant_map]:
                        constant_map.append({
                            "kind": "constant",
                            "value": val,
                            "consumer_op": attrs.get("attrs", {}).get("fx_target", ""),
                        })

        for nid in node_ids:
            attrs = graph.node_attrs(nid)
            if attrs.get("attrs", {}).get("kind") != "op":
                continue
            for succ_id in graph.successors(nid):
                if succ_id not in group_set:
                    out_key = attrs.get("name", "")
                    if out_key not in seen_outputs:
                        seen_outputs.add(out_key)
                        output_map.append({
                            "name": out_key,
                            "producer_op": attrs.get("attrs", {}).get("fx_target", ""),
                        })

        return {
            "input_map": input_map,
            "parameter_map": parameter_map,
            "constant_map": constant_map,
            "output_map": output_map,
        }

    def _fetch_attr(self, gm: torch.fx.GraphModule, target: str) -> Any:
        atoms = target.split(".")
        obj = gm
        for atom in atoms:
            obj = getattr(obj, atom)
        return obj

    def _classify_args(
        self,
        fx_node: torch.fx.Node,
        node_map: Dict[torch.fx.Node, int],
        graph: ComputeGraph,
    ) -> List[Dict[str, Any]]:
        result = []
        for arg in fx_node.args:
            if isinstance(arg, torch.fx.Node):
                if arg in node_map:
                    pred_attrs = graph.node_attrs(node_map[arg])
                    result.append({
                        "kind": pred_attrs.get("attrs", {}).get("kind", "unknown"),
                        "name": arg.name,
                    })
                else:
                    result.append({"kind": "unknown", "name": arg.name})
            else:
                result.append({"kind": "constant", "value": repr(arg)})
        return result
