"""Graph-based operator fusion driven by JSON fusion rules.

Usage::

    from screenshot_ops.fusion_pass import FusionRule, FusionPass
    rules = FusionRule.from_specs(specs)
    fused_graph, result = FusionPass(rules).apply(graph)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from screenshot_ops.compute_graph import ComputeGraph


@dataclass
class FusionRule:
    """A fusion pattern loaded from JSON or derived from FusionSpec."""

    rule_name: str
    module_key: str
    aten_op_sequence: List[str]
    num_sub_ops: int = 0
    fusion_level: str = "leaf"
    occurrences: int = 1
    example_module_path: str = ""
    input_map: List[Dict] = field(default_factory=list)
    output_map: List[Dict] = field(default_factory=list)
    parameter_map: List[Dict] = field(default_factory=list)
    constant_map: List[Dict] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: str | Path) -> List["FusionRule"]:
        data = json.loads(Path(path).read_text())
        rules = []
        for entry in data:
            rules.append(cls(
                rule_name=entry.get("rule_name", entry.get("module_key", "")),
                module_key=entry.get("module_key", entry.get("module_class", "")),
                aten_op_sequence=entry["aten_op_sequence"],
                num_sub_ops=entry.get("num_sub_ops", len(entry["aten_op_sequence"])),
                fusion_level=entry.get("fusion_level", "leaf"),
                occurrences=entry.get("occurrences", 1),
                example_module_path=entry.get("example_module_path", ""),
                input_map=entry.get("input_map", []),
                output_map=entry.get("output_map", []),
                parameter_map=entry.get("parameter_map", []),
                constant_map=entry.get("constant_map", []),
            ))
        return rules

    @classmethod
    def from_specs(cls, specs) -> List["FusionRule"]:
        rules = []
        for s in specs:
            rules.append(cls(
                rule_name=s.module_key,
                module_key=s.module_key,
                aten_op_sequence=s.aten_op_sequence,
                num_sub_ops=s.num_sub_ops,
                fusion_level=s.fusion_level,
                occurrences=s.occurrences,
                example_module_path=s.example_module_path,
                input_map=s.input_map,
                output_map=s.output_map,
                parameter_map=getattr(s, "parameter_map", []),
                constant_map=getattr(s, "constant_map", []),
            ))
        return rules

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "rule_name": self.rule_name,
            "module_key": self.module_key,
            "aten_op_sequence": self.aten_op_sequence,
            "num_sub_ops": self.num_sub_ops,
            "fusion_level": self.fusion_level,
            "occurrences": self.occurrences,
            "example_module_path": self.example_module_path,
            "input_map": self.input_map,
            "output_map": self.output_map,
        }
        if self.parameter_map:
            d["parameter_map"] = self.parameter_map
        if self.constant_map:
            d["constant_map"] = self.constant_map
        return d


@dataclass
class FusionResult:
    original_nodes: int
    fused_nodes: int
    fusions_applied: List[str] = field(default_factory=list)

    @property
    def nodes_eliminated(self) -> int:
        return self.original_nodes - self.fused_nodes

    def summary(self) -> str:
        lines = [
            f"Fusion: {self.original_nodes} ops -> {self.fused_nodes} ops "
            f"({self.nodes_eliminated} eliminated)",
            f"Fusions applied: {len(self.fusions_applied)}",
        ]
        for f in self.fusions_applied[:20]:
            lines.append(f"  - {f}")
        if len(self.fusions_applied) > 20:
            lines.append(f"  ... and {len(self.fusions_applied) - 20} more")
        return "\n".join(lines)


def _strip_layer_prefix(module_path: str) -> str:
    parts = module_path.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                int(parts[i + 1])
                return ".".join(parts[i + 2:]) or module_path
            except ValueError:
                pass
    return module_path


class FusionPass:
    """Apply fusion rules to a ComputeGraph, producing a new fused graph.

    Supports two modes controlled by ``mode``:
      - ``"fx"``: De-fusion on FX-derived graphs.
      - ``"module_key"``: Module-key grouping (for TorchDispatchMode-derived
        graphs).  Matches by stripping the layer prefix from module_path.

    In both modes, the original graph is not modified.
    """

    def __init__(self, rules: List[FusionRule], mode: str = "module_key"):
        self.rules = rules
        self.mode = mode
        self._seq_to_rule: Dict[Tuple[str, ...], FusionRule] = {}
        self._key_to_rule: Dict[str, FusionRule] = {}
        for r in rules:
            key = tuple(r.aten_op_sequence)
            if key not in self._seq_to_rule:
                self._seq_to_rule[key] = r
            if r.module_key:
                self._key_to_rule[r.module_key] = r

    def apply(self, graph: ComputeGraph) -> Tuple[ComputeGraph, FusionResult]:
        if self.mode == "fx":
            return self._apply_fx(graph)
        else:
            return self._apply_module_key(graph)

    def _apply_fx(self, graph: ComputeGraph) -> Tuple[ComputeGraph, FusionResult]:
        original_count = graph.num_nodes
        fusions_applied: List[str] = []

        fused_ids: set = set()
        pending_fusions: List[Tuple[FusionRule, List[int]]] = []

        order = graph.topo_order()

        i = 0
        while i < len(order):
            node_id = order[i]
            if node_id in fused_ids:
                i += 1
                continue

            attrs = graph.node_attrs(node_id)
            kind = attrs.get("attrs", {}).get("kind", "")
            if kind != "op":
                i += 1
                continue

            matched = self._try_match_sequence(graph, order, i, fused_ids)
            if matched is not None:
                rule, run = matched
                pending_fusions.append((rule, run))
                for n in run:
                    fused_ids.add(n)
                names = [graph.node_attrs(n).get("name", str(n)) for n in run]
                fusions_applied.append(
                    f"{rule.rule_name}({len(run)} ops): {' + '.join(names[:3])}"
                    + (f" + ... ({len(names) - 3} more)" if len(names) > 3 else "")
                )
                i += len(run)
            else:
                i += 1

        new_graph, node_map, fused_replacement = self._build_fused_graph(
            graph, pending_fusions, fused_ids)

        fusion_result = FusionResult(
            original_nodes=original_count,
            fused_nodes=new_graph.num_nodes,
            fusions_applied=fusions_applied,
        )
        return new_graph, fusion_result

    def _try_match_sequence(
        self,
        graph: ComputeGraph,
        order: List[int],
        start_idx: int,
        fused_ids: set,
    ) -> Optional[Tuple[FusionRule, List[int]]]:
        for seq_tuple, rule in self._seq_to_rule.items():
            if len(seq_tuple) == 0:
                continue
            op_name = graph.node_attrs(order[start_idx]).get("op_name", "")
            if op_name != seq_tuple[0]:
                continue

            run = []
            idx = start_idx
            for expected_op in seq_tuple:
                if idx >= len(order):
                    break
                nid = order[idx]
                if nid in fused_ids:
                    break
                n_attrs = graph.node_attrs(nid)
                if n_attrs.get("attrs", {}).get("kind") != "op":
                    break
                if n_attrs.get("op_name") != expected_op:
                    break
                run.append(nid)
                idx += 1

            if len(run) == len(seq_tuple):
                return rule, run

        return None

    def _apply_module_key(self, graph: ComputeGraph) -> Tuple[ComputeGraph, FusionResult]:
        original_count = graph.num_nodes
        fusions_applied: List[str] = []

        fused_ids: set = set()
        pending_fusions: List[Tuple[FusionRule, List[int]]] = []

        order = graph.topo_order()

        i = 0
        while i < len(order):
            node_id = order[i]
            if node_id in fused_ids:
                i += 1
                continue

            attrs = graph.node_attrs(node_id).get("attrs", {})
            module_path = attrs.get("module_path", "")
            layer = attrs.get("layer", "")

            module_key = _strip_layer_prefix(module_path) if module_path else ""
            rule = self._key_to_rule.get(module_key)
            if rule is None:
                i += 1
                continue

            run = [node_id]
            j = i + 1
            while j < len(order):
                next_id = order[j]
                if next_id in fused_ids:
                    break
                next_attrs = graph.node_attrs(next_id).get("attrs", {})
                next_path = next_attrs.get("module_path", "")
                next_key = _strip_layer_prefix(next_path) if next_path else ""
                if (next_key == module_key
                        and next_attrs.get("module_path") == module_path
                        and next_attrs.get("layer") == layer):
                    run.append(next_id)
                    j += 1
                else:
                    break

            if len(run) >= 2:
                pending_fusions.append((rule, run))
                for n in run:
                    fused_ids.add(n)
                names = [graph.node_attrs(n).get("name", str(n)) for n in run]
                fusions_applied.append(
                    f"{rule.rule_name}({len(run)} ops): {' + '.join(names[:3])}"
                    + (f" + ... ({len(names) - 3} more)" if len(names) > 3 else "")
                )

            i = j if len(run) >= 2 else i + 1

        new_graph, node_map, fused_replacement = self._build_fused_graph(
            graph, pending_fusions, fused_ids)

        fusion_result = FusionResult(
            original_nodes=original_count,
            fused_nodes=new_graph.num_nodes,
            fusions_applied=fusions_applied,
        )
        return new_graph, fusion_result

    def _build_fused_graph(
        self,
        graph: ComputeGraph,
        pending_fusions: List[Tuple[FusionRule, List[int]]],
        fused_ids: set,
    ) -> Tuple[ComputeGraph, Dict[int, int], Dict[int, int]]:
        new_graph = ComputeGraph(graph.name + "_fused")
        node_map: Dict[int, int] = {}
        fused_replacement: Dict[int, int] = {}

        for rule, match_node_ids in pending_fusions:
            first_attrs = graph.node_attrs(match_node_ids[0])

            fused_name = f"fused_{rule.rule_name}"
            fused_node_id = new_graph.add_node(
                op_name=fused_name,
                name=fused_name,
                attrs={
                    "rule_name": rule.rule_name,
                    "module_key": rule.module_key,
                    "aten_op_sequence": rule.aten_op_sequence,
                    "num_sub_ops": len(match_node_ids),
                    "fusion_level": rule.fusion_level,
                    "module_path": first_attrs.get("attrs", {}).get("module_path", ""),
                    "layer": first_attrs.get("attrs", {}).get("layer", ""),
                    "component": first_attrs.get("attrs", {}).get("component", ""),
                    "input_map": rule.input_map,
                    "parameter_map": rule.parameter_map,
                    "constant_map": rule.constant_map,
                    "output_map": rule.output_map,
                    "_matched_node_ids": match_node_ids,
                },
            )

            for old_id in match_node_ids:
                fused_replacement[old_id] = fused_node_id

        for node_id in graph.topo_order():
            if node_id in fused_ids:
                continue
            old_attrs = graph.node_attrs(node_id)
            new_id = new_graph.add_node(
                op_name=old_attrs["op_name"],
                name=old_attrs["name"],
                attrs=old_attrs.get("attrs", {}),
            )
            node_map[node_id] = new_id

        for old_id, new_id in fused_replacement.items():
            node_map[old_id] = new_id

        seen_edges: set = set()
        for src_id in graph.topo_order():
            src_new = node_map.get(src_id)
            if src_new is None:
                continue
            for dst_id in graph.successors(src_id):
                dst_new = node_map.get(dst_id)
                if dst_new is None:
                    continue
                if src_new == dst_new:
                    continue
                edge_key = (src_new, dst_new)
                if edge_key not in seen_edges:
                    new_graph.add_edge(src_new, dst_new)
                    seen_edges.add(edge_key)

        return new_graph, node_map, fused_replacement


def export_fusion_rules_json(
    rules: List[FusionRule],
    output_path: Path,
) -> Path:
    json_path = output_path.with_name(output_path.stem + "_fusion_rules.json")
    json_data = [r.to_dict() for r in rules]
    json_path.write_text(json.dumps(json_data, indent=2))
    return json_path


def load_fusion_rules_json(path: str | Path) -> List[FusionRule]:
    return FusionRule.from_json(path)
