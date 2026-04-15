"""Graph-based operator fusion driven by JSON fusion rules.

Supports two fusion strategies:

1. **De-fusion (FX-based)**: Given a decomposed FX Graph (from make_fx +
   core_aten_decompositions), identify sub-sequences of basic ops that
   correspond to known composite ops (e.g. pow+mean+add+rsqrt+mul+mul →
   RMSNorm) and merge them back.  This leverages FX Graph's precise
   node provenance (placeholder = input, get_attr = parameter) to
   produce accurate input_map / parameter_map / output_map.

2. **Module-class fusion (legacy)**: Group ops by module_class from
   TorchDispatchMode tracing.  Kept as fallback for models incompatible
   with make_fx.

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
    module_class: str
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
                rule_name=entry.get("rule_name", entry.get("module_class", "")),
                module_class=entry["module_class"],
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
                rule_name=s.module_class,
                module_class=s.module_class,
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
            "module_class": self.module_class,
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


class FusionPass:
    """Apply fusion rules to a ComputeGraph, producing a new fused graph.

    Supports two modes controlled by ``mode``:
      - ``"fx"``: De-fusion on FX-derived graphs.  Matches op sequences
        by walking the DAG in topological order and checking if a run of
        nodes matches a rule's ``aten_op_sequence``.
      - ``"module_class"``: Legacy module-class grouping (for
        TorchDispatchMode-derived graphs).

    In both modes, the original graph is not modified.
    """

    def __init__(self, rules: List[FusionRule], mode: str = "fx"):
        self.rules = rules
        self.mode = mode
        self._seq_to_rule: Dict[Tuple[str, ...], FusionRule] = {}
        self._class_to_rule: Dict[str, FusionRule] = {}
        for r in rules:
            key = tuple(r.aten_op_sequence)
            if key not in self._seq_to_rule:
                self._seq_to_rule[key] = r
            if r.module_class:
                self._class_to_rule[r.module_class] = r

    def apply(self, graph: ComputeGraph) -> Tuple[ComputeGraph, FusionResult]:
        if self.mode == "fx":
            return self._apply_fx(graph)
        else:
            return self._apply_module_class(graph)

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

            op_name = attrs.get("op_name", "")
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

        new_graph = ComputeGraph(graph.name + "_fused")
        node_map: Dict[int, int] = {}
        fused_replacement: Dict[int, int] = {}

        for rule, match_node_ids in pending_fusions:
            first_attrs = graph.node_attrs(match_node_ids[0])
            last_attrs = graph.node_attrs(match_node_ids[-1])

            fused_name = f"fused_{rule.rule_name}"
            fused_node_id = new_graph.add_node(
                op_name=fused_name,
                name=fused_name,
                attrs={
                    "rule_name": rule.rule_name,
                    "aten_op_sequence": rule.aten_op_sequence,
                    "num_sub_ops": len(match_node_ids),
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
        """Try to match any rule's aten_op_sequence starting at order[start_idx]."""
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

    def _apply_module_class(self, graph: ComputeGraph) -> Tuple[ComputeGraph, FusionResult]:
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
            module_class = attrs.get("module_class", "")
            module_path = attrs.get("module_path", "")
            layer = attrs.get("layer", "")

            rule = self._class_to_rule.get(module_class)
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
                if (next_attrs.get("module_class") == module_class
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

        new_graph = ComputeGraph(graph.name + "_fused")
        node_map: Dict[int, int] = {}
        fused_replacement: Dict[int, int] = {}

        for rule, match_node_ids in pending_fusions:
            first_attrs = graph.node_attrs(match_node_ids[0])
            last_attrs = graph.node_attrs(match_node_ids[-1])

            fused_name = f"fused_{rule.rule_name}"
            fused_node_id = new_graph.add_node(
                op_name=fused_name,
                name=fused_name,
                attrs={
                    "rule_name": rule.rule_name,
                    "module_class": rule.module_class,
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

        fusion_result = FusionResult(
            original_nodes=original_count,
            fused_nodes=new_graph.num_nodes,
            fusions_applied=fusions_applied,
        )
        return new_graph, fusion_result


class FusionRuleDiscoverer:
    """Auto-discover fusion rules from a ComputeGraph built via FXGraphAdapter.

    Walks the graph in topological order.  When a run of consecutive ``op``
    nodes has all their non-op predecessors (placeholder / get_attr) in
    common, the run is a candidate for fusion.  The discoverer groups
    identical op sequences and records precise input_map / parameter_map /
    output_map using FXGraphAdapter.extract_io_map().
    """

    def __init__(self, graph: ComputeGraph, adapter=None):
        self._graph = graph
        self._adapter = adapter

    def discover(self) -> List[FusionRule]:
        order = self._graph.topo_order()

        visited: set = set()
        runs: List[List[int]] = []

        i = 0
        while i < len(order):
            nid = order[i]
            if nid in visited:
                i += 1
                continue
            attrs = self._graph.node_attrs(nid)
            kind = attrs.get("attrs", {}).get("kind", "")
            if kind != "op":
                i += 1
                continue

            run = [nid]
            visited.add(nid)
            j = i + 1
            while j < len(order):
                next_id = order[j]
                if next_id in visited:
                    break
                next_attrs = self._graph.node_attrs(next_id)
                next_kind = next_attrs.get("attrs", {}).get("kind", "")
                if next_kind != "op":
                    break
                run.append(next_id)
                visited.add(next_id)
                j += 1

            if len(run) >= 2:
                runs.append(run)
            i = j

        seq_groups: Dict[Tuple[str, ...], List[List[int]]] = {}
        for run in runs:
            seq = tuple(self._graph.node_attrs(n).get("op_name", "") for n in run)
            seq_groups.setdefault(seq, []).append(run)

        rules: List[FusionRule] = []
        for seq, run_list in seq_groups.items():
            first_run = run_list[0]
            rule_name = self._infer_rule_name(seq)

            io_map = {}
            if self._adapter is not None:
                io_map = self._adapter.extract_io_map(
                    first_run, self._graph)

            rule = FusionRule(
                rule_name=rule_name,
                module_class=rule_name,
                aten_op_sequence=list(seq),
                num_sub_ops=len(seq),
                fusion_level="leaf",
                occurrences=len(run_list),
                input_map=io_map.get("input_map", []),
                parameter_map=io_map.get("parameter_map", []),
                constant_map=io_map.get("constant_map", []),
                output_map=io_map.get("output_map", []),
            )
            rules.append(rule)

        return sorted(rules, key=lambda r: -r.occurrences)

    def _infer_rule_name(self, seq: Tuple[str, ...]) -> str:
        op_shorts = []
        for op in seq:
            parts = op.split(".")
            short = parts[1] if len(parts) >= 2 else op
            op_shorts.append(short)

        if op_shorts[:2] == ["pow", "mean"]:
            if "rsqrt" in op_shorts:
                return "RMSNorm"
            if "sqrt" in op_shorts:
                return "LayerNorm"

        if "bmm" in op_shorts and "softmax" in op_shorts:
            return "Attention"

        if "mm" in op_shorts and "silu" in op_shorts:
            return "MLP_SiLU"

        if "mm" in op_shorts and "gelu" in op_shorts:
            return "MLP_GELU"

        if "topk" in op_shorts:
            return "MoEGate"

        if "cos" in op_shorts and "sin" in op_shorts:
            return "RoPE"

        return "_".join(op_shorts[:3])


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
