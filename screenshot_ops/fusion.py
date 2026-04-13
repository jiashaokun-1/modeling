"""Two-pass automatic operator fusion using module hierarchy."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from screenshot_ops.tracker import ModuleTracker

MAX_PARENT_FUSE_OPS = 30


@dataclass
class FusionSpec:
    """A fusion pattern auto-discovered from dispatch tracing."""
    module_class: str
    aten_op_sequence: List[str]
    num_sub_ops: int
    fusion_level: str
    example_module_path: str
    occurrences: int = 1
    fused_input_shapes: str = ""
    fused_input_dtypes: str = ""
    fused_input_sources: str = ""
    fused_output_shapes: str = ""
    fused_output_dtypes: str = ""
    fused_output_sources: str = ""


def _split_shape_list(s: str) -> List[str]:
    """Split '[1, 128], [64]' into ['[1, 128]', '[64]']."""
    if not s:
        return []
    result = []
    depth = 0
    current = []
    for ch in s:
        if ch == "[":
            depth += 1
            current.append(ch)
        elif ch == "]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            result.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        result.append("".join(current).strip())
    return result


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


def _parent_path(module_path: str) -> str:
    if "." in module_path:
        return module_path.rsplit(".", 1)[0]
    return ""


def _compute_fused_io(ops: List[Dict[str, Any]]) -> Dict[str, Any]:
    produced_ids: set = set()
    for op in ops:
        for tid in op.get("_output_ids", []):
            produced_ids.add(tid)

    consumed_ids: set = set()
    for op in ops:
        for tid in op.get("_input_ids", []):
            consumed_ids.add(tid)

    external_input_ids = consumed_ids - produced_ids
    external_output_ids = produced_ids - consumed_ids

    seen_in: set = set()
    fused_input_ids = []
    for op in ops:
        for tid in op.get("_input_ids", []):
            if tid in external_input_ids and tid not in seen_in:
                seen_in.add(tid)
                fused_input_ids.append(tid)

    seen_out: set = set()
    fused_output_ids = []
    for op in reversed(ops):
        for tid in op.get("_output_ids", []):
            if tid in external_output_ids and tid not in seen_out:
                seen_out.add(tid)
                fused_output_ids.insert(0, tid)

    input_shapes = []
    input_dtypes = []
    input_sources = []
    for tid in fused_input_ids:
        for op in ops:
            if tid in op.get("_input_ids", []):
                idx = op["_input_ids"].index(tid)
                shapes = _split_shape_list(op["input_shapes"])
                dtypes = op["input_dtypes"].split(", ")
                input_shapes.append(shapes[idx] if idx < len(shapes) else "?")
                input_dtypes.append(dtypes[idx] if idx < len(dtypes) else "?")
                input_sources.append(f"{op['aten_op']} (in[{idx}])")
                break

    output_shapes = []
    output_dtypes = []
    output_sources = []
    for tid in fused_output_ids:
        for op in reversed(ops):
            if tid in op.get("_output_ids", []):
                idx = op["_output_ids"].index(tid)
                shapes = _split_shape_list(op["output_shapes"])
                dtypes = op["output_dtypes"].split(", ")
                output_shapes.append(shapes[idx] if idx < len(shapes) else "?")
                output_dtypes.append(dtypes[idx] if idx < len(dtypes) else "?")
                output_sources.append(f"{op['aten_op']} (out[{idx}])")
                break

    return {
        "fused_input_shapes": ", ".join(input_shapes),
        "fused_input_dtypes": ", ".join(input_dtypes),
        "fused_input_sources": " | ".join(input_sources),
        "fused_output_shapes": ", ".join(output_shapes),
        "fused_output_dtypes": ", ".join(output_dtypes),
        "fused_output_sources": " | ".join(output_sources),
    }


def _make_fused_entry(ops: List[Dict[str, Any]], tracker: ModuleTracker,
                      fusion_level: str = "leaf") -> Dict[str, Any]:
    first, last = ops[0], ops[-1]
    path = first["module_path"]
    module_class = tracker.path_to_class.get(path, first.get("module_class", ""))

    aten_ops = list(dict.fromkeys(r["aten_op"] for r in ops))

    short_path = _strip_layer_prefix(path) if path else ""
    if module_class and len(ops) > 1:
        label = f"{short_path} ({module_class})"
    elif short_path:
        label = short_path
    else:
        fn_parts = first["aten_op"].split(".")
        label = fn_parts[1] if len(fn_parts) >= 2 else first["aten_op"]

    io = _compute_fused_io(ops)

    return {
        "fused_op": label,
        "module_path": path,
        "module_class": module_class,
        "fusion_level": fusion_level,
        "aten_ops": " \u2192 ".join(aten_ops),
        "num_sub_ops": len(ops),
        "layer": first["layer"],
        "input_shapes": first["input_shapes"],
        "input_dtypes": first["input_dtypes"],
        "output_shapes": last["output_shapes"],
        "output_dtypes": last["output_dtypes"],
        **io,
        "_children": ops,
    }


class FusionEngine:
    """Two-pass automatic fusion using module hierarchy."""

    def __init__(self, tracker: ModuleTracker):
        self._tracker = tracker

    def fuse(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups = self._pass1_leaf(records)
        groups = self._pass2_parent(groups)
        for i, g in enumerate(groups):
            g["idx"] = i
            g.pop("_children", None)
        return groups

    def extract_specs(self, fused: List[Dict[str, Any]]) -> List[FusionSpec]:
        specs_by_key: Dict[Tuple[str, str], FusionSpec] = {}
        for g in fused:
            if g["num_sub_ops"] <= 1:
                continue
            key = (g["module_class"], g["fusion_level"])
            if key in specs_by_key:
                specs_by_key[key].occurrences += 1
            else:
                specs_by_key[key] = FusionSpec(
                    module_class=g["module_class"],
                    aten_op_sequence=g["aten_ops"].split(" \u2192 "),
                    num_sub_ops=g["num_sub_ops"],
                    fusion_level=g["fusion_level"],
                    example_module_path=g["module_path"],
                    occurrences=1,
                    fused_input_shapes=g.get("fused_input_shapes", ""),
                    fused_input_dtypes=g.get("fused_input_dtypes", ""),
                    fused_input_sources=g.get("fused_input_sources", ""),
                    fused_output_shapes=g.get("fused_output_shapes", ""),
                    fused_output_dtypes=g.get("fused_output_dtypes", ""),
                    fused_output_sources=g.get("fused_output_sources", ""),
                )
        return sorted(specs_by_key.values(), key=lambda s: -s.occurrences)

    def _pass1_leaf(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not records:
            return []
        groups = []
        current_group = [records[0]]
        for rec in records[1:]:
            same_path = rec["module_path"] == current_group[0]["module_path"]
            same_layer = rec["layer"] == current_group[0]["layer"]
            has_path = rec["module_path"] != ""
            if same_path and same_layer and has_path:
                current_group.append(rec)
            else:
                groups.append(_make_fused_entry(current_group, self._tracker, "leaf"))
                current_group = [rec]
        groups.append(_make_fused_entry(current_group, self._tracker, "leaf"))
        return groups

    def _pass2_parent(self, groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not groups:
            return []

        parent_child_count: Dict[str, set] = defaultdict(set)
        parent_total_ops: Dict[str, int] = defaultdict(int)
        for g in groups:
            p = _parent_path(g["module_path"])
            if p:
                parent_child_count[p].add(g["module_path"])
                parent_total_ops[p] += g["num_sub_ops"]

        def _is_fusible_parent(parent: str) -> bool:
            if parent not in self._tracker.path_to_class:
                return False
            children = self._tracker.path_to_children.get(parent, [])
            if not children:
                return False
            if len(parent_child_count.get(parent, set())) > 5:
                return False
            if parent_total_ops.get(parent, 0) > MAX_PARENT_FUSE_OPS:
                return False
            return True

        result = []
        i = 0
        while i < len(groups):
            parent = _parent_path(groups[i]["module_path"])

            if parent and _is_fusible_parent(parent):
                j = i + 1
                total_ops = groups[i]["num_sub_ops"]
                while j < len(groups):
                    g = groups[j]
                    g_parent = _parent_path(g["module_path"])
                    if ((g_parent == parent or g["module_path"] == parent)
                            and g["layer"] == groups[i]["layer"]):
                        total_ops += g["num_sub_ops"]
                        if total_ops > MAX_PARENT_FUSE_OPS:
                            break
                        j += 1
                    else:
                        break

                if j > i + 1:
                    merged_ops = []
                    for g in groups[i:j]:
                        merged_ops.extend(g["_children"])
                    parent_class = self._tracker.path_to_class[parent]
                    short = _strip_layer_prefix(parent)
                    aten_ops = list(dict.fromkeys(r["aten_op"] for r in merged_ops))
                    io = _compute_fused_io(merged_ops)
                    result.append({
                        "fused_op": f"{short} ({parent_class})",
                        "module_path": parent,
                        "module_class": parent_class,
                        "fusion_level": "parent",
                        "aten_ops": " \u2192 ".join(aten_ops),
                        "num_sub_ops": len(merged_ops),
                        "layer": groups[i]["layer"],
                        "input_shapes": merged_ops[0]["input_shapes"],
                        "input_dtypes": merged_ops[0]["input_dtypes"],
                        "output_shapes": merged_ops[-1]["output_shapes"],
                        "output_dtypes": merged_ops[-1]["output_dtypes"],
                        **io,
                        "_children": merged_ops,
                    })
                    i = j
                    continue

            result.append(groups[i])
            i += 1

        return result
