"""JSON serialization / deserialization for OpGraph IR types.

Round-trip guarantee: ``from_dict(to_dict(graph)) == graph`` (structurally).

Schema
------
OpGraph JSON::

    {
        "name":     str,
        "phase":    str,
        "metadata": {...},
        "nodes": {
            "<node_id>": {
                "id":          str,
                "op_type":     str,
                "inputs":      [TensorMeta, ...],
                "outputs":     [TensorMeta, ...],
                "attrs":       {...},
                "scope":       str,
                "category":    str,
                "annotations": {...},
                "op_short":       str,
                "module_class":   str,
                "layer":          str,
                "component":      str,
                "fused_from":     [str, ...],
                "num_sub_ops":    int,
                "fusion_level":   str,
                "src_file": str, "src_line": int, "src_code": str
            },
            ...
        },
        "edges": [
            {
                "src": str, "src_idx": int,
                "dst": str, "dst_idx": int,
                "tensor": TensorMeta | null,
                "tensor_id": int | null
            },
            ...
        ]
    }

TensorMeta JSON::

    {"id": str, "shape": [int, ...], "dtype": str, "mem_bytes": int}
"""
from __future__ import annotations

import json
from typing import Any

from .edge import Edge
from .graph import OpGraph
from .node import OpNode
from .types import DType, TensorMeta


# ─────────────────────────────────────────────────────────────────────────────
# TensorMeta
# ─────────────────────────────────────────────────────────────────────────────

def tensor_meta_to_dict(t: TensorMeta) -> dict[str, Any]:
    return {
        "id":        t.id,
        "shape":     list(t.shape),
        "dtype":     t.dtype.value,
        "mem_bytes": t.mem_bytes,
    }


def tensor_meta_from_dict(d: dict[str, Any]) -> TensorMeta:
    return TensorMeta(
        id=d["id"],
        shape=tuple(d["shape"]),
        dtype=DType(d["dtype"]),
        mem_bytes=d["mem_bytes"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# OpNode
# ─────────────────────────────────────────────────────────────────────────────

def op_node_to_dict(n: OpNode) -> dict[str, Any]:
    return {
        "id":           n.id,
        "op_type":      n.op_type,
        "inputs":       [tensor_meta_to_dict(t) for t in n.inputs],
        "outputs":      [tensor_meta_to_dict(t) for t in n.outputs],
        "attrs":        n.attrs,
        "scope":        n.scope,
        "category":     n.category,
        "annotations":  n.annotations,
        "op_short":     n.op_short,
        "module_class": n.module_class,
        "layer":        n.layer,
        "component":    n.component,
        "fused_from":   n.fused_from,
        "num_sub_ops":  n.num_sub_ops,
        "fusion_level": n.fusion_level,
        "src_file":     n.src_file,
        "src_line":     n.src_line,
        "src_code":     n.src_code,
    }


def op_node_from_dict(d: dict[str, Any]) -> OpNode:
    return OpNode(
        id=d["id"],
        op_type=d["op_type"],
        inputs=[tensor_meta_from_dict(t) for t in d.get("inputs", [])],
        outputs=[tensor_meta_from_dict(t) for t in d.get("outputs", [])],
        attrs=d.get("attrs", {}),
        scope=d.get("scope", ""),
        category=d.get("category", "compute"),
        annotations=d.get("annotations", {}),
        op_short=d.get("op_short", ""),
        module_class=d.get("module_class", ""),
        layer=d.get("layer", ""),
        component=d.get("component", ""),
        fused_from=d.get("fused_from", []),
        num_sub_ops=d.get("num_sub_ops", 0),
        fusion_level=d.get("fusion_level", ""),
        src_file=d.get("src_file", ""),
        src_line=d.get("src_line", 0),
        src_code=d.get("src_code", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Edge
# ─────────────────────────────────────────────────────────────────────────────

def edge_to_dict(e: Edge) -> dict[str, Any]:
    return {
        "src":       e.src,
        "src_idx":   e.src_idx,
        "dst":       e.dst,
        "dst_idx":   e.dst_idx,
        "tensor":    tensor_meta_to_dict(e.tensor) if e.tensor else None,
        "tensor_id": e.tensor_id,
    }


def edge_from_dict(d: dict[str, Any]) -> Edge:
    t_raw = d.get("tensor")
    return Edge(
        src=d["src"],
        src_idx=d.get("src_idx", 0),
        dst=d["dst"],
        dst_idx=d.get("dst_idx", 0),
        tensor=tensor_meta_from_dict(t_raw) if t_raw else None,
        tensor_id=d.get("tensor_id"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# OpGraph
# ─────────────────────────────────────────────────────────────────────────────

def opgraph_to_dict(g: OpGraph) -> dict[str, Any]:
    return {
        "name":     g.name,
        "phase":    g.phase,
        "metadata": g.metadata,
        "nodes":    {nid: op_node_to_dict(n) for nid, n in g.nodes.items()},
        "edges":    [edge_to_dict(e) for e in g.edges],
    }


def opgraph_from_dict(d: dict[str, Any]) -> OpGraph:
    nodes = {nid: op_node_from_dict(nd) for nid, nd in d.get("nodes", {}).items()}
    edges = [edge_from_dict(ed) for ed in d.get("edges", [])]
    return OpGraph(
        name=d["name"],
        phase=d["phase"],
        nodes=nodes,
        edges=edges,
        metadata=d.get("metadata", {}),
    )


# ─────────────────────────────────────────────────────────────────────────────
# File I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_json(graph: OpGraph, path: str, indent: int = 2) -> None:
    """Serialize ``graph`` to a JSON file at ``path``."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(opgraph_to_dict(graph), f, ensure_ascii=False, indent=indent)


def load_json(path: str) -> OpGraph:
    """Deserialize an OpGraph from a JSON file at ``path``."""
    with open(path, "r", encoding="utf-8") as f:
        return opgraph_from_dict(json.load(f))
