"""Retemplate: rebind variable shape dimensions in a captured OpGraph.

When a graph is captured with ``seq_len=4096, batch_size=1``, tensor
shapes are "frozen" to those values.  ``retemplate()`` substitutes
symbolic tags (S/Q/B/BQ/BS) with new runtime dimensions so the same
captured graph can be replayed at a different batch size or sequence
length.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .types import memory_bytes

if TYPE_CHECKING:
    from .graph import OpGraph


_TAG_MAP = {
    "S":  "seq_len",
    "Q":  "query_len",
    "B":  "batch_size",
    "BS": "batch_seq",
    "BQ": "batch_query",
}


def retemplate(
    graph: "OpGraph",
    batch_size: int,
    seq_len: int,
    query_len: int | None = None,
) -> "OpGraph":
    """Return a cloned graph with shape_template tags expanded to values.

    Parameters
    ----------
    graph : OpGraph
        Captured graph whose tensors may carry ``shape_template``.
    batch_size : int
        New micro-batch size per GPU.
    seq_len : int
        New sequence length (prefill window or total context).
    query_len : int or None
        New query length (decode = 1, prefill = seq_len).
        Defaults to *seq_len* when not set.

    Returns
    -------
    OpGraph
        Cloned graph with ``.shape`` and ``.mem_bytes`` updated on every
        tensor that had a non-None ``shape_template``.
    """
    if query_len is None:
        query_len = seq_len

    bindings = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "query_len": query_len,
        "batch_seq": batch_size * seq_len,
        "batch_query": batch_size * query_len,
    }

    g = graph.clone()

    for node in g.nodes.values():
        for tlist in (node.inputs, node.outputs):
            for i, tmeta in enumerate(tlist):
                if tmeta.shape_template is None:
                    continue
                new_shape = tuple(
                    _resolve(dim, bindings) for dim in tmeta.shape_template
                )
                tlist[i] = tmeta.with_shape(new_shape)

    return g


def _resolve(dim: int | str, bindings: dict[str, int]) -> int:
    """Resolve a single dimension tag or pass through an int."""
    if isinstance(dim, int):
        return dim
    tag = _TAG_MAP.get(dim)
    if tag is not None:
        return bindings[tag]
    # Unknown string tag — preserve as-is (should not normally happen)
    return bindings.get(dim, 0)
