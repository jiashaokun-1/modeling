"""Tensor helpers and zero-cost op filter list."""
from __future__ import annotations

from typing import Any, List

import torch

SKIP_OPS = {
    "prim.device.default",
    "aten.t.default",
    # "aten.detach.default", "aten.alias.default",
    # "aten.view.default", "aten._unsafe_view.default",
    # "aten.expand.default", "aten.contiguous.default",
    # "aten.slice.Tensor", "aten.select.int",
    # "aten.unsqueeze.default", "aten.squeeze.dim",
    # "aten.split.Tensor", "aten.split_with_sizes.default",
    # "aten.permute.default", "aten.reshape.default",
    # "aten.clone.default",
    # "aten.arange.default", "aten.arange.start",
    # "aten.ones.default", "aten.zeros.default",
    # "aten.full.default", "aten.scalar_tensor.default",
    # "aten.tril.default", "aten.triu.default",
    # "aten.empty_like.default", "aten.zeros_like.default",
    # "aten.index_put_.default", "aten.index_put.default",
    # "aten.scatter_.value", "aten.scatter_.src",
    # "aten.histc.default", "aten.cumsum.default",
    # "aten.bitwise_not.default",
    # "aten.sort.default", "aten.sort.stable",
}


def shape_str(t: torch.Tensor) -> str:
    return str(list(t.shape))


def tag_dims(shape: tuple[int, ...], batch: int, query_len: int, seq_len: int,
             fixed_values: set[int]) -> list[int | str]:
    """Tag each dimension as variable (B/Q/S/BQ/BS) or static (concrete int).

    Variable tags take priority over fixed-value matching.
    Collision case (e.g. hidden == seq_len): variable wins.
    """
    bs = batch * seq_len
    bq = batch * query_len
    tags: list[int | str] = []
    for dim in shape:
        if dim == bq:
            tags.append("BQ")
        elif dim == bs:
            tags.append("BS")
        elif dim == query_len:
            tags.append("Q")
        elif dim == seq_len:
            tags.append("S")
        elif dim == batch:
            tags.append("B")
        else:
            tags.append(dim)
    return tags


def tags_str(tags: list[int | str]) -> str:
    """Serialize tag list to string for record storage, e.g. '[BQ, 7168]'."""
    return "[" + ", ".join(str(t) if isinstance(t, int) else t for t in tags) + "]"


def collect_tensors(args: tuple, kwargs: dict) -> List[torch.Tensor]:
    tensors = []
    for a in args:
        if isinstance(a, torch.Tensor):
            tensors.append(a)
        elif isinstance(a, (list, tuple)):
            for item in a:
                if isinstance(item, torch.Tensor):
                    tensors.append(item)
    for v in kwargs.values():
        if isinstance(v, torch.Tensor):
            tensors.append(v)
        elif isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, torch.Tensor):
                    tensors.append(item)
    return tensors


def collect_output_tensors(out: Any) -> List[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        return [item for item in out if isinstance(item, torch.Tensor)]
    return []
