"""Tensor helpers and zero-cost op filter list."""
from __future__ import annotations

from typing import Any, List

import torch

SKIP_OPS = {
    "aten.detach.default", "aten.alias.default", "aten.t.default",
    "aten.view.default", "aten._unsafe_view.default",
    "aten.expand.default", "aten.contiguous.default",
    "aten.slice.Tensor", "aten.select.int",
    "aten.unsqueeze.default", "aten.squeeze.dim",
    "aten.split.Tensor", "aten.split_with_sizes.default",
    "aten.permute.default", "aten.reshape.default",
    "aten.clone.default",
    "aten.arange.default", "aten.arange.start",
    "aten.ones.default", "aten.zeros.default",
    "aten.full.default", "aten.scalar_tensor.default",
    "aten.tril.default", "aten.triu.default",
    "aten.empty_like.default", "aten.zeros_like.default",
    "aten.index_put_.default", "aten.index_put.default",
    "aten.scatter_.value", "aten.scatter_.src",
    "aten.histc.default", "aten.cumsum.default",
    "aten.bitwise_not.default",
    "aten.sort.default", "aten.sort.stable",
}


def shape_str(t: torch.Tensor) -> str:
    return str(list(t.shape))


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
