"""Trace HuggingFace models using make_fx + FakeTensor + decomposition.

Produces a torch.fx.GraphModule where composite ops (rms_norm, layer_norm,
linear, etc.) are decomposed into basic aten ops.  The resulting FX Graph
has precise node types:

- ``placeholder``  — function input arguments
- ``get_attr``     — model parameters (weight, bias)
- ``call_function`` — aten op invocations
- ``output``       — function return values

This gives us exact provenance for every tensor: whether it came from an
external input, a model parameter, or an intermediate computation.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
from torch._decomp import core_aten_decompositions
from torch.fx.experimental.proxy_tensor import make_fx

logger = logging.getLogger(__name__)


class FXTracer:
    """Trace a model's forward pass using make_fx with decomposition."""

    def trace(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        decomposition_table: Optional[Dict] = None,
        use_decomposition: bool = True,
    ) -> torch.fx.GraphModule:
        """Trace model forward, returning a GraphModule.

        Parameters
        ----------
        model:
            The nn.Module to trace (should be on meta device).
        input_ids, attention_mask, position_ids:
            Example inputs (meta tensors).
        decomposition_table:
            Custom decomposition table.  Defaults to
            ``core_aten_decompositions()`` when *use_decomposition* is True.
        use_decomposition:
            If False, trace without decomposition (composite ops stay intact).
        """
        if use_decomposition and decomposition_table is None:
            decomposition_table = core_aten_decompositions()
        if not use_decomposition:
            decomposition_table = {}

        def forward_fn(input_ids, attention_mask, position_ids):
            return model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )

        logger.info("Tracing with make_fx (decomposition=%s, %d rules) …",
                     use_decomposition,
                     len(decomposition_table) if decomposition_table else 0)

        gm = make_fx(
            forward_fn,
            decomposition_table=decomposition_table,
        )(input_ids, attention_mask, position_ids)

        n_ph = sum(1 for n in gm.graph.nodes if n.op == "placeholder")
        n_ga = sum(1 for n in gm.graph.nodes if n.op == "get_attr")
        n_cf = sum(1 for n in gm.graph.nodes if n.op == "call_function")
        logger.info("FX Graph: %d placeholders, %d get_attr, %d call_function",
                     n_ph, n_ga, n_cf)

        return gm
