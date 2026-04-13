"""TorchDispatchMode-based recorder that intercepts every aten op."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from screenshot_ops.tensor_utils import (
    SKIP_OPS,
    collect_tensors,
    collect_output_tensors,
    shape_str,
)
from screenshot_ops.tracker import ModuleTracker
from screenshot_ops.classifier import extract_layer_idx, classify_component


class TensorTracker:
    """Assign stable unique IDs to tensors seen during tracing.

    id(tensor) is not reliable on meta device, so we maintain our own counter.
    """

    def __init__(self):
        self._counter = 0
        self._id_map: Dict[int, int] = {}

    def reset(self):
        self._counter = 0
        self._id_map.clear()

    def get_id(self, t: torch.Tensor) -> int:
        oid = id(t)
        if oid not in self._id_map:
            self._id_map[oid] = self._counter
            self._counter += 1
        return self._id_map[oid]


class RecordingDispatch(TorchDispatchMode):
    """Intercept every aten op and record its metadata."""

    def __init__(self, tensor_tracker: TensorTracker,
                 module_tracker: Optional[ModuleTracker] = None,
                 skip_reshapes: bool = True):
        super().__init__()
        self.tensor_tracker = tensor_tracker
        self.records: List[Dict[str, Any]] = []
        self._module_tracker = module_tracker
        self._skip_reshapes = skip_reshapes

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)

        func_name = str(func.overloadpacket) + "." + func._overloadname

        input_tensors = collect_tensors(args, kwargs)
        output_tensors = collect_output_tensors(out)

        input_ids = [self.tensor_tracker.get_id(t) for t in input_tensors]
        output_ids = [self.tensor_tracker.get_id(t) for t in output_tensors]

        if self._skip_reshapes and func_name in SKIP_OPS:
            return out

        module_path = ""
        module_class = ""
        if self._module_tracker:
            module_path = self._module_tracker.current_module
            module_class = self._module_tracker.current_module_class

        input_shapes = [shape_str(t) for t in input_tensors]
        input_dtypes = [str(t.dtype) for t in input_tensors]
        output_shapes = [shape_str(t) for t in output_tensors]
        output_dtypes = [str(t.dtype) for t in output_tensors]

        self.records.append({
            "idx": len(self.records),
            "aten_op": func_name,
            "module_path": module_path,
            "module_class": module_class,
            "layer": extract_layer_idx(module_path),
            "component": classify_component(module_path, func_name),
            "input_shapes": ", ".join(input_shapes),
            "input_dtypes": ", ".join(input_dtypes),
            "output_shapes": ", ".join(output_shapes),
            "output_dtypes": ", ".join(output_dtypes),
            "num_inputs": len(input_tensors),
            "num_outputs": len(output_tensors),
            "_input_ids": input_ids,
            "_output_ids": output_ids,
        })

        return out
