"""Track which nn.Module is currently executing via forward hooks."""
from __future__ import annotations

from typing import Any, Dict, List

import torch.nn as nn


class ModuleTracker:
    """Track which nn.Module is currently executing via forward hooks.

    Also records module metadata (class name, parent-child relationships)
    for automatic fusion rule discovery.
    """

    def __init__(self, root: nn.Module):
        self._stack: List[str] = []
        self._class_stack: List[str] = []
        self._handles: List[Any] = []
        self.path_to_class: Dict[str, str] = {}
        self.path_to_children: Dict[str, List[str]] = {}
        self._install(root, "")

    def _install(self, module: nn.Module, prefix: str):
        child_paths = []
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            class_name = type(child).__name__
            self.path_to_class[full_name] = class_name
            child_paths.append(full_name)

            def _pre_hook(m, inp, _fn=full_name, _cls=class_name):
                self._stack.append(_fn)
                self._class_stack.append(_cls)

            def _post_hook(m, inp, out, _fn=full_name):
                if self._stack and self._stack[-1] == _fn:
                    self._stack.pop()
                    if self._class_stack:
                        self._class_stack.pop()

            h1 = child.register_forward_pre_hook(_pre_hook)
            h2 = child.register_forward_hook(_post_hook)
            self._handles.extend([h1, h2])
            self._install(child, full_name)
        self.path_to_children[prefix] = child_paths

    @property
    def current_module(self) -> str:
        return self._stack[-1] if self._stack else ""

    @property
    def current_module_class(self) -> str:
        return self._class_stack[-1] if self._class_stack else ""

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
