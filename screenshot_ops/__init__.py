"""Capture the operator sequence of DeepSeek-V3 via dispatch-level tracing
and write the results to an Excel file.

Inspired by xpu_simulator/frontend/dispatch_extractor.py — we use
TorchDispatchMode to intercept every aten op that fires during a forward
pass on meta tensors, then dump the ordered sequence to Excel.
"""
from screenshot_ops.main import main

__all__ = ["main"]
