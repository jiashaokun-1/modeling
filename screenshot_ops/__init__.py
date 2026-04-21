"""Capture the operator sequence of any HuggingFace causal LM via
dispatch-level tracing and write the results to an Excel file.

Public API::

    from screenshot_ops import run_trace, load_model, build_config_summary
"""
from screenshot_ops.main import main, run_trace, build_config_summary
from screenshot_ops.model_loader import load_model

__all__ = [
    "main",
    "run_trace",
    "build_config_summary",
    "load_model",
]
