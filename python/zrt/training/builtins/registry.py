"""Built-in model registry — persist and load captured OpGraphs.

A built-in model is a directory entry under ``builtins/models/``
containing one JSON file per phase (e.g. ``deepseek_v3.prefill.json``)
and an optional ``<model>.meta.yaml`` with capture-time metadata.

Usage::

    from zrt.training.builtins import builtin_registry as br
    models = br.list_models()
    graph, meta = br.load("deepseek_v3", phase="prefill")
    br.save("deepseek_v3", phase="prefill", graph, meta)
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any

from zrt.ir.graph import OpGraph
from zrt.ir.serde import load_json, save_json


_MODELS_DIR = Path(__file__).parent / "models"


class BuiltinRegistry:
    """Registry for pre-captured OpGraph models (the built-in library)."""

    def list_models(self) -> list[str]:
        """Return sorted list of registered model IDs."""
        seen: set[str] = set()
        if _MODELS_DIR.is_dir():
            for p in sorted(_MODELS_DIR.glob("*.json")):
                stem = p.stem  # e.g. "deepseek_v3.prefill"
                if "." in stem:
                    seen.add(stem.rsplit(".", 1)[0])
            for p in sorted(_MODELS_DIR.glob("*.yaml")):
                stem = p.stem
                if stem.endswith(".meta"):
                    seen.add(stem[:-5])  # strip ".meta" suffix
        return sorted(seen)

    def list_phases(self, model_id: str) -> list[str]:
        """Return sorted list of phases captured for *model_id*."""
        phases: set[str] = set()
        prefix = f"{model_id}."
        if _MODELS_DIR.is_dir():
            for p in sorted(_MODELS_DIR.glob(f"{model_id}.*.json")):
                stem = p.stem  # e.g. "deepseek_v3.prefill"
                after = stem[len(prefix):] if stem.startswith(prefix) else ""
                if after:
                    phases.add(after)
        return sorted(phases)

    def load(self, model_id: str, phase: str) -> tuple[OpGraph, dict[str, Any]]:
        """Load a captured OpGraph and its metadata.

        Returns
        -------
        (graph, meta_dict)
            *graph* is the deserialized OpGraph.
            *meta_dict* is the merged metadata (graph.metadata + YAML meta).
        """
        json_path = _MODELS_DIR / f"{model_id}.{phase}.json"
        if not json_path.is_file():
            raise FileNotFoundError(
                f"Built-in model '{model_id}' phase '{phase}' not found at {json_path}"
            )
        graph = load_json(str(json_path))

        # Merge model-level meta if available
        meta: dict[str, Any] = dict(graph.metadata)
        yaml_path = _MODELS_DIR / f"{model_id}.meta.yaml"
        if yaml_path.is_file():
            with open(yaml_path, encoding="utf-8") as f:
                yaml_meta = yaml.safe_load(f) or {}
            meta.update(yaml_meta)

        return graph, meta

    def save(
        self,
        model_id: str,
        phase: str,
        graph: OpGraph,
        meta: dict[str, Any],
    ) -> None:
        """Persist a captured OpGraph and its metadata."""
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)

        json_path = _MODELS_DIR / f"{model_id}.{phase}.json"
        save_json(graph, str(json_path))

        yaml_path = _MODELS_DIR / f"{model_id}.meta.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(_sanitize_meta(meta), f, default_flow_style=False)

    def save_graph(self, model_id: str, phase: str, graph: OpGraph) -> None:
        """Persist only the graph (no metadata update)."""
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        json_path = _MODELS_DIR / f"{model_id}.{phase}.json"
        save_json(graph, str(json_path))

    def save_meta(self, model_id: str, meta: dict[str, Any]) -> None:
        """Persist / update the metadata YAML for a model."""
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        yaml_path = _MODELS_DIR / f"{model_id}.meta.yaml"
        existing: dict[str, Any] = {}
        if yaml_path.is_file():
            with open(yaml_path, encoding="utf-8") as f:
                existing = yaml.safe_load(f) or {}
        existing.update(meta)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(_sanitize_meta(existing), f, default_flow_style=False)


# ── Singleton ──────────────────────────────────────────────────────────────────

builtin_registry = BuiltinRegistry()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sanitize_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Remove unserializable values from meta dict."""
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool, list, dict, type(None))):
            out[k] = v
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = str(v)
    return out
