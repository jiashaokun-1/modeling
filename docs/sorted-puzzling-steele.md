# Training Graph Unification — Revised Plan

## Context

The project has two parallel estimation paths that both converge on the same `PipelineComposer` classes:

- **Stack A (spec-based)**: `ModelSpec → build_graph() → Graph → stage_time() → PipelineComposer → Report`
- **Stack B (graph-capture)**: `OpGraph(fwd+bwd) → stitch_fwd_bwd() → TransformPipeline → DAGScheduler → PipelineComposer → TrainingReport`

The original unification plan (`docs/training_graph_unification_plan.md`) proposes making `OpGraph` the single IR by adding `OpGraph.from_model_spec()` and migrating `flops.py`/`comm.py`/`memory.py` to accept `OpGraph`. However, `docs/training_modeller_zh.md` explicitly states that Stack A is **not the primary path** — Stack B (trace-based) is primary.

**Three problems with the original plan:**

1. **Phase 2 inverts the dependency**: Making `TrainingGraph` wrap `OpGraph` forces a synthetic node-level representation for what is currently handled cleanly by layer-level formulas — this adds complexity without benefit since Stack A doesn't need graph traversal.
2. **Already-done work not acknowledged**: `stitch_fwd_bwd()` already exists in `ir/adapter.py` — Phase 0 of the design doc is implemented.
3. **Output type mismatch ignored**: Stack A returns `Report`; Stack B returns `TrainingReport` — callers cannot use both paths interchangeably today.

**Goal**: Unify the paths by establishing a clean contract — both paths produce `TrainingReport`, Stack A becomes a lightweight spec-only fallback when traces are unavailable, and `OpGraph.from_model_spec()` is added as a synthetic factory for the fast-estimate path.

---

## What Already Exists (Do Not Re-Implement)

| Component | File | Status |
|---|---|---|
| `stitch_fwd_bwd()` | `python/zrt/ir/adapter.py` | ✅ Done |
| `PipelineComposer` + 5 concrete composers | `python/zrt/training/compose/schedules.py` | ✅ Shared by both paths |
| `TrainingPipelinePass` bridging | `python/zrt/transform/analysis/training.py` | ✅ Done |
| `estimate_training_from_graphs()` | `python/zrt/transform/analysis/modeller.py` | ✅ Primary path |

---

## Phase 1: Add `OpGraph.from_model_spec()` Factory (Non-Breaking)

**File**: `python/zrt/ir/graph.py`

Add a `@classmethod` that converts `ModelSpec + Strategy → OpGraph` via the existing Stack A IR:

```python
@classmethod
def from_model_spec(cls, model: ModelSpec, strategy: Strategy, phase: str = "training") -> "OpGraph":
    from zrt.training.ir.builders import build_graph
    training_g = build_graph(model, strategy)
    
    nodes = {}
    for op in training_g.ops:
        nodes[op.name] = OpNode(
            id=op.name,
            op_type=op.kind,
            annotations={"layer_id": op.layer_id, "layer_kind": op.layer_kind},
            meta={**op.meta},
        )
    
    edges = []
    op_names = list(nodes.keys())
    for i in range(len(op_names) - 1):
        curr, nxt = op_names[i], op_names[i + 1]
        if nodes[curr].annotations.get("layer_id") == nodes[nxt].annotations.get("layer_id"):
            edges.append(Edge(src=curr, dst=nxt))
    
    return cls(
        name=f"{model.name}_{phase}",
        nodes=nodes,
        edges=edges,
        metadata={
            "source": "model_spec",
            "model": model.name,
            "strategy": strategy,
            "collectives": {c.name: c for c in training_g.collectives},
        }
    )
```

**Note**: Op fields are `name, kind, inputs, outputs, meta, layer_id, layer_kind` — the code sketch above uses correct field names.

**Tests**: `tests/training/test_opgraph_from_spec.py`
- Verify `len(opgraph.nodes) == len(training_g.ops)` 
- Verify op types match
- Verify `metadata["source"] == "model_spec"`

---

## Phase 2: Unify Output Types

**Problem**: Stack A's `estimate()` returns `Report` (`python/zrt/training/search/estimator.py:21`); Stack B returns `TrainingReport` (`python/zrt/transform/analysis/modeller.py:23`). Callers cannot use both paths interchangeably.

**Option A (preferred)**: Move `TrainingReport` to `python/zrt/training/spec/report.py` (shared location), import it from both Stack A and Stack B. Stack A's `estimate()` returns `TrainingReport` with the subset of fields it can populate.

**Option B (lighter)**: Add a `to_training_report()` method on `Report` that returns a `TrainingReport` with equivalent fields — a pure data conversion.

**Files to change**:
- `python/zrt/training/search/estimator.py` — return `TrainingReport`
- `python/zrt/transform/analysis/modeller.py` — update import if `TrainingReport` moves

---

## Phase 3: Remove Stack A Type Leakage from Stack B

**Problem**: `TrainingPipelinePass` (`python/zrt/transform/analysis/training.py`) imports:
```python
from python.zrt.training.compose.stage import StageTime as _StageTime
from python.zrt.training.compose.schedules import PP_SCHED_BY_NAME, COMPOSER_BY_SCHED
from python.zrt.training.spec.strategy import Strategy as _Strategy, OptKind
```

The `COMPOSER_BY_SCHED` import is fine (shared composters). The `_StageTime` and `_Strategy` imports are the leakage points.

**Fix**: 
- `_StageTime` is already in `zrt.training.compose.stage` — this import is correct, no change needed
- `_Strategy`: `TrainingPipelinePass` uses it to extract `dp_overlap`, `pp_schedule`, etc. from `ctx.training`. Since `ctx.training` is already typed as `Strategy`, the import is necessary but should be a direct import (not aliased with `_`)
- No structural change required here — this cross-import is intentional and represents the bridge between the two stacks; clean up the underscore aliases for readability

---

## Phase 4: Make Stack A Delegate to Stack B (Optional / Future)

When trace graphs are available, Stack A's `estimate()` could call `estimate_training_from_graphs()` instead of its own pipeline. This makes Stack A a pure convenience wrapper.

**Not blocking**: Stack A serves a different use case (fast analytical estimate when traces aren't available or for search/sweep). Keep both entry points. The unification goal is the output type contract (Phase 2), not eliminating Stack A.

---

## What NOT to Do (Deviations from Original Plan)

| Original Plan Item | Why to Skip |
|---|---|
| Phase 2: Make `TrainingGraph` wrap `OpGraph` | Inverts the dependency; adds synthetic node traversal overhead for no gain |
| Migrate `flops.py`/`comm.py`/`memory.py` to accept `OpGraph` | These serve Stack A's layer-level formulas; switching to node-level traversal loses the clean analytical structure |
| Update `PipelineComposer` to accept `graph: OpGraph` | Already shared; adding `graph` param would force graph dependency on spec-path composers |
| Phase 4: Delete `TrainingGraph` | Stack A's `Graph` IR is the right abstraction for spec-based estimation; deleting it destroys the analytical path |

---

## Implementation Order

1. **Phase 1** (add `from_model_spec()`) — 1 PR, ~80 lines, no breaking changes
2. **Phase 2** (unify output types) — 1 PR, ~30 lines + test updates
3. **Phase 3** (cleanup aliases) — minor cleanup, 1 PR with Phase 2

---

## Files Modified

| Phase | File | Change | Est. Lines |
|---|---|---|---|
| 1 | `python/zrt/ir/graph.py` | Add `from_model_spec()` classmethod | +80 |
| 1 | `tests/training/test_opgraph_from_spec.py` | New unit tests | +60 |
| 2 | `python/zrt/training/search/estimator.py` | Return `TrainingReport` | ~10 |
| 2 | `python/zrt/training/spec/report.py` (new or existing) | Shared `TrainingReport` definition | ~30 |
| 2 | `python/zrt/transform/analysis/modeller.py` | Update import path | ~2 |
| 3 | `python/zrt/transform/analysis/training.py` | Clean up `_` aliases | ~5 |

**Total**: ~200 lines across ~6 files

---

## Verification

```bash
# Phase 1: factory test
PYTHONPATH=python pytest tests/training/test_opgraph_from_spec.py -v

# Phase 2: estimator output type compatibility
PYTHONPATH=python pytest tests/training/ -v -k "estimator or report"

# Full regression: all training tests must pass
PYTHONPATH=python pytest tests/training/ -v 2>&1 | tail -n 30

# Anchor regression: MFU must not drift
PYTHONPATH=python pytest tests/training/anchors/test_anchors.py -v
```

---

## Success Criteria

1. `OpGraph.from_model_spec(model, strategy)` produces a valid OpGraph with nodes matching `build_graph(model, strategy).ops`
2. Both `estimator.estimate()` and `estimate_training_from_graphs()` return `TrainingReport`
3. All existing training tests pass (no regression)
4. Stack A and Stack B remain independent execution paths — no forced dependency on the other's runtime
