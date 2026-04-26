"""Integration tests for the built-in model library path (Phase 3-4)."""

import pytest
import tempfile
import json
from pathlib import Path

from zrt.ir.types import DType, TensorMeta
from zrt.ir.graph import OpGraph
from zrt.ir.node import OpNode
from zrt.ir.serde import save_json, load_json
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import GPU, NetTier, SystemSpec


# ── Helpers ─────────────────────────────────────────────────────────────────────


def _make_dummy_opgraph(seq_len=128, batch_size=1):
    """Create a captured OpGraph with tagged shapes for a 1-layer model."""
    g = OpGraph(name="test_prefill", phase="prefill", metadata={
        "seq_len": seq_len, "batch_size": batch_size,
        "hidden": 512, "num_layers": 1,
    })
    g.add_node(OpNode(
        id="op_0", op_type="aten.embedding.default",
        inputs=[TensorMeta.from_shape_dtype("t0", (batch_size, seq_len), DType.INT64,
                                            shape_template=("B", "Q"))],
        outputs=[TensorMeta.from_shape_dtype("t1", (batch_size * seq_len, 512), DType.BF16,
                                             shape_template=("BQ", 512))],
        scope="model.embed_tokens", layer="-1",
    ))
    g.add_node(OpNode(
        id="op_1", op_type="aten.mm.default",
        inputs=[TensorMeta.from_shape_dtype("t1", (batch_size * seq_len, 512), DType.BF16,
                                            shape_template=("BQ", 512))],
        outputs=[TensorMeta.from_shape_dtype("t2", (batch_size * seq_len, 1536), DType.BF16,
                                             shape_template=("BQ", 1536))],
        scope="model.layers.0.self_attn.qkv_proj", layer="0",
    ))
    g.add_node(OpNode(
        id="op_2", op_type="aten.mm.default",
        inputs=[TensorMeta.from_shape_dtype("t2", (batch_size * seq_len, 1536), DType.BF16,
                                            shape_template=("BQ", 1536))],
        outputs=[TensorMeta.from_shape_dtype("t3", (batch_size * seq_len, 512), DType.BF16,
                                             shape_template=("BQ", 512))],
        scope="model.layers.0.self_attn.o_proj", layer="0",
    ))
    g.add_node(OpNode(
        id="op_3", op_type="aten.mm.default",
        inputs=[TensorMeta.from_shape_dtype("t4", (batch_size * seq_len, 512), DType.BF16,
                                            shape_template=("BQ", 512))],
        outputs=[TensorMeta.from_shape_dtype("t5", (batch_size * seq_len, 1024), DType.BF16,
                                             shape_template=("BQ", 1024))],
        scope="model.layers.0.mlp.gate_proj", layer="0",
    ))
    g.add_node(OpNode(
        id="op_4", op_type="aten.mm.default",
        inputs=[TensorMeta.from_shape_dtype("t6", (batch_size * seq_len, 1024), DType.BF16,
                                            shape_template=("BQ", 1024))],
        outputs=[TensorMeta.from_shape_dtype("t7", (batch_size * seq_len, 512), DType.BF16,
                                             shape_template=("BQ", 512))],
        scope="model.layers.0.mlp.down_proj", layer="0",
    ))
    # Add a LN node so that layer 0 aggregation finds something
    g.add_node(OpNode(
        id="op_6", op_type="aten.rms_norm.default",
        inputs=[TensorMeta.from_shape_dtype("t1b", (batch_size * seq_len, 512), DType.BF16,
                                            shape_template=("BQ", 512))],
        outputs=[TensorMeta.from_shape_dtype("t1c", (batch_size * seq_len, 512), DType.BF16,
                                             shape_template=("BQ", 512))],
        scope="model.layers.0.input_layernorm", layer="0",
    ))
    g.add_node(OpNode(
        id="op_5", op_type="aten.mm.default",
        inputs=[TensorMeta.from_shape_dtype("t8", (batch_size * seq_len, 512), DType.BF16,
                                            shape_template=("BQ", 512))],
        outputs=[TensorMeta.from_shape_dtype("t9", (batch_size * seq_len, 1000), DType.BF16,
                                             shape_template=("BQ", 1000))],
        scope="lm_head", layer="-1",
    ))
    return g


def _make_dummy_model(seq_len=128):
    return ModelSpec(
        hidden=512, ffn=1024, num_heads=8, num_kv_heads=8, head_dim=64,
        vocab=1000, seq_len=seq_len, layers=[LayerKind.DENSE],
    )


def _make_dummy_system():
    return SystemSpec(
        gpu=GPU(name="test", flops_bf16=100.0, flops_fp8=200.0,
                hbm_gb=80, hbm_bw_gbps=3000.0),
        nodes=1, gpus_per_node=1, host_mem_gb=256,
        nets=[NetTier(scope="intra_node", bw_gbps=900.0, latency_us=1.0,
                      topology="nvlink")],
    )


# ── Tests ───────────────────────────────────────────────────────────────────────


def test_builtin_registry_save_load():
    """Save a dummy graph via registry, load it back, verify round-trip."""
    from zrt.training.builtins import builtin_registry

    g = _make_dummy_opgraph()
    meta = {"model_id": "test_model", "captured_with": {"seq_len": 128, "batch_size": 1}}

    builtin_registry.save("test_model", "prefill", g, meta)

    g2, meta2 = builtin_registry.load("test_model", "prefill")
    assert len(g2.nodes) == len(g.nodes)
    assert meta2.get("model_id") == "test_model"


def test_builtin_registry_list():
    """list_models includes the saved model."""
    from zrt.training.builtins import builtin_registry

    # Save a model directly in this test
    g = _make_dummy_opgraph()
    meta = {"model_id": "test_model2"}
    builtin_registry.save("test_model2", "prefill", g, meta)

    models = builtin_registry.list_models()
    assert "test_model2" in models, f"Expected test_model2 in {models}"

    phases = builtin_registry.list_phases("test_model2")
    assert "prefill" in phases


def test_retemplate_integration():
    """retemplate on dummy graph with shape templates."""
    from zrt.ir.retemplate import retemplate

    g = _make_dummy_opgraph(seq_len=128, batch_size=1)
    g2 = retemplate(g, batch_size=2, seq_len=256)

    # Check shapes updated from (128, ...) → (512, ...) for BQ dims
    embed_node = g2.nodes["op_0"]
    assert embed_node.outputs[0].shape == (512, 512), \
        f"Expected (512, 512) for embed output, got {embed_node.outputs[0].shape}"

    qkv_node = g2.nodes["op_1"]
    assert qkv_node.inputs[0].shape == (512, 512)


def test_aggregate_integration():
    """aggregate_to_training_ir on a dummy OpGraph produces valid training Graph."""
    from zrt.training.ir.from_opgraph import aggregate_to_training_ir

    g = _make_dummy_opgraph(seq_len=128, batch_size=1)
    model = _make_dummy_model(seq_len=128)
    tgraph = aggregate_to_training_ir(g, model)

    # At minimum we should get some ops
    assert len(tgraph.ops) > 0, "Expected at least some semantic ops"
    # Layer 0 should have ops
    assert 0 in tgraph.layer_index
    # Non-layer ops (embed, lm_head) should be present
    op_names = [op.name for op in tgraph.ops]
    assert any("embed" in n for n in op_names), f"No embed op in {op_names}"
    assert any("lm_head" in n for n in op_names), f"No lm_head op in {op_names}"


def test_estimate_old_path_still_works():
    """Old (non-builtin) path still produces valid estimates."""
    from zrt.training.search.estimator import estimate

    model = _make_dummy_model()
    system = _make_dummy_system()
    # world_size = 2 to match dp=2
    system.nodes = 2
    strategy = Strategy(tp=1, pp=1, dp=2, micro_batch=1, global_batch=2,
                       zero_stage=0, builtin_model_id=None)

    report = estimate(model, system, strategy)
    assert report.step_time_ms > 0
    assert report.mfu >= 0.0


def test_aggegate_scope_matching():
    """All expected scope patterns in _SCOPE_RULES work correctly."""
    from zrt.training.ir.from_opgraph import _match_scope

    pairs = [
        ("model.layers.0.self_attn.q_a_proj", "matmul"),
        ("model.layers.0.self_attn.q_proj", "matmul"),
        ("model.layers.0.self_attn.o_proj", "matmul"),
        ("model.layers.0.mlp.gate_proj", "matmul"),
        ("model.layers.0.mlp.down_proj", "matmul"),
        ("model.layers.0.input_layernorm", "ln"),
        ("model.layers.0.post_attention_layernorm", "ln"),
        ("model.embed_tokens", "embed"),
        ("lm_head", "lm_head"),
        ("model.layers.0.mlp.MoEGate", "router"),
    ]
    for scope, expected_kind in pairs:
        result = _match_scope(scope)
        assert result is not None, f"No match for {scope}"
        assert result[0] == expected_kind, \
            f"{scope}: expected {expected_kind}, got {result[0]}"


@pytest.fixture(autouse=True)
def _cleanup_builtins():
    """Remove test artifacts after each test."""
    yield
    from pathlib import Path
    import glob as _glob
    model_dir = Path(__file__).parent.parent.parent / "python" / "zrt" / \
                "training" / "builtins" / "models"
    try:
        for f in _glob.glob(str(model_dir / "test_model*.*")):
            Path(f).unlink(missing_ok=True)
    except Exception:
        pass  # ok if dir doesn't exist
