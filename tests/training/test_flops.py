"""Test FLOPs model — 6P rule, matmul cost."""

import pytest
from zrt.training.ir.builders import build_graph
from zrt.training.models.flops import OpCost, op_cost, total_training_flops
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import ModelSpec, LayerKind
from zrt.training.spec.strategy import Strategy


def test_matmul_cost():
    """Matmul: fwd = dx = dw = 2*m*n*k."""
    from zrt.training.ir.graph import Op
    op = Op(name="test_mm", kind="matmul", meta={"m": 1024, "n": 4096, "k": 4096})
    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE],
    )
    cost = op_cost(op, model)
    expected = 2 * 1024 * 4096 * 4096
    assert cost.fwd_flops == expected
    assert cost.dx_flops == expected
    assert cost.dw_flops == expected
    assert cost.bound == "compute"


def test_attn_core_cost():
    """Attention core: causal fwd = 2*b*s^2*h*d."""
    from zrt.training.ir.graph import Op
    op = Op(name="test_attn", kind="attn_core", meta={
        "b": 1, "s": 2048, "heads": 32, "head_dim": 128, "causal": True,
    })
    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE],
    )
    cost = op_cost(op, model)
    # fwd = 2 * 1 * 2048 * 2048 * 32 * 128
    expected_fwd = 2 * 1 * 2048 * 2048 * 32 * 128
    assert cost.fwd_flops == expected_fwd
    assert cost.dx_flops == pytest.approx(2.5 * expected_fwd, rel=0.01)
    assert cost.dw_flops == 0.0


def test_memory_bound_cost():
    """Memory-bound ops (ln, softmax, etc.) should have byte traffic."""
    from zrt.training.ir.graph import Op
    op = Op(name="test_ln", kind="ln", meta={"bytes_fwd": 1000})
    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE],
    )
    cost = op_cost(op, model)
    assert cost.bound == "memory"
    assert cost.fwd_bytes == 1000
    assert cost.dx_bytes > 0


def test_6p_rule():
    """Total training FLOPs for dense model should follow 6P rule."""
    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * 4,
    )
    strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1, global_batch=1)
    graph = build_graph(model, strategy)

    total = total_training_flops(graph, model, strategy)

    # 6P rule: 6 * total_params * tokens
    tokens = 1 * 2048  # micro_batch * seq_len
    P = model.total_params()
    expected_6p = 6 * P * tokens

    # Allow 50% tolerance because the 6P rule is approximate
    # (it doesn't account for embedding/lm_head exactly, and we have
    # memory-bound ops that don't contribute FLOPs)
    ratio = total / expected_6p
    assert 0.5 < ratio < 1.5, f"6P ratio: {ratio:.2f}, total={total:.2e}, 6P={expected_6p:.2e}"


def test_unknown_op_zero_cost():
    """Unknown op kinds should return zero cost."""
    from zrt.training.ir.graph import Op
    op = Op(name="unknown", kind="custom_op", meta={})
    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE],
    )
    cost = op_cost(op, model)
    assert cost.fwd_flops == 0.0
    assert cost.dx_flops == 0.0


def test_moe_effective_params_is_sane():
    """MoE effective params should be less than total params when top_k < num_experts."""
    from zrt.training.spec.system import GPU, NetTier, SystemSpec
    from zrt.training.spec.strategy import Strategy
    from zrt.training.compose.pipeline import compute_mfu

    # Minimal MoE model: 2 layers, 4 experts, top_k=1
    model = ModelSpec(
        hidden=4096, ffn=2048, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.MOE] * 2,
        num_experts=4, moe_ffn=1024, top_k=1,
    )

    total = model.total_params()
    effective = model.effective_params_for_flops()

    # Effective should be less than total since only 1/4 of experts active
    assert effective < total, f"effective={effective}, total={total}"
    ratio = effective / total
    # With top_k=1, num_experts=4:
    # - Attention, router, shared FFN are 100% active
    # - Only routed experts are sparse (1/4 active)
    # So ratio should be between 50% (mostly routed experts) and 85% (mostly attention/shared)
    assert 0.5 < ratio < 0.9, f"effective/total ratio {ratio:.3f} outside expected range for top_k=1, num_experts=4"


def test_moe_mfu_is_sane():
    """MoE MFU should be between 0 and 1, not collapse to 1.0."""
    from zrt.training.io.config_loader import load_specs
    from zrt.training.search.estimator import estimate

    # Run estimate on deepseek_v3 config
    model, system, strategy = load_specs("python/zrt/training/configs/llama3_70b_3d.yaml")

    # Temporarily make it MoE-like for testing
    from dataclasses import replace
    model_moe = replace(model,
        layers=[LayerKind.MOE] * 2,
        num_experts=4,
        moe_ffn=1024,
        top_k=1,
    )

    report = estimate(model_moe, system, strategy)

    # MFU should be sane: strictly between 0 and 1 (not 0, not 1)
    assert 0.0 < report.mfu < 1.0, f"MoE MFU collapsed to {report.mfu}, expected 0 < MFU < 1"
