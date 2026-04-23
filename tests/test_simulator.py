"""Tests for python.zrt.simulator (Roofline + SimulatorHub).

Run:
    pytest tests/test_simulator.py -v
"""
from __future__ import annotations

import math

import pytest

from python.zrt.ir.node import OpNode
from python.zrt.ir.types import DType, TensorMeta
from python.zrt.hardware import load as load_hw
from python.zrt.simulator import SimulatorHub, RooflineSimulator, SimResult, content_hash


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def hw_910b():
    return load_hw("ascend_910b")


@pytest.fixture(scope="module")
def hw_h100():
    return load_hw("nvidia_h100_sxm")


@pytest.fixture(scope="module")
def hub():
    return SimulatorHub.default()


def _tm(tid: str, shape: tuple[int, ...], dtype: DType = DType.BF16) -> TensorMeta:
    return TensorMeta.from_shape_dtype(tid, shape, dtype)


def _node(op_type: str, inputs: list[TensorMeta], outputs: list[TensorMeta],
          fused_from: list[str] | None = None) -> OpNode:
    return OpNode(
        id="test_node",
        op_type=op_type,
        inputs=inputs,
        outputs=outputs,
        fused_from=fused_from or [],
    )


# ─────────────────────────────────────────────────────────────────────────────
# SimResult basic contract
# ─────────────────────────────────────────────────────────────────────────────

def test_simresult_fields():
    r = SimResult(
        op_node_id="x", latency_us=10.0, compute_us=8.0, memory_us=5.0,
        flops=int(1e9), read_bytes=int(1e6), write_bytes=int(1e6),
        arithmetic_intensity=500.0, bound="compute",
        hw_utilization=0.8, backend="roofline", confidence=0.3,
    )
    assert r.total_bytes == 2_000_000
    assert math.isclose(r.latency_ms, 0.01)
    assert "compute" in repr(r)


# ─────────────────────────────────────────────────────────────────────────────
# Roofline formulas — matmul variants
# ─────────────────────────────────────────────────────────────────────────────

class TestRooflineMatmul:

    def test_mm_basic(self, hw_910b):
        M, K, N = 1024, 4096, 7168
        node = _node(
            "aten.mm.default",
            inputs  = [_tm("a", (M, K)), _tm("b", (K, N))],
            outputs = [_tm("o", (M, N))],
        )
        sim = RooflineSimulator()
        r = sim.simulate(node, hw_910b)
        expected_flops = 2 * M * N * K
        assert r.flops == expected_flops, f"Expected {expected_flops}, got {r.flops}"
        assert r.latency_us > 0
        assert r.bound in ("compute", "memory")
        assert 0.0 <= r.hw_utilization <= 1.0

    def test_addmm(self, hw_910b):
        M, K, N = 512, 2048, 4096
        node = _node(
            "aten.addmm.default",
            inputs  = [_tm("bias", (N,)), _tm("a", (M, K)), _tm("b", (K, N))],
            outputs = [_tm("o", (M, N))],
        )
        sim = RooflineSimulator()
        r = sim.simulate(node, hw_910b)
        # FLOPs = 2*M*N*K + M*N (bias add)
        expected = 2 * M * N * K + M * N
        assert r.flops == expected

    def test_bmm(self, hw_910b):
        B, M, K, N = 4, 128, 512, 512
        node = _node(
            "aten.bmm.default",
            inputs  = [_tm("a", (B, M, K)), _tm("b", (B, K, N))],
            outputs = [_tm("o", (B, M, N))],
        )
        sim = RooflineSimulator()
        r = sim.simulate(node, hw_910b)
        expected = 2 * B * M * N * K
        assert r.flops == expected

    def test_linear_with_bias(self, hw_910b):
        batch, I, O = 128, 4096, 7168
        node = _node(
            "aten.linear.default",
            inputs  = [_tm("x", (batch, I)), _tm("w", (O, I)), _tm("b", (O,))],
            outputs = [_tm("o", (batch, O))],
        )
        sim = RooflineSimulator()
        r = sim.simulate(node, hw_910b)
        # FLOPs = 2*batch*O*I + batch*O
        expected = 2 * batch * O * I + batch * O
        assert r.flops == expected

    def test_mm_compute_bound(self, hw_910b):
        """Large matmul on 910B should be compute-bound."""
        M, K, N = 4096, 8192, 7168
        node = _node(
            "aten.mm.default",
            inputs  = [_tm("a", (M, K)), _tm("b", (K, N))],
            outputs = [_tm("o", (M, N))],
        )
        r = RooflineSimulator().simulate(node, hw_910b)
        assert r.bound == "compute", f"Expected compute-bound, got {r.bound}"

    def test_mm_memory_bound_small(self, hw_910b):
        """Very thin matmul (decode step) should be memory-bound."""
        M, K, N = 1, 7168, 7168   # single-token decode
        node = _node(
            "aten.mm.default",
            inputs  = [_tm("a", (M, K)), _tm("b", (K, N))],
            outputs = [_tm("o", (M, N))],
        )
        r = RooflineSimulator().simulate(node, hw_910b)
        assert r.bound == "memory", f"Expected memory-bound for decode step, got {r.bound}"


# ─────────────────────────────────────────────────────────────────────────────
# Norm / softmax / elementwise
# ─────────────────────────────────────────────────────────────────────────────

class TestRooflineOtherOps:

    def test_layer_norm(self, hw_910b):
        B, S, H = 1, 128, 7168
        node = _node(
            "aten.layer_norm.default",
            inputs  = [_tm("x", (B, S, H)), _tm("w", (H,)), _tm("b", (H,))],
            outputs = [_tm("o", (B, S, H))],
        )
        r = RooflineSimulator().simulate(node, hw_910b)
        assert r.flops == 5 * B * S * H
        assert r.bound == "memory"   # norm is always memory-bound

    def test_softmax(self, hw_910b):
        B, H, S = 1, 16, 512
        node = _node(
            "aten._softmax.default",
            inputs  = [_tm("x", (B, H, S, S))],
            outputs = [_tm("o", (B, H, S, S))],
        )
        r = RooflineSimulator().simulate(node, hw_910b)
        assert r.flops == 5 * B * H * S * S
        assert r.bound == "memory"

    def test_elementwise_add(self, hw_910b):
        shape = (1, 128, 7168)
        node = _node(
            "aten.add.Tensor",
            inputs  = [_tm("a", shape), _tm("b", shape)],
            outputs = [_tm("o", shape)],
        )
        r = RooflineSimulator().simulate(node, hw_910b)
        assert r.flops == 1 * 128 * 7168
        assert r.bound == "memory"

    def test_silu_activation(self, hw_910b):
        n = 1024
        node = _node(
            "aten.silu.default",
            inputs  = [_tm("x", (n,))],
            outputs = [_tm("o", (n,))],
        )
        r = RooflineSimulator().simulate(node, hw_910b)
        assert r.flops == 4 * n   # 4 ops/elem for silu

    def test_sdpa(self, hw_910b):
        B, H, Sq, D = 1, 32, 128, 128
        Sk = 128
        node = _node(
            "aten._scaled_dot_product_flash_attention.default",
            inputs  = [_tm("q", (B, H, Sq, D)), _tm("k", (B, H, Sk, D)), _tm("v", (B, H, Sk, D))],
            outputs = [_tm("o", (B, H, Sq, D))],
        )
        r = RooflineSimulator().simulate(node, hw_910b)
        # FLOPs = 4*B*H*Sq*Sk*D + softmax(5*B*H*Sq*Sk)
        expected = 4 * B * H * Sq * Sk * D + 5 * B * H * Sq * Sk
        assert r.flops == expected

    def test_embedding(self, hw_910b):
        B, S, H = 1, 32, 4096
        node = _node(
            "aten.embedding.default",
            inputs  = [_tm("w", (32000, H)), _tm("idx", (B, S), DType.INT64)],
            outputs = [_tm("o", (B, S, H))],
        )
        r = RooflineSimulator().simulate(node, hw_910b)
        assert r.flops == 0


# ─────────────────────────────────────────────────────────────────────────────
# Shape / view ops
# ─────────────────────────────────────────────────────────────────────────────

def test_shape_ops_near_zero_flops(hw_910b):
    node = _node(
        "aten.view.default",
        inputs  = [_tm("x", (128, 512))],
        outputs = [_tm("o", (64, 1024))],
    )
    r = RooflineSimulator().simulate(node, hw_910b)
    assert r.flops == 0


# ─────────────────────────────────────────────────────────────────────────────
# Fused semantic label ops
# ─────────────────────────────────────────────────────────────────────────────

class TestFusedOps:

    def test_rms_norm_label(self, hw_910b):
        B, S, H = 1, 128, 7168
        node = _node(
            "rms_norm",
            inputs  = [_tm("x", (B, S, H)), _tm("w", (H,))],
            outputs = [_tm("o", (B, S, H))],
        )
        r = RooflineSimulator().simulate(node, hw_910b)
        assert r.flops == 4 * B * S * H
        assert r.bound == "memory"

    def test_add_rms_norm_label(self, hw_910b):
        B, S, H = 1, 128, 7168
        node = _node(
            "add_rms_norm",
            inputs  = [_tm("x", (B, S, H)), _tm("res", (B, S, H)), _tm("w", (H,))],
            outputs = [_tm("o", (B, S, H))],
        )
        r = RooflineSimulator().simulate(node, hw_910b)
        assert r.flops == 6 * B * S * H   # 5 + 1 for add

    def test_flash_attn_label(self, hw_910b):
        B, H, S, D = 1, 32, 128, 128
        node = _node(
            "flash_attn",
            inputs  = [_tm("q", (B, H, S, D)), _tm("k", (B, H, S, D)), _tm("v", (B, H, S, D))],
            outputs = [_tm("o", (B, H, S, D))],
        )
        r = RooflineSimulator().simulate(node, hw_910b)
        assert r.flops > 0
        assert r.latency_us > 0

    def test_gated_mlp_label(self, hw_910b):
        B, S, H, I = 1, 128, 7168, 18944
        node = _node(
            "gated_mlp",
            inputs  = [_tm("x", (B, S, H)), _tm("gate_w", (I, H)), _tm("up_w", (I, H)), _tm("down_w", (H, I))],
            outputs = [_tm("o", (B, S, H))],
        )
        r = RooflineSimulator().simulate(node, hw_910b)
        assert r.flops > 0
        assert r.latency_us > 0

    def test_fused_with_fused_from(self, hw_910b):
        """Fused node whose op_type is unknown but fused_from has known ops."""
        M, K, N = 512, 4096, 4096
        node = OpNode(
            id="fused_0",
            op_type="unknown_fused_op",
            inputs  = [_tm("a", (M, K)), _tm("b", (K, N))],
            outputs = [_tm("o", (M, N))],
            fused_from=["aten.mm.default", "aten.relu.default"],
        )
        r = RooflineSimulator().simulate(node, hw_910b)
        assert r.flops > 0


# ─────────────────────────────────────────────────────────────────────────────
# SimulatorHub
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulatorHub:

    def test_default_hub_has_roofline(self, hw_910b):
        from python.zrt.simulator.backends.backend_register import BACKEND_MAP, BackendType
        assert BackendType.ROOFLINE in BACKEND_MAP

    def test_simulate_single_node(self, hub, hw_910b):
        M, K, N = 1024, 4096, 7168
        node = _node(
            "aten.mm.default",
            inputs  = [_tm("a", (M, K)), _tm("b", (K, N))],
            outputs = [_tm("o", (M, N))],
        )
        node.id = "op_0"
        r = hub.simulate(node, hw_910b)
        assert isinstance(r, SimResult)
        assert r.op_node_id == "op_0"
        assert r.backend == "roofline"

    def test_cache_hit(self, hw_910b):
        hub = SimulatorHub.default()
        node = _node(
            "aten.mm.default",
            inputs  = [_tm("a", (64, 128)), _tm("b", (128, 64))],
            outputs = [_tm("o", (64, 64))],
        )
        node.id = "op_cache"
        hub.simulate(node, hw_910b)
        hub.simulate(node, hw_910b)
        assert hub.cache_stats["hits"] >= 1

    def test_custom_backend_priority(self, hw_910b):
        """A higher-priority backend should override roofline."""
        from python.zrt.simulator.base import OpSimulator

        class MockBackend(OpSimulator):
            name = "mock"
            priority = 99

            def can_simulate(self, node, hw):
                return node.op_type == "aten.mm.default"

            def simulate(self, node, hw):
                return SimResult(
                    op_node_id=node.id, latency_us=42.0,
                    compute_us=42.0, memory_us=0.0,
                    flops=0, read_bytes=0, write_bytes=0,
                    arithmetic_intensity=float("inf"), bound="compute",
                    hw_utilization=1.0, backend="mock", confidence=1.0,
                )

        hub = SimulatorHub.default()
        hub.register(MockBackend())

        node = _node(
            "aten.mm.default",
            inputs  = [_tm("a", (64, 64)), _tm("b", (64, 64))],
            outputs = [_tm("o", (64, 64))],
        )
        node.id = "op_mock"
        r = hub.simulate(node, hw_910b)
        assert r.backend == "mock"
        assert r.latency_us == 42.0

    def test_simulate_graph(self, hw_910b):
        """End-to-end: simulate a tiny 3-node graph."""
        from python.zrt.ir.graph import OpGraph
        from python.zrt.ir.edge import Edge

        graph = OpGraph(name="tiny", phase="prefill")

        n0 = OpNode(id="op_0", op_type="aten.mm.default",
                    inputs=[_tm("a", (64, 128)), _tm("b", (128, 64))],
                    outputs=[_tm("out0", (64, 64))])
        n1 = OpNode(id="op_1", op_type="aten.silu.default",
                    inputs=[_tm("out0", (64, 64))],
                    outputs=[_tm("out1", (64, 64))])
        n2 = OpNode(id="op_2", op_type="aten.add.Tensor",
                    inputs=[_tm("out1", (64, 64)), _tm("res", (64, 64))],
                    outputs=[_tm("out2", (64, 64))])

        for n in (n0, n1, n2):
            graph.add_node(n)
        graph.add_edge(Edge(src="op_0", src_idx=0, dst="op_1", dst_idx=0, tensor=n0.outputs[0]))
        graph.add_edge(Edge(src="op_1", src_idx=0, dst="op_2", dst_idx=0, tensor=n1.outputs[0]))

        hub = SimulatorHub.default()
        results = hub.simulate_graph(graph, hw_910b)

        assert len(results) == 3
        for nid, r in results.items():
            assert isinstance(r, SimResult)
            assert r.op_node_id == nid
            assert r.latency_us > 0


# ─────────────────────────────────────────────────────────────────────────────
# Content hash
# ─────────────────────────────────────────────────────────────────────────────

def test_content_hash_stable(hw_910b):
    node = _node(
        "aten.mm.default",
        inputs=[_tm("a", (64, 64)), _tm("b", (64, 64))],
        outputs=[_tm("o", (64, 64))],
    )
    h1 = content_hash(node, hw_910b)
    h2 = content_hash(node, hw_910b)
    assert h1 == h2


def test_content_hash_different_shapes(hw_910b):
    n1 = _node("aten.mm.default",
               inputs=[_tm("a", (64, 64)), _tm("b", (64, 64))],
               outputs=[_tm("o", (64, 64))])
    n2 = _node("aten.mm.default",
               inputs=[_tm("a", (128, 64)), _tm("b", (64, 64))],
               outputs=[_tm("o", (128, 64))])
    assert content_hash(n1, hw_910b) != content_hash(n2, hw_910b)


def test_content_hash_different_hw(hw_910b, hw_h100):
    node = _node("aten.mm.default",
                 inputs=[_tm("a", (64, 64)), _tm("b", (64, 64))],
                 outputs=[_tm("o", (64, 64))])
    assert content_hash(node, hw_910b) != content_hash(node, hw_h100)


# ─────────────────────────────────────────────────────────────────────────────
# Arithmetic intensity sanity
# ─────────────────────────────────────────────────────────────────────────────

def test_arithmetic_intensity_large_matmul(hw_910b):
    """Large matmul should have high AI (compute-bound territory)."""
    M, K, N = 4096, 4096, 4096
    node = _node("aten.mm.default",
                 inputs=[_tm("a", (M, K)), _tm("b", (K, N))],
                 outputs=[_tm("o", (M, N))])
    r = RooflineSimulator().simulate(node, hw_910b)
    # AI = 2*M*N*K / ((M*K + K*N + M*N) * 2 bytes)
    # ≈ 2*4096^3 / (3 * 4096^2 * 2) = 4096/3 ≈ 1365 ops/byte
    assert r.arithmetic_intensity > 100


def test_arithmetic_intensity_elementwise(hw_910b):
    """Elementwise add should have very low AI (memory-bound)."""
    shape = (4096, 4096)
    node = _node("aten.add.Tensor",
                 inputs=[_tm("a", shape), _tm("b", shape)],
                 outputs=[_tm("o", shape)])
    r = RooflineSimulator().simulate(node, hw_910b)
    # AI = (4096*4096) / (3 * 4096*4096 * 2) = 1/6 ≈ 0.17 ops/byte
    assert r.arithmetic_intensity < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Cross-hardware comparison
# ─────────────────────────────────────────────────────────────────────────────

def test_h100_faster_than_910b_on_large_mm(hw_910b, hw_h100):
    """H100 has higher BF16 peak than 910B, so large mm should be faster."""
    M, K, N = 4096, 4096, 4096
    node = _node("aten.mm.default",
                 inputs=[_tm("a", (M, K)), _tm("b", (K, N))],
                 outputs=[_tm("o", (M, N))])
    sim = RooflineSimulator()
    r_910b = sim.simulate(node, hw_910b)
    r_h100 = sim.simulate(node, hw_h100)
    # H100 BF16 = 989T vs 910B BF16 = 320T
    assert r_h100.latency_us < r_910b.latency_us
