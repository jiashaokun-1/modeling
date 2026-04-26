"""Tests for retemplate (Phase 1.3)."""

import pytest
from zrt.ir.types import DType, TensorMeta
from zrt.ir.graph import OpGraph
from zrt.ir.node import OpNode
from zrt.ir.retemplate import retemplate


def _make_graph_with_template():
    """Create a tiny OpGraph with shape_template for testing."""
    g = OpGraph(name="test", phase="prefill")

    # Node with tagged input/output
    node = OpNode(
        id="op_0", op_type="aten.mm.default",
        inputs=[TensorMeta.from_shape_dtype("t0", (4096, 7168), DType.BF16,
                                            shape_template=("BQ", 7168))],
        outputs=[TensorMeta.from_shape_dtype("t1", (4096, 16384), DType.BF16,
                                             shape_template=("BQ", 16384))],
        scope="model.layers.0.self_attn.q_proj",
    )
    g.add_node(node)
    return g


def test_retemplate_seq_len_change():
    """Rebind seq_len 4096 → 8192 and verify shapes update."""
    g = _make_graph_with_template()
    g2 = retemplate(g, batch_size=1, seq_len=8192)

    node = g2.nodes["op_0"]
    assert node.inputs[0].shape == (8192, 7168), \
        f"Expected (8192, 7168), got {node.inputs[0].shape}"
    assert node.outputs[0].shape == (8192, 16384), \
        f"Expected (8192, 16384), got {node.outputs[0].shape}"


def test_retemplate_batch_change():
    """Rebind batch 1 → 2 and verify shapes update."""
    g = _make_graph_with_template()
    g2 = retemplate(g, batch_size=2, seq_len=4096)

    node = g2.nodes["op_0"]
    assert node.inputs[0].shape == (8192, 7168), \
        f"batch=2,seq=4096: expected (8192, 7168), got {node.inputs[0].shape}"


def test_retemplate_preserves_fixed_dims():
    """Static dimensions (7168, 16384) remain unchanged."""
    g = _make_graph_with_template()
    g2 = retemplate(g, batch_size=2, seq_len=8192)

    node = g2.nodes["op_0"]
    assert node.inputs[0].shape[1] == 7168
    assert node.outputs[0].shape[1] == 16384


def test_retemplate_no_template():
    """Node without shape_template is left untouched."""
    g = OpGraph(name="test", phase="prefill")
    node = OpNode(
        id="op_0", op_type="aten.mm.default",
        inputs=[TensorMeta.from_shape_dtype("t0", (4096, 7168), DType.BF16)],
        outputs=[TensorMeta.from_shape_dtype("t1", (4096, 16384), DType.BF16)],
    )
    g.add_node(node)
    g2 = retemplate(g, batch_size=2, seq_len=8192)
    # Shape unchanged (no template to rebind)
    assert g2.nodes["op_0"].inputs[0].shape == (4096, 7168)
