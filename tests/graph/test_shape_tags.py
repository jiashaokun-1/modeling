"""Tests for shape tagging during graph capture (Phase 1)."""

import pytest
from python.zrt.graph.tensor_utils import tag_dims, tags_str


def test_tag_dims_prefill():
    """QKV input (4096, 7168) in prefill: batch=1, seq=4096, hidden fixed."""
    tags = tag_dims((4096, 7168), batch=1, query_len=4096, seq_len=4096,
                    fixed_values={7168})
    assert tags == ["BQ", 7168], f"Expected [BQ, 7168], got {tags}"


def test_tag_dims_decode():
    """QKV input (1, 7168) in decode: batch=1, query_len=1."""
    tags = tag_dims((1, 7168), batch=1, query_len=1, seq_len=4096,
                    fixed_values={7168})
    assert tags == ["BQ", 7168], f"Expected [BQ, 7168], got {tags}"


def test_tag_dims_attention_mask():
    """Attention mask (1, 1, 4096, 4096): dims 0/1 fixed, dims 2/3 = BS (batch*seq)."""
    tags = tag_dims((1, 1, 4096, 4096), batch=1, query_len=4096, seq_len=4096,
                    fixed_values=set())
    # In prefill, BS == BQ (both = 4096); BQ checked first, so we get BQ
    # The first two dims are 1 = batch_size = B
    assert tags == ["B", "B", "BQ", "BQ"], f"Expected [B, B, BQ, BQ], got {tags}"


def test_tag_dims_ffn():
    """FFN gate_proj up output (4096, 18432): seq*batch, ffn fixed."""
    tags = tag_dims((4096, 18432), batch=1, query_len=4096, seq_len=4096,
                    fixed_values={18432})
    assert tags == ["BQ", 18432], f"Expected [BQ, 18432], got {tags}"


def test_tag_dims_variable_priority():
    """When hidden matches a variable dim value, variable tag wins.

    Use decode mode (query_len=1, seq_len=4096, batch=2) so that:
    - BQ = batch*query_len = 2*1 = 2
    - BS = batch*seq_len = 2*4096 = 8192
    - hidden = 4096 which equals BS/2 but not BS or BQ directly.

    Test with shape (8192, 4096): first dim = BS, second dim = seq_len (S).
    The dim 4096 could be a static value (hidden=4096) but S priority wins.
    """
    tags = tag_dims((8192, 4096), batch=2, query_len=1, seq_len=4096,
                    fixed_values={4096, 8192})
    # First dim 8192 = batch*seq = BS
    # Second dim 4096 = seq_len (S), NOT static 4096 (variable priority)
    assert tags == ["BS", "S"], f"Expected [BS, S], got {tags}"


def test_tag_dims_bs_vs_bq():
    """BS takes priority over BQ when batch*seq != batch*query."""
    # batch=2, seq_len=4096, query_len=1 → BS=8192, BQ=2
    tags = tag_dims((8192, 7168), batch=2, query_len=1, seq_len=4096,
                    fixed_values={7168})
    assert tags == ["BS", 7168], f"Expected [BS, 7168], got {tags}"


def test_tags_str_roundtrip():
    """tags_str produces bracket notation compatible with split_shape_list."""
    result = tags_str(["BQ", 7168, "S"])
    assert result == "[BQ, 7168, S]", f"Unexpected format: {result}"


def test_shape_tags_record_integration():
    """Quick integration: dispatch records carry shape tags."""
    from python.zrt.graph.dispatch import RecordingDispatch, TensorTracker

    tracker = TensorTracker()
    geom = {"batch_size": 1, "seq_len": 128, "query_len": 128,
            "hidden_size": 4096, "num_attention_heads": 32}
    recorder = RecordingDispatch(
        tensor_tracker=tracker, geometry_params=geom, active=False)
    assert recorder._batch == 1
    assert recorder._seq_len == 128
    assert 4096 in recorder._fixed_values
