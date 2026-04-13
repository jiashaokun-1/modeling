"""Self-validation test suite for screenshot_ops.py.

Covers:
  * All local model configs in hf_models/ (no network required)
  * HF Hub model IDs for DeepSeek-V3.2 and Qwen3 (requires network + HF access)
  * Unit tests for component classification and MoE detection helpers

Run all local tests (fast, no network):
    pytest test_screenshot_ops.py -v -m "not network"

Run including HF Hub tests (downloads config ~KB, not weights):
    pytest test_screenshot_ops.py -v

Run a single model end-to-end:
    pytest test_screenshot_ops.py -v -k "deepseek_v3"
"""
from __future__ import annotations

import socket
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from typing import Any

# ── helpers ───────────────────────────────────────────────────────────────────

HF_MODELS = Path(__file__).parent / "hf_models"


def _network_available(host: str = "huggingface.co", port: int = 443,
                        timeout: float = 3.0) -> bool:
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except OSError:
        return False


network = pytest.mark.skipif(
    not _network_available(),
    reason="HuggingFace Hub not reachable — skipping network tests",
)

# ── import subject ────────────────────────────────────────────────────────────

from screenshot_ops import (
    _classify_component,
    _extract_layer_idx,
    _is_moe_module,
    _patch_moe_for_meta,
    build_config_summary,
    load_model,
    run_trace,
)


# ═════════════════════════════════════════════════════════════════════════════
# Unit tests — no model loading
# ═════════════════════════════════════════════════════════════════════════════

class TestExtractLayerIdx:
    def test_standard_layers(self):
        assert _extract_layer_idx("model.layers.0.self_attn.q_proj") == "0"
        assert _extract_layer_idx("model.layers.5.mlp.gate_proj") == "5"
        assert _extract_layer_idx("transformer.h.3.attn") == "3"

    def test_no_layer(self):
        assert _extract_layer_idx("model.embed_tokens") == ""
        assert _extract_layer_idx("lm_head") == ""
        assert _extract_layer_idx("model.norm") == ""


class TestClassifyComponent:
    # ── norm ─────────────────────────────────────────────────────────────────
    def test_pre_attn_norm(self):
        c = _classify_component(
            "model.layers.0.input_layernorm", "aten.mul.Tensor")
        assert c == "norm.pre_attn"

    def test_post_attn_norm(self):
        c = _classify_component(
            "model.layers.0.post_attention_layernorm", "aten.mul.Tensor")
        assert c == "norm.post_attn"

    def test_final_norm(self):
        c = _classify_component("model.norm", "aten.mul.Tensor")
        assert c == "norm.final"

    # ── attention ────────────────────────────────────────────────────────────
    def test_attn_q_proj(self):
        c = _classify_component(
            "model.layers.0.self_attn.q_proj", "aten.mm.default")
        assert c == "attn.q_proj"

    def test_attn_o_proj(self):
        c = _classify_component(
            "model.layers.0.self_attn.o_proj", "aten.mm.default")
        assert c == "attn.o_proj"

    def test_attn_softmax(self):
        c = _classify_component(
            "model.layers.0.self_attn", "aten.softmax.int")
        assert c == "attn.softmax"

    def test_attn_score_matmul(self):
        c = _classify_component(
            "model.layers.0.self_attn", "aten.matmul.default")
        assert c == "attn.qk_score"

    # ── MLA (DeepSeek) ───────────────────────────────────────────────────────
    def test_mla_q_a_proj(self):
        c = _classify_component(
            "model.layers.0.self_attn.q_a_proj", "aten.mm.default")
        assert c == "attn.q_a_proj"

    def test_mla_kv_a_proj(self):
        c = _classify_component(
            "model.layers.0.self_attn.kv_a_proj_with_mqa", "aten.mm.default")
        assert c == "attn.kv_a_proj"

    # ── MoE ──────────────────────────────────────────────────────────────────
    def test_moe_router(self):
        c = _classify_component(
            "model.layers.3.mlp.gate", "aten.mm.default")
        assert c.startswith("moe.router.")

    def test_moe_expert_proj(self):
        c = _classify_component(
            "model.layers.3.mlp.experts.0.gate_proj", "aten.mm.default")
        assert c == "moe.expert.gate_proj"

    def test_moe_shared_expert(self):
        c = _classify_component(
            "model.layers.3.mlp.shared_experts.down_proj", "aten.mm.default")
        assert c == "moe.shared.down_proj"

    # ── FFN ──────────────────────────────────────────────────────────────────
    def test_ffn_gate_proj(self):
        c = _classify_component(
            "model.layers.0.mlp.gate_proj", "aten.mm.default")
        assert c == "ffn.gate_proj"

    def test_ffn_up_proj(self):
        c = _classify_component(
            "model.layers.0.mlp.up_proj", "aten.mm.default")
        assert c == "ffn.up_proj"

    def test_ffn_down_proj(self):
        c = _classify_component(
            "model.layers.0.mlp.down_proj", "aten.mm.default")
        assert c == "ffn.down_proj"

    # ── embedding / lm_head ──────────────────────────────────────────────────
    def test_embedding(self):
        assert _classify_component("model.embed_tokens", "aten.embedding.default") == "embedding"

    def test_lm_head(self):
        assert _classify_component("lm_head", "aten.mm.default") == "lm_head"


class TestMoEDetection:
    def test_detects_module_list_experts(self):
        class FakeMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])

        assert _is_moe_module(FakeMoE())

    def test_ignores_single_module(self):
        class FakeDense(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = nn.Linear(8, 8)   # not a ModuleList

        assert not _is_moe_module(FakeDense())

    def test_ignores_already_patched(self):
        class FakeMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = nn.ModuleList([nn.Linear(8, 8)])
                self._meta_patched = True

        assert not _is_moe_module(FakeMoE())

    def test_patch_replaces_forward(self):
        class FakeExpert(nn.Module):
            def forward(self, x):
                return x * 2

        class FakeMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = nn.ModuleList([FakeExpert()])

            def forward(self, x):
                raise RuntimeError("original forward should not run")

        moe = FakeMoE()
        original_fwd = moe.forward
        _patch_moe_for_meta(moe)
        assert moe.forward is not original_fwd, "forward should have been replaced"
        assert moe._meta_patched is True


# ═════════════════════════════════════════════════════════════════════════════
# Local model tests — reads only config.json, no weights download
# ═════════════════════════════════════════════════════════════════════════════

def _assert_trace_valid(records, model_id: str) -> None:
    """Common assertions for a successful trace."""
    assert len(records) > 0, f"{model_id}: no ops captured"

    ops = {r["aten_op"] for r in records}
    # Every transformer model produces at least one matrix multiplication
    mm_ops = {o for o in ops if any(x in o for x in ("mm", "matmul", "bmm", "addmm"))}
    assert mm_ops, f"{model_id}: no matmul-family ops found — got {ops}"

    components = {r["component"] for r in records}
    # At minimum: some attention op and some FFN op
    has_attn = any(c.startswith("attn.") for c in components)
    has_ffn  = any(c.startswith(("ffn.", "moe.")) for c in components)
    assert has_attn, f"{model_id}: no attention components found — got {components}"
    assert has_ffn,  f"{model_id}: no FFN/MoE components found — got {components}"


@pytest.mark.parametrize("model_path,num_layers", [
    pytest.param(str(HF_MODELS / "deepseek_v3"), 4,
                 id="local-deepseek_v3"),
    pytest.param(str(HF_MODELS / "llama3_8b"), 2,
                 id="local-llama3_8b"),
    pytest.param(str(HF_MODELS / "llama3_70b"), 2,
                 id="local-llama3_70b"),
    pytest.param(str(HF_MODELS / "qwen2_7b"), 2,
                 id="local-qwen2_7b"),
    pytest.param(str(HF_MODELS / "qwen2_72b"), 2,
                 id="local-qwen2_72b"),
    pytest.param(str(HF_MODELS / "mistral_7b"), 2,
                 id="local-mistral_7b"),
    pytest.param(str(HF_MODELS / "mixtral_8x7b"), 2,
                 id="local-mixtral_8x7b"),
])
def test_local_model(model_path: str, num_layers: int, tmp_path: Path):
    """End-to-end trace on a local config (no weight download)."""
    if not Path(model_path).is_dir():
        pytest.skip(f"Local model dir not found: {model_path}")

    out, records = run_trace(
        model_id=model_path,
        num_layers=num_layers,
        batch_size=1,
        seq_len=64,
        output_path=tmp_path / "ops.xlsx",
    )
    assert out.exists(), "Excel file was not created"
    _assert_trace_valid(records, model_path)

    # Report for human review
    components = sorted({r["component"] for r in records})
    print(f"\n[{Path(model_path).name}]  {len(records)} ops  "
          f"components: {components}")


@pytest.mark.parametrize("model_path,num_layers", [
    pytest.param(str(HF_MODELS / "deepseek_v3"), 4,
                 id="moe-deepseek_v3"),
    pytest.param(str(HF_MODELS / "mixtral_8x7b"), 2,
                 id="moe-mixtral_8x7b"),
])
def test_moe_components_present(model_path: str, num_layers: int, tmp_path: Path):
    """MoE models should produce moe.* component labels."""
    if not Path(model_path).is_dir():
        pytest.skip(f"Local model dir not found: {model_path}")

    _out, records = run_trace(
        model_id=model_path,
        num_layers=num_layers,
        batch_size=1,
        seq_len=64,
        output_path=tmp_path / "ops.xlsx",
    )
    components = {r["component"] for r in records}
    moe_comps = {c for c in components if c.startswith("moe.")}
    assert moe_comps, (
        f"Expected moe.* components for {Path(model_path).name}, "
        f"got: {components}")
    print(f"\n[{Path(model_path).name}] MoE components: {sorted(moe_comps)}")


def test_deepseek_v3_mla_components(tmp_path: Path):
    """DeepSeek-V3 MLA projections should appear in the trace."""
    path = str(HF_MODELS / "deepseek_v3")
    if not Path(path).is_dir():
        pytest.skip("Local deepseek_v3 dir not found")

    _out, records = run_trace(
        model_id=path,
        num_layers=4,
        batch_size=1,
        seq_len=64,
        output_path=tmp_path / "ops.xlsx",
    )
    components = {r["component"] for r in records}
    mla = {c for c in components if "q_a_proj" in c or "kv_a_proj" in c}
    assert mla, f"Expected MLA projection ops, got: {components}"


def test_layer_attribution(tmp_path: Path):
    """Every record from a block-internal module should carry a layer index."""
    path = str(HF_MODELS / "llama3_8b")
    if not Path(path).is_dir():
        pytest.skip("Local llama3_8b dir not found")

    _out, records = run_trace(
        model_id=path,
        num_layers=2,
        batch_size=1,
        seq_len=32,
        output_path=tmp_path / "ops.xlsx",
    )
    # All attention and FFN records should have a numeric layer index
    block_records = [
        r for r in records
        if r["component"].startswith(("attn.", "ffn.", "norm.pre", "norm.post"))
    ]
    assert block_records, "No block-internal records found"
    missing_layer = [r for r in block_records if r["layer"] == ""]
    assert not missing_layer, (
        f"{len(missing_layer)} block records have no layer index: "
        f"{[r['module_path'] for r in missing_layer[:5]]}")


def test_config_summary_fields(tmp_path: Path):
    """Config summary must include model_type and at least the core dims."""
    path = str(HF_MODELS / "qwen2_7b")
    if not Path(path).is_dir():
        pytest.skip("Local qwen2_7b dir not found")

    out, records = run_trace(
        model_id=path,
        num_layers=2,
        batch_size=1,
        seq_len=32,
        output_path=tmp_path / "ops.xlsx",
    )
    # Re-derive summary from loaded config for inspection
    from screenshot_ops import load_model, build_config_summary
    _, config = load_model(path, num_hidden_layers=2)
    summary = build_config_summary(path, config, 2, 1, 32)

    for required in ("model_id", "model_type", "hidden_size",
                     "num_attention_heads", "vocab_size"):
        assert required in summary, f"Missing field '{required}' in summary"


# ═════════════════════════════════════════════════════════════════════════════
# HF Hub tests — downloads config only (~few KB), NOT model weights
# Marked @network: skipped automatically when HF Hub is unreachable.
# ═════════════════════════════════════════════════════════════════════════════

@network
@pytest.mark.parametrize("model_id,num_layers,description", [
    pytest.param(
        "deepseek-ai/DeepSeek-V3-0324", 4,
        "DeepSeek-V3.2 (March 2024 update) — MLA + MoE architecture",
        id="hub-deepseek-v3.2",
    ),
    pytest.param(
        "deepseek-ai/DeepSeek-V3", 4,
        "DeepSeek-V3 base — MLA + 256 routed experts",
        id="hub-deepseek-v3",
    ),
    pytest.param(
        "Qwen/Qwen3-8B", 4,
        "Qwen3-8B dense model",
        id="hub-qwen3-8b",
    ),
    pytest.param(
        "Qwen/Qwen3-0.6B", 4,
        "Qwen3-0.6B (smallest Qwen3 dense)",
        id="hub-qwen3-0.6b",
    ),
    pytest.param(
        "Qwen/Qwen3-30B-A3B", 4,
        "Qwen3-30B-A3B MoE model",
        id="hub-qwen3-moe",
    ),
    pytest.param(
        "meta-llama/Llama-3.1-8B", 2,
        "Llama-3.1 8B — dense GQA model",
        id="hub-llama3.1-8b",
    ),
])
def test_hub_model(model_id: str, num_layers: int, description: str,
                   tmp_path: Path):
    """End-to-end trace via HF Hub config (no weights downloaded).

    Only the config.json (~few KB) is fetched; model weights are never loaded
    because we use ``from_config`` on a meta device.
    """
    print(f"\nTesting: {description}")
    try:
        out, records = run_trace(
            model_id=model_id,
            num_layers=num_layers,
            batch_size=1,
            seq_len=64,
            output_path=tmp_path / "ops.xlsx",
        )
    except Exception as exc:
        # Model might require gated access or specific HF token
        if "gated" in str(exc).lower() or "access" in str(exc).lower() \
                or "401" in str(exc) or "403" in str(exc):
            pytest.skip(f"Model requires HF authentication: {exc}")
        raise

    assert out.exists()
    _assert_trace_valid(records, model_id)

    components = sorted({r["component"] for r in records})
    print(f"  {len(records)} ops — components: {components}")


@network
@pytest.mark.parametrize("model_id,expect_moe", [
    pytest.param("deepseek-ai/DeepSeek-V3-0324", True,  id="deepseek-v3.2-moe"),
    pytest.param("Qwen/Qwen3-8B",                False, id="qwen3-8b-dense"),
    pytest.param("Qwen/Qwen3-30B-A3B",           True,  id="qwen3-moe"),
])
def test_hub_moe_detection(model_id: str, expect_moe: bool, tmp_path: Path):
    """Validate MoE component presence/absence for known hub models."""
    try:
        _out, records = run_trace(
            model_id=model_id,
            num_layers=4,
            batch_size=1,
            seq_len=32,
            output_path=tmp_path / "ops.xlsx",
        )
    except Exception as exc:
        if "gated" in str(exc).lower() or "401" in str(exc) or "403" in str(exc):
            pytest.skip(f"Gated model: {exc}")
        raise

    components = {r["component"] for r in records}
    has_moe = any(c.startswith("moe.") for c in components)

    if expect_moe:
        assert has_moe, f"{model_id}: expected MoE ops, got {sorted(components)}"
    else:
        assert not has_moe, f"{model_id}: unexpected MoE ops: {sorted(c for c in components if c.startswith('moe.'))}"
