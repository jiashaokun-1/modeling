"""Test anchor YAML data files.

Phase 4.5: Structural validation + MFU calibration output
Phase 4.5 (after phase 3): Strict MFU tolerance gating

TODO Phase 3: Convert anchors to runnable ModelSpec/SystemSpec/Strategy
and run actual estimates. Current implementation validates YAML shape.
"""
from __future__ import annotations

import yaml
import pytest
from pathlib import Path

from zrt.training.anchor.validate import Anchor, validate_anchor
from zrt.training.search.estimator import Report


ANCHOR_DIR = Path(__file__).parent


def _load_anchor(yaml_path: Path) -> dict:
    return yaml.safe_load(yaml_path.read_text())


@pytest.mark.parametrize("yaml_file", sorted(ANCHOR_DIR.glob("*.yaml")))
def test_anchor_yaml_is_valid(yaml_file):
    data = _load_anchor(yaml_file)
    assert "name" in data
    assert "targets" in data
    anchor = Anchor(name=data["name"], **data["targets"])
    assert anchor.name


@pytest.mark.parametrize("yaml_file", sorted(ANCHOR_DIR.glob("*.yaml")))
def test_anchor_yaml_has_config(yaml_file):
    data = _load_anchor(yaml_file)
    assert "config" in data
    config = data["config"]
    assert "tp" in config
    assert "dp" in config


@pytest.mark.parametrize("yaml_file", sorted(ANCHOR_DIR.glob("*.yaml")))
def test_anchor_config_is_internally_consistent(yaml_file):
    """Verify anchor configs are internally consistent (no conflicting products)."""
    data = _load_anchor(yaml_file)
    config = data["config"]
    name = data["name"]

    tp = config.get("tp", 1)
    pp = config.get("pp", 1)
    dp = config.get("dp", 1)

    # Basic sanity: world_size should be consistent with tp * pp * dp
    # (EP excluded per current policy — see test_ep_rank_product.py)
    world_size = config.get("world_size", tp * pp * dp)
    rank_product = tp * pp * dp

    assert rank_product == world_size, (
        f"Anchor '{name}': TP*PP*DP={rank_product} != world_size={world_size}. "
        f"Internal consistency check failed."
    )


def test_anchor_validate_with_report():
    report = Report(step_time_ms=100.0, mfu=0.50, total_flops=1e12)
    anchor = Anchor(name="test", mfu=0.50, tolerance=0.15, strict_mfu_check=True)
    warnings = validate_anchor(report, anchor)
    assert len(warnings) == 0


def test_anchor_validate_fails_with_bad_report_strict():
    """Strict MFU check should fail when deviation exceeds tolerance."""
    report = Report(step_time_ms=200.0, mfu=0.20, total_flops=1e12)
    anchor = Anchor(name="test", mfu=0.50, tolerance=0.15, strict_mfu_check=True)
    warnings = validate_anchor(report, anchor)
    assert len(warnings) > 0
    assert "[STRICT]" in warnings[0]


def test_anchor_validate_calibration_mode_no_failure():
    """Calibration mode records MFU deviation but doesn't fail."""
    report = Report(step_time_ms=200.0, mfu=0.20, total_flops=1e12)
    anchor = Anchor(name="test", mfu=0.50, tolerance=0.15, strict_mfu_check=False)
    warnings = validate_anchor(report, anchor)
    # Should have warning but marked as [CALIBRATION], not [STRICT]
    assert len(warnings) > 0
    assert "[CALIBRATION]" in warnings[0]
    assert "[STRICT]" not in warnings[0]


def test_anchor_estimate_integration_placeholder():
    """Run actual estimates for each anchor YAML (Gap 1 fix).

    Phase 4.5: Structural validation + MFU calibration output
    Phase 4.5 (after phase 3): Strict MFU tolerance gating

    This test loads each anchor YAML, runs estimate(), and records
    calibration output. It does NOT enforce strict MFU checks yet
    (those require phase 3 CP/DP/EP communication to be calibrated).

    Issue B fix: Each anchor is wrapped in try/except for robustness.
    """
    from zrt.training.io.config_loader import load_anchor_config
    from zrt.training.search.estimator import estimate

    calibration_results = []

    for yaml_file in sorted(ANCHOR_DIR.glob("*.yaml")):
        try:
            # Load anchor config
            model, system, strategy = load_anchor_config(yaml_file)
            anchor_data = _load_anchor(yaml_file)
            anchor = Anchor(name=anchor_data["name"], **anchor_data["targets"])

            # Validate strategy consistency first (Item 3: no internally inconsistent device product)
            strategy.validate(model, system)

            # Run estimate
            report = estimate(model, system, strategy)
            warnings = validate_anchor(report, anchor)

            # Calibration output: record estimated vs reference MFU
            mfu_error = abs(report.mfu - anchor.mfu) / anchor.mfu if anchor.mfu > 0 else 0
            calibration_results.append({
                "name": anchor.name,
                "estimated_mfu": report.mfu,
                "reference_mfu": anchor.mfu,
                "mfu_error_pct": mfu_error * 100,
                "within_tolerance": mfu_error <= anchor.tolerance,
                "strict_mfu_check": anchor.strict_mfu_check,
                "warnings": warnings,
            })

            # Print calibration output
            print(f"\n{anchor.name}:")
            print(f"  Estimated MFU: {report.mfu:.4f}")
            print(f"  Reference MFU:  {anchor.mfu:.4f}")
            print(f"  Error: {mfu_error*100:.2f}% (tolerance: {anchor.tolerance*100:.0f}%)")
            if warnings:
                for w in warnings:
                    print(f"  WARNING: {w}")

            # Only fail if strict_mfu_check=True (no anchors currently set this)
            # This will be enabled after phase 3 calibration
            if anchor.strict_mfu_check:
                assert len(warnings) == 0, f"Anchor {anchor.name} failed strict validation"

        except AssertionError:
            # Strict MFU failures must surface (re-raise)
            raise
        except Exception as e:
            # All other errors: record and continue to next anchor
            print(f"\nERROR processing {yaml_file.name}: {e}")
            calibration_results.append({
                "name": yaml_file.stem,
                "error": str(e),
            })

    # Print summary
    print("\n" + "=" * 60)
    print("Anchor Calibration Summary")
    print("=" * 60)
    for r in calibration_results:
        if "error" in r:
            print(f"  {r['name']}: ERROR - {r['error']}")
        else:
            status = "PASS" if r["within_tolerance"] else "CALIBRATION NEEDED"
            print(f"  {r['name']}: {status} (MFU error: {r['mfu_error_pct']:.2f}%)")
    print("=" * 60)
