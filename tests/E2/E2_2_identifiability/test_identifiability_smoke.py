"""End-to-end smoke test for E2.2 direction identifiability.

Uses synthetic data (features_h5='synthetic') to verify the full pipeline
runs without errors and produces expected output artefacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.E2.E2_2_identifiability.analysis.identifiability import (
    IdentifiabilityResult,
    IdentifiabilityRoutineConfig,
    run_identifiability,
)

_VALID_REGIMES = {"stable_unconfounded", "stable_confounded", "subspace"}
_VALID_DECISIONS = {"use_naive", "use_tcav", "investigate_disagreement", "defer_e2_6"}


@pytest.fixture()
def smoke_cfg(tmp_path: Path) -> IdentifiabilityRoutineConfig:
    return IdentifiabilityRoutineConfig(
        features_h5=Path("synthetic"),
        source_h5=Path("synthetic"),
        clinical_xlsx=Path("synthetic"),
        naive_direction_npz=Path("synthetic"),
        output_dir=tmp_path,
        n_boot=20,
        large_quantile=0.7,
        small_quantile=0.3,
        age_window=10.0,
        min_pairs=5,
        seed=42,
        log_level="WARNING",
        laterality_cache_parquet=None,
    )


def test_run_returns_identifiability_result(smoke_cfg: IdentifiabilityRoutineConfig) -> None:
    result = run_identifiability(smoke_cfg)
    assert isinstance(result, IdentifiabilityResult)


def test_all_scalar_fields_populated(smoke_cfg: IdentifiabilityRoutineConfig) -> None:
    result = run_identifiability(smoke_cfg)
    assert np.isfinite(result.rho_boot_mean), "rho_boot_mean must be finite"
    assert np.isfinite(result.rho_boot_p5), "rho_boot_p5 must be finite"
    assert 0.0 <= result.rho_conf_tcav <= 1.0, "rho_conf_tcav must be in [0, 1]"
    assert 0.0 <= result.rho_conf_fwl <= 1.0, "rho_conf_fwl must be in [0, 1]"
    assert result.tcav_norm > 0.0, "tcav_norm must be positive"
    assert 0.0 <= result.tcav_roc_auc <= 1.0, "tcav_roc_auc must be in [0, 1]"
    assert result.n_matched_pairs >= 1
    assert result.n_scans_used >= 1
    assert result.n_imputed_for_fwl >= 0
    assert 0.0 <= result.fraction_unknown_grade_pairs <= 1.0
    assert np.isfinite(result.tcav_fwl_disagreement)


def test_regime_is_valid(smoke_cfg: IdentifiabilityRoutineConfig) -> None:
    result = run_identifiability(smoke_cfg)
    # "path_d" must never appear on planted-direction synthetic data
    assert result.regime in _VALID_REGIMES, f"Unexpected regime: {result.regime!r}"


def test_decision_is_valid(smoke_cfg: IdentifiabilityRoutineConfig) -> None:
    result = run_identifiability(smoke_cfg)
    assert result.decision in _VALID_DECISIONS, f"Unexpected decision: {result.decision!r}"


def test_output_files_exist(smoke_cfg: IdentifiabilityRoutineConfig) -> None:
    run_identifiability(smoke_cfg)
    out = smoke_cfg.output_dir
    assert (out / "result.json").exists(), "result.json missing"
    assert (out / "tcav_direction.npz").exists(), "tcav_direction.npz missing"
    assert (out / "fwl_direction.npz").exists(), "fwl_direction.npz missing"
    assert (out / "bootstrap_directions.npz").exists(), "bootstrap_directions.npz missing"
    assert (out / "sensitivity.json").exists(), "sensitivity.json missing"
    for png in [
        "bootstrap_cosine_hist.png",
        "svd_scree.png",
        "rho_conf_bar.png",
        "drop_one_sensitivity.png",
    ]:
        assert (out / png).exists(), f"{png} missing"


def test_result_json_is_valid(smoke_cfg: IdentifiabilityRoutineConfig) -> None:
    run_identifiability(smoke_cfg)
    with open(smoke_cfg.output_dir / "result.json") as fh:
        data = json.load(fh)
    assert "regime" in data
    assert "decision" in data
    assert "rho_boot_mean" in data
    assert "rho_conf_tcav" in data


def test_tcav_direction_is_unit_norm(smoke_cfg: IdentifiabilityRoutineConfig) -> None:
    run_identifiability(smoke_cfg)
    npz = np.load(smoke_cfg.output_dir / "tcav_direction.npz")
    direction = npz["direction"]
    assert direction.shape == (4,), f"Expected shape (4,); got {direction.shape}"
    np.testing.assert_allclose(np.linalg.norm(direction), 1.0, atol=1e-6)


def test_fwl_direction_is_unit_norm(smoke_cfg: IdentifiabilityRoutineConfig) -> None:
    run_identifiability(smoke_cfg)
    npz = np.load(smoke_cfg.output_dir / "fwl_direction.npz")
    direction = npz["direction"]
    assert direction.shape == (4,), f"Expected shape (4,); got {direction.shape}"
    np.testing.assert_allclose(np.linalg.norm(direction), 1.0, atol=1e-6)


def test_blocked_txt_written_when_blocked(smoke_cfg: IdentifiabilityRoutineConfig) -> None:
    result = run_identifiability(smoke_cfg)
    blocked_path = smoke_cfg.output_dir / "BLOCKED.txt"
    if result.blocked_carry_forward:
        assert blocked_path.exists(), "BLOCKED.txt missing but blocked_carry_forward=True"
    else:
        assert not blocked_path.exists(), "BLOCKED.txt present but blocked_carry_forward=False"


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_reproducibility(tmp_path: Path, seed: int) -> None:
    """Same seed → same rho_boot_mean across two runs."""
    cfg = IdentifiabilityRoutineConfig(
        features_h5=Path("synthetic"),
        source_h5=Path("synthetic"),
        clinical_xlsx=Path("synthetic"),
        naive_direction_npz=Path("synthetic"),
        output_dir=tmp_path / f"seed_{seed}_run1",
        n_boot=10,
        large_quantile=0.7,
        small_quantile=0.3,
        age_window=10.0,
        min_pairs=5,
        seed=seed,
        log_level="WARNING",
    )
    cfg2 = IdentifiabilityRoutineConfig(
        features_h5=Path("synthetic"),
        source_h5=Path("synthetic"),
        clinical_xlsx=Path("synthetic"),
        naive_direction_npz=Path("synthetic"),
        output_dir=tmp_path / f"seed_{seed}_run2",
        n_boot=10,
        large_quantile=0.7,
        small_quantile=0.3,
        age_window=10.0,
        min_pairs=5,
        seed=seed,
        log_level="WARNING",
    )
    r1 = run_identifiability(cfg)
    r2 = run_identifiability(cfg2)
    assert r1.rho_boot_mean == r2.rho_boot_mean
    assert r1.regime == r2.regime
