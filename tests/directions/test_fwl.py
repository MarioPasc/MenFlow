"""Tests for fwl_residualize and fwl_direction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge

from menflow.directions import fwl_direction, fwl_residualize

# ---------------------------------------------------------------------------
# Fixture factory (same confounded setup as test_tcav.py)
# ---------------------------------------------------------------------------


def _make_confounded_fixture(
    N: int = 400,
    C: int = 8,
    seed: int = 0,
    vol_sex_shift: float = 0.8,
    conf_strength: float = 0.6,
    noise: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u_vol = rng.standard_normal(C)
    u_vol /= np.linalg.norm(u_vol)
    u_conf = rng.standard_normal(C)
    u_conf /= np.linalg.norm(u_conf)

    sex_arr = rng.choice(["Male", "Female"], size=N)
    sex_eff = np.where(sex_arr == "Male", 1.0, -1.0)

    log_v = rng.standard_normal(N) + vol_sex_shift * (sex_arr == "Male").astype(float)
    z = (
        log_v[:, None] * u_vol
        + conf_strength * sex_eff[:, None] * u_conf
        + noise * rng.standard_normal((N, C))
    )

    covariates = pd.DataFrame(
        {
            "sex": sex_arr,
            "grade_bin": ["1"] * N,
            "laterality": ["L"] * N,
            "age": rng.uniform(40, 80, N),
        }
    )
    groups = np.arange(N).astype(str)
    return z, log_v, covariates, groups, u_vol, u_conf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fwl_direction_recovers_volume_better_than_naive() -> None:
    """FWL direction aligns better with planted volume direction than naive ridge."""
    z, log_v, covariates, groups, u_vol, _ = _make_confounded_fixture()

    z_resid, y_resid = fwl_residualize(z, log_v, covariates, groups)
    u_fwl = fwl_direction(z_resid, y_resid, groups)

    # Naive direction from full (un-residualized) ridge
    ridge_coef = Ridge(alpha=1.0).fit(z, log_v).coef_
    u_naive = ridge_coef / np.linalg.norm(ridge_coef)

    cos_fwl = abs(float(np.dot(u_fwl, u_vol)))
    cos_naive = abs(float(np.dot(u_naive, u_vol)))

    assert cos_fwl > cos_naive, f"Expected FWL (cos={cos_fwl:.3f}) > naive (cos={cos_naive:.3f})"
    assert cos_fwl > 0.80, f"FWL direction poorly aligned with volume direction: cos={cos_fwl:.3f}"


def test_fwl_residualize_output_shapes() -> None:
    """z_resid and y_resid must match input shapes."""
    z, log_v, covariates, groups, _, _ = _make_confounded_fixture()
    z_resid, y_resid = fwl_residualize(z, log_v, covariates, groups)
    assert z_resid.shape == z.shape
    assert y_resid.shape == log_v.shape


def test_fwl_residualize_removes_sex_correlation() -> None:
    """After FWL residualization, correlation between sex (one-hot) and each z dim is near zero."""
    z, log_v, covariates, groups, _, _ = _make_confounded_fixture()
    z_resid, _ = fwl_residualize(z, log_v, covariates, groups)

    # Encode sex as 0/1
    sex_bin = (covariates["sex"] == "Male").astype(float).to_numpy()

    for c in range(z_resid.shape[1]):
        r, _ = pearsonr(sex_bin, z_resid[:, c])
        assert abs(r) < 0.15, f"Residual dim {c} still correlated with sex: r={r:.3f}"


def test_fwl_direction_unit_norm() -> None:
    """fwl_direction must return a unit-norm vector."""
    z, log_v, covariates, groups, _, _ = _make_confounded_fixture()
    z_resid, y_resid = fwl_residualize(z, log_v, covariates, groups)
    u_fwl = fwl_direction(z_resid, y_resid, groups)
    np.testing.assert_allclose(np.linalg.norm(u_fwl), 1.0, atol=1e-12)


def test_fwl_direction_shape() -> None:
    """fwl_direction must return a (C,) vector."""
    z, log_v, covariates, groups, _, _ = _make_confounded_fixture(C=8)
    z_resid, y_resid = fwl_residualize(z, log_v, covariates, groups)
    u_fwl = fwl_direction(z_resid, y_resid, groups)
    assert u_fwl.shape == (8,)


def test_fwl_residualize_invalid_z_shape_raises() -> None:
    """fwl_residualize must raise ValueError for non-2D z."""
    rng = np.random.default_rng(0)
    z_bad = rng.standard_normal(100)  # 1-D
    log_v = rng.standard_normal(100)
    covariates = pd.DataFrame(
        {
            "sex": ["Male"] * 100,
            "grade_bin": ["1"] * 100,
            "laterality": ["L"] * 100,
            "age": np.ones(100),
        }
    )
    groups = np.arange(100).astype(str)
    with pytest.raises(ValueError, match="N, C"):
        fwl_residualize(z_bad, log_v, covariates, groups)
