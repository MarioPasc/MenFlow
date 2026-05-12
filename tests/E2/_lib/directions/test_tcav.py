"""Tests for matched_control_tcav."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from experiments.E2._lib.directions import (
    InsufficientMatchedControlsError,
    TCAVResult,
    matched_control_tcav,
)

# ---------------------------------------------------------------------------
# Fixture factory
# ---------------------------------------------------------------------------


def _make_confounded_fixture(
    N: int = 400,
    C: int = 8,
    seed: int = 0,
    vol_sex_shift: float = 0.8,
    conf_strength: float = 0.6,
    noise: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    """Synthetic fixture with planted volume direction + sex-correlated confound.

    Returns
    -------
    z : (N, C)
    log_v : (N,)
    covariates : DataFrame with sex, grade_bin, laterality, age
    u_vol : (C,) planted volume direction (unit)
    u_conf : (C,) planted confound direction (unit)
    """
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
    return z, log_v, covariates, u_vol, u_conf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tcav_recovers_volume_direction_better_than_naive() -> None:
    """TCAV direction is more aligned with the planted volume direction than naive ridge."""
    z, log_v, covariates, u_vol, _ = _make_confounded_fixture()

    result = matched_control_tcav(z, log_v, covariates)

    # Naive ridge direction for comparison
    ridge_coef = Ridge(alpha=1.0).fit(z, log_v).coef_
    u_naive = ridge_coef / np.linalg.norm(ridge_coef)

    cos_tcav = abs(float(np.dot(result.direction, u_vol)))
    cos_naive = abs(float(np.dot(u_naive, u_vol)))

    assert cos_tcav > cos_naive, f"Expected TCAV (cos={cos_tcav:.3f}) > naive (cos={cos_naive:.3f})"
    assert cos_tcav > 0.85, (
        f"TCAV direction poorly aligned with volume direction: cos={cos_tcav:.3f}"
    )


def test_tcav_result_n_pairs_at_least_30() -> None:
    """TCAV must match at least 30 pairs on the N=400 fixture."""
    z, log_v, covariates, _, _ = _make_confounded_fixture()
    result = matched_control_tcav(z, log_v, covariates)
    assert result.n_pairs >= 30


def test_tcav_direction_unit_norm() -> None:
    """Returned direction must be unit-norm."""
    z, log_v, covariates, _, _ = _make_confounded_fixture()
    result = matched_control_tcav(z, log_v, covariates)
    np.testing.assert_allclose(np.linalg.norm(result.direction), 1.0, atol=1e-12)


def test_tcav_roc_auc_reasonable() -> None:
    """In-sample ROC AUC should be >= 0.7 for a well-planted direction."""
    z, log_v, covariates, _, _ = _make_confounded_fixture()
    result = matched_control_tcav(z, log_v, covariates)
    assert result.roc_auc >= 0.7, f"ROC AUC too low: {result.roc_auc:.3f}"


def test_tcav_insufficient_pairs_raises() -> None:
    """With N=20 and min_pairs=30, InsufficientMatchedControlsError must be raised."""
    z, log_v, covariates, _, _ = _make_confounded_fixture(N=20)
    with pytest.raises(InsufficientMatchedControlsError):
        matched_control_tcav(z, log_v, covariates, min_pairs=30)


def test_tcav_nan_sex_rows_dropped() -> None:
    """Rows with NaN sex are dropped; n_pairs should be lower than without NaN."""
    # Use N=600 so that setting 200 rows to NaN still leaves enough valid rows
    # for matching to exceed min_pairs=30 in both configurations.
    z, log_v, cov_full, _, _ = _make_confounded_fixture(N=600)

    # Full run (no NaN)
    result_full = matched_control_tcav(z, log_v, cov_full)

    # First 200 rows have sex set to NaN (removed before matching)
    cov_nan = cov_full.copy()
    cov_nan.loc[cov_nan.index[:200], "sex"] = np.nan
    result_nan = matched_control_tcav(z, log_v, cov_nan)

    assert result_nan.n_pairs < result_full.n_pairs, (
        "Expected fewer pairs when 200/600 sex values are NaN"
    )


def test_tcav_result_is_dataclass() -> None:
    """TCAVResult must be an instance of the correct dataclass with the required fields."""
    z, log_v, covariates, _, _ = _make_confounded_fixture()
    result = matched_control_tcav(z, log_v, covariates)
    assert isinstance(result, TCAVResult)
    assert hasattr(result, "direction")
    assert hasattr(result, "norm")
    assert hasattr(result, "n_pairs")
    assert hasattr(result, "roc_auc")
    assert hasattr(result, "pairs_index")
    assert result.pairs_index.shape == (2 * result.n_pairs,)
