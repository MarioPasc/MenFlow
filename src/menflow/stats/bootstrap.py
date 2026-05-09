"""Patient-level bootstrap confidence intervals.

A bootstrap that resamples patients (not scans) is the right level when
multiple scans of the same patient are correlated. For E2.1 BraTS-MEN-Preop
nearly every patient has one scan, but the protocol still uses patient-level
resampling defensively so the same code applies to MenGrowth (E2.6) where each
patient has 2-6 longitudinal scans.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from menflow.probes.metrics import r2_score

logger = logging.getLogger(__name__)


def patient_level_bootstrap_r2(
    z: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    fit_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], object],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    ci: tuple[float, float] = (2.5, 97.5),
) -> tuple[float, float, np.ndarray]:
    """Patient-level bootstrap CI for the R² of a probe.

    Parameters
    ----------
    z, y, groups
        Feature matrix, target, and patient ids (length N).
    fit_fn
        Callable ``(z, y, groups) -> model``; ``model.predict(z)`` must return
        a 1-D array of length N. ``fit_linear_probe`` from
        :mod:`menflow.probes.linear` is wrapped externally to match this.
    n_boot
        Number of bootstrap replicates (default 1000).
    seed
        RNG seed for reproducibility.
    ci
        Percentiles for the lower/upper CI bounds.

    Returns
    -------
    ci_lo, ci_hi : float
        Bootstrap percentile CI on R².
    scores : np.ndarray
        All ``n_boot`` bootstrap R² values (in-bag scores).
    """
    rng = np.random.default_rng(seed)
    unique_pts = np.unique(groups)
    rows_by_patient: dict[object, np.ndarray] = {p: np.where(groups == p)[0] for p in unique_pts}

    scores = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sampled_patients = rng.choice(unique_pts, size=len(unique_pts), replace=True)
        idx = np.concatenate([rows_by_patient[p] for p in sampled_patients])
        z_b, y_b, g_b = z[idx], y[idx], groups[idx]
        try:
            model = fit_fn(z_b, y_b, g_b)
            scores[b] = r2_score(model.predict(z_b), y_b)
        except Exception as exc:  # noqa: BLE001 — bootstrap robustness
            logger.warning("bootstrap fit %d failed: %s", b, exc)
            scores[b] = np.nan

    valid = scores[~np.isnan(scores)]
    ci_lo = float(np.percentile(valid, ci[0]))
    ci_hi = float(np.percentile(valid, ci[1]))
    return ci_lo, ci_hi, scores
