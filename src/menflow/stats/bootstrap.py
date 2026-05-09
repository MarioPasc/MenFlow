"""Patient-level bootstrap confidence intervals and direction stability.

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

from menflow.probes.linear import fit_linear_probe
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


def bootstrap_directions(
    z: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int = 200,
    seed: int = 0,
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0),
) -> np.ndarray:
    """Bootstrap patient-level resamples and return a matrix of unit directions.

    Each replicate resamples patients with replacement (mirroring
    :func:`patient_level_bootstrap_r2`), fits a Ridge probe via
    :func:`~menflow.probes.linear.fit_linear_probe`, normalizes the
    coefficient to a unit vector, and sign-aligns to the first replicate so
    that pairwise cosines are interpretable (without alignment, antiparallel
    directions can collapse the mean cosine to zero).

    Parameters
    ----------
    z:
        Feature matrix ``(N, C)``.
    y:
        Target vector ``(N,)``.
    groups:
        Patient IDs ``(N,)`` for patient-level CV and resampling.
    n_boot:
        Number of bootstrap replicates.
    seed:
        RNG seed for reproducibility.
    alphas:
        Regularization grid passed to :func:`~menflow.probes.linear.fit_linear_probe`.

    Returns
    -------
    np.ndarray
        Matrix of unit direction vectors, shape ``(n_boot, C)``.  Each row is
        sign-aligned to ``W[0]`` (the first valid replicate).
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)
    unique_pts = np.unique(groups)
    rows_by_patient: dict[object, np.ndarray] = {p: np.where(groups == p)[0] for p in unique_pts}

    n_dims = z.shape[1]
    W = np.empty((n_boot, n_dims), dtype=np.float64)
    reference: np.ndarray | None = None

    valid_count = 0
    for b in range(n_boot):
        sampled_patients = rng.choice(unique_pts, size=len(unique_pts), replace=True)
        idx = np.concatenate([rows_by_patient[p] for p in sampled_patients])
        z_b, y_b, g_b = z[idx], y[idx], groups[idx]
        try:
            model, _, _ = fit_linear_probe(z_b, y_b, g_b, alphas=alphas)
            coef = model.coef_.ravel().astype(np.float64)
            norm = float(np.linalg.norm(coef))
            if norm == 0.0:
                logger.warning("bootstrap_directions: replicate %d has zero-norm coef; skipping", b)
                W[b] = np.nan
                continue
            w = coef / norm
            # Sign-align to the first valid direction
            if reference is None:
                reference = w.copy()
            elif float(np.dot(w, reference)) < 0.0:
                w = -w
            W[b] = w
            valid_count += 1
        except Exception as exc:  # noqa: BLE001 — bootstrap robustness
            logger.warning("bootstrap_directions: replicate %d failed: %s", b, exc)
            W[b] = np.nan

    n_nan = int(np.isnan(W).any(axis=1).sum())
    if n_nan > 0:
        logger.warning(
            "bootstrap_directions: %d / %d replicates failed or had zero norm", n_nan, n_boot
        )

    return W


def pairwise_cosine_stats(W: np.ndarray) -> tuple[float, float]:
    """Compute mean and 5th-percentile of all pairwise cosines in a direction matrix.

    Parameters
    ----------
    W:
        Matrix of unit direction vectors, shape ``(n_boot, C)``.  Rows
        containing NaN are excluded before computation.

    Returns
    -------
    mean_cos : float
        Mean of all ``n*(n-1)/2`` unique pairwise cosines.
    p5_cos : float
        5th percentile of those cosines (lower bound on directional stability).
    """
    W = np.asarray(W, dtype=np.float64)
    # Drop NaN rows
    valid_mask = ~np.isnan(W).any(axis=1)
    W_valid = W[valid_mask]
    n = W_valid.shape[0]
    if n < 2:
        raise ValueError(f"Need at least 2 valid directions; got {n}")

    # Gram matrix — entries are cosines because rows are unit vectors
    G = W_valid @ W_valid.T  # (n, n)
    # Extract upper triangle (i < j)
    upper_i, upper_j = np.triu_indices(n, k=1)
    cosines = G[upper_i, upper_j]

    mean_cos = float(np.mean(cosines))
    p5_cos = float(np.percentile(cosines, 5))
    return mean_cos, p5_cos
