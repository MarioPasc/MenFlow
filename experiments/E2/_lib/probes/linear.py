"""Linear (ridge) probe with patient-grouped K-fold cross-validation.

Selects the regularization strength ``alpha`` over a small grid by mean
held-out R² across patient-grouped folds, then refits the chosen ridge on the
full data. Returned model exposes ``.coef_`` and ``.intercept_`` directly.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import Ridge

from experiments.E2._lib.probes.cv import patient_grouped_kfold_indices
from experiments.E2._lib.probes.metrics import r2_score

logger = logging.getLogger(__name__)


def linear_probe_cv_score(
    z: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    alpha: float,
    n_splits: int = 5,
) -> float:
    """Mean held-out R² of a Ridge probe at fixed ``alpha``."""
    scores: list[float] = []
    for tr, te in patient_grouped_kfold_indices(groups, n_splits=n_splits):
        m = Ridge(alpha=alpha).fit(z[tr], y[tr])
        scores.append(r2_score(m.predict(z[te]), y[te]))
    return float(np.mean(scores))


def fit_linear_probe(
    z: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0),
    n_splits: int = 5,
) -> tuple[Ridge, float, float]:
    """Fit a Ridge probe with CV-selected ``alpha``.

    Parameters
    ----------
    z
        Feature matrix ``(N, C)``.
    y
        Target vector ``(N,)``.
    groups
        Patient identifier per row (``(N,)``); CV splits at the patient level.
    alphas
        Grid of ridge strengths to try.
    n_splits
        Number of folds (default 5; matches E2 spec).

    Returns
    -------
    model : Ridge
        Refit on full data with the best ``alpha``.
    r2_cv : float
        Mean held-out R² at the best ``alpha`` (the value used for decisions).
    best_alpha : float
        Selected regularization strength.
    """
    if z.ndim != 2:
        raise ValueError(f"z must be (N, C); got {z.shape}")
    if y.shape[0] != z.shape[0] or groups.shape[0] != z.shape[0]:
        raise ValueError("z, y, and groups must share the leading dim")

    best_alpha = float(alphas[0])
    best_score = -np.inf
    for alpha in alphas:
        s = linear_probe_cv_score(z, y, groups, alpha=alpha, n_splits=n_splits)
        logger.debug("ridge alpha=%g -> CV R²=%.4f", alpha, s)
        if s > best_score:
            best_alpha, best_score = float(alpha), s

    model = Ridge(alpha=best_alpha).fit(z, y)
    return model, float(best_score), best_alpha
