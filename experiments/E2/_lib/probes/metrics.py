"""Probe-evaluation metrics."""

from __future__ import annotations

import numpy as np


def r2_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Coefficient of determination ``1 - SS_res / SS_tot``.

    A constant predictor returns 0; a perfect predictor returns 1; arbitrarily
    bad predictors return values < 0. ``SS_tot`` is computed against the *true*
    mean (not the prediction mean), matching ``sklearn.metrics.r2_score``.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    if y_pred.shape != y_true.shape:
        raise ValueError(f"shape mismatch: pred={y_pred.shape} true={y_true.shape}")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)
