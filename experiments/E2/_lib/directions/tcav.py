"""Matched-control TCAV direction for latent-space tumor-volume analysis.

The TCAV direction is a logistic-regression classifier trained to distinguish
large-tumour scans from matched small-tumour controls, with covariate matching
on sex, histological grade, laterality, and age. This yields a direction in
latent space that reflects volume variation while holding confounders fixed.

Reference:
    Kim et al., "Interpretability Beyond Classification", ICML 2018.
    https://proceedings.mlr.press/v80/kim18d.html
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class InsufficientMatchedControlsError(RuntimeError):
    """Raised when too few large-small pairs can be matched on covariates."""


@dataclass(frozen=True, slots=True)
class TCAVResult:
    """Result of a matched-control TCAV fit.

    Attributes
    ----------
    direction:
        Unit vector in latent space aligned with tumor volume, shape ``(C,)``.
    norm:
        Euclidean norm of the raw logistic coefficient *before* normalization.
    n_pairs:
        Number of matched large/small scan pairs used to train the classifier.
    roc_auc:
        In-sample ROC AUC of the logistic classifier on the matched set.
        Reported for diagnostic purposes (not generalization); a value near 1.0
        confirms the direction separates the two groups well.
    pairs_index:
        Integer indices into the *original* ``z_tumor`` array of shape
        ``(2 * n_pairs,)``, interleaved as
        ``[idx_L0, idx_S0, idx_L1, idx_S1, ...]``.
    """

    direction: np.ndarray
    norm: float
    n_pairs: int
    roc_auc: float
    pairs_index: np.ndarray


def matched_control_tcav(
    z_tumor: np.ndarray,
    log_v: np.ndarray,
    covariates: pd.DataFrame,
    *,
    large_quantile: float = 0.8,
    small_quantile: float = 0.2,
    age_window: float = 5.0,
    min_pairs: int = 30,
    seed: int = 0,
) -> TCAVResult:
    """Compute a TCAV direction via greedy 1-1 covariate-matched controls.

    Parameters
    ----------
    z_tumor:
        Latent feature matrix ``(N, C)`` for all scans.
    log_v:
        Log-volume vector ``(N,)`` in consistent units (e.g. log cm³).
    covariates:
        DataFrame with RangeIndex 0..N-1 and columns
        ``sex`` (str), ``grade_bin`` (str), ``laterality`` (str),
        ``age`` (float). Rows with NaN in any of these columns are dropped
        before quantile-binning.
    large_quantile:
        Upper volume quantile threshold; scans at or above this percentile of
        log-volume form the *large* group.
    small_quantile:
        Lower volume quantile threshold; scans at or below form the *small*
        group.
    age_window:
        Maximum absolute age difference (years) between a matched pair.
    min_pairs:
        Minimum required number of matched pairs; raises
        :class:`InsufficientMatchedControlsError` if not met.
    seed:
        Random seed (currently unused; matching is deterministic but the
        parameter is kept for API stability).

    Returns
    -------
    TCAVResult

    Raises
    ------
    InsufficientMatchedControlsError
        If fewer than ``min_pairs`` pairs can be matched.

    Notes
    -----
    The ROC AUC is computed on the training set (in-sample). Because the
    matched-control design deliberately limits N, no held-out set is used.
    Treat the AUC as a diagnostic of direction quality, not a generalization
    estimate.
    """
    z_tumor = np.asarray(z_tumor, dtype=np.float64)
    log_v = np.asarray(log_v, dtype=np.float64)
    if z_tumor.ndim != 2:
        raise ValueError(f"z_tumor must be (N, C); got {z_tumor.shape}")
    n_total = z_tumor.shape[0]

    # --- 1. Drop rows with NaN in any matching covariate ---
    required_cols = ["sex", "grade_bin", "laterality", "age"]
    mask_valid = covariates[required_cols].notna().all(axis=1).to_numpy()
    n_dropped = int(n_total - mask_valid.sum())
    if n_dropped > 0:
        logger.info(
            "matched_control_tcav: dropping %d / %d rows with NaN covariates",
            n_dropped,
            n_total,
        )
    orig_indices = np.where(mask_valid)[0]  # original indices into z_tumor
    cov_kept = covariates.iloc[orig_indices].reset_index(drop=True)
    log_v_kept = log_v[orig_indices]

    # --- 2. Quantile thresholds on the kept set ---
    v_lo, v_hi = np.quantile(log_v_kept, [small_quantile, large_quantile])
    large_mask = log_v_kept >= v_hi
    small_mask = log_v_kept <= v_lo

    large_local_idx = np.where(large_mask)[0]
    small_local_idx = np.where(small_mask)[0]
    logger.debug(
        "TCAV: %d large (>= %.3f), %d small (<= %.3f)",
        len(large_local_idx),
        v_hi,
        len(small_local_idx),
        v_lo,
    )

    # --- 3. Greedy 1-1 matching (no replacement) ---
    small_available = set(small_local_idx.tolist())
    pairs_local: list[tuple[int, int]] = []  # (large_local, small_local)

    for li in large_local_idx:
        row_l = cov_kept.iloc[li]
        key_l = (row_l["sex"], row_l["grade_bin"], row_l["laterality"])
        age_l = float(row_l["age"])

        best_si: int | None = None
        best_age_diff = np.inf
        for si in small_available:
            row_s = cov_kept.iloc[si]
            key_s = (row_s["sex"], row_s["grade_bin"], row_s["laterality"])
            if key_s != key_l:
                continue
            age_diff = abs(float(row_s["age"]) - age_l)
            if age_diff > age_window:
                continue
            if age_diff < best_age_diff:
                best_age_diff = age_diff
                best_si = si

        if best_si is not None:
            pairs_local.append((li, best_si))
            small_available.discard(best_si)

    n_pairs = len(pairs_local)
    logger.info("TCAV: matched %d pairs", n_pairs)

    if n_pairs < min_pairs:
        raise InsufficientMatchedControlsError(f"only {n_pairs} matched pairs; need >= {min_pairs}")

    # --- 4. Build training arrays ---
    large_local = np.array([p[0] for p in pairs_local], dtype=np.intp)
    small_local = np.array([p[1] for p in pairs_local], dtype=np.intp)

    X = np.concatenate(
        [z_tumor[orig_indices[large_local]], z_tumor[orig_indices[small_local]]], axis=0
    )
    y = np.concatenate([np.ones(n_pairs, dtype=np.int32), np.zeros(n_pairs, dtype=np.int32)])

    # --- 5. Fit logistic regression ---
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(X, y)

    raw_coef = clf.coef_.ravel().astype(np.float64)  # (C,)
    norm = float(np.linalg.norm(raw_coef))
    direction = raw_coef / norm if norm > 0.0 else raw_coef

    roc_auc = float(roc_auc_score(y, clf.decision_function(X)))

    # --- 6. Build pairs_index (indices into ORIGINAL z_tumor) ---
    pairs_index = np.empty(2 * n_pairs, dtype=np.intp)
    pairs_index[0::2] = orig_indices[large_local]
    pairs_index[1::2] = orig_indices[small_local]

    return TCAVResult(
        direction=direction,
        norm=norm,
        n_pairs=n_pairs,
        roc_auc=roc_auc,
        pairs_index=pairs_index,
    )
