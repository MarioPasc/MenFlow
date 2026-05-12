"""Frisch-Waugh-Lovell (FWL) residualization for confounder-robust directions.

FWL theorem: the OLS coefficient on X in Y ~ X + W equals the OLS coefficient
in M_W Y ~ M_W X, where M_W is the annihilator of W. We apply this using
ridge regression (not OLS) to be robust in high-dimensional latent spaces.

The routine residualizes each latent dimension (and log-volume) against
demographic covariates independently, then fits a new linear probe on the
residuals. The resulting direction is interpretable as the volume-related axis
*net of* confounders.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from experiments.E2._lib.probes.cv import patient_grouped_kfold_indices
from experiments.E2._lib.probes.linear import fit_linear_probe

logger = logging.getLogger(__name__)

_CAT_COLS = ["sex", "grade_bin", "laterality"]
_NUM_COLS = ["age"]


def _build_covariate_matrix(covariates: pd.DataFrame) -> np.ndarray:
    """Encode covariates into a dense design matrix.

    Parameters
    ----------
    covariates:
        DataFrame with columns ``sex``, ``grade_bin``, ``laterality`` (str)
        and ``age`` (float). Callers are responsible for imputing NaN before
        calling this function.

    Returns
    -------
    np.ndarray
        Dense float64 design matrix, shape ``(N, K)`` where K depends on
        cardinality of categorical columns.
    """
    ct = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                _CAT_COLS,
            ),
            ("num", StandardScaler(), _NUM_COLS),
        ],
        remainder="drop",
    )
    return ct.fit_transform(covariates).astype(np.float64)


def _ridge_residualize_vector(
    W: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    alphas: tuple[float, ...],
) -> np.ndarray:
    """Residualize a 1-D target against the covariate matrix W.

    Alpha is selected by patient-grouped 5-fold CV maximizing held-out R².

    Parameters
    ----------
    W:
        Covariate design matrix ``(N, K)``.
    target:
        1-D target vector ``(N,)``.
    groups:
        Patient IDs ``(N,)`` for grouped CV.
    alphas:
        Regularization grid.

    Returns
    -------
    np.ndarray
        Residual vector ``target - W @ beta_hat``, shape ``(N,)``.
    """
    best_alpha = float(alphas[0])
    best_score = -np.inf
    for alpha in alphas:
        scores: list[float] = []
        for tr, te in patient_grouped_kfold_indices(groups, n_splits=5):
            m = Ridge(alpha=alpha).fit(W[tr], target[tr])
            pred = m.predict(W[te])
            ss_res = float(np.sum((target[te] - pred) ** 2))
            ss_tot = float(np.sum((target[te] - target[te].mean()) ** 2))
            scores.append(1.0 - ss_res / max(ss_tot, 1e-12))
        mean_r2 = float(np.mean(scores))
        if mean_r2 > best_score:
            best_score = mean_r2
            best_alpha = float(alpha)

    model = Ridge(alpha=best_alpha).fit(W, target)
    return target - model.predict(W)


def fwl_residualize(
    z: np.ndarray,
    log_v: np.ndarray,
    covariates: pd.DataFrame,
    groups: np.ndarray,
    *,
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Residualize latent features and log-volume against demographic covariates.

    For each latent dimension independently, fit Ridge (alpha chosen by
    patient-grouped CV) on the covariate design matrix and subtract the
    prediction. Apply the same residualization to ``log_v``.

    Parameters
    ----------
    z:
        Latent feature matrix ``(N, C)``.
    log_v:
        Log-volume vector ``(N,)``.
    covariates:
        DataFrame indexed 0..N-1 with columns ``sex``, ``grade_bin``,
        ``laterality`` (str) and ``age`` (float). NaN values should have
        been imputed before calling this function; unknown categories are
        handled by ``handle_unknown="ignore"`` in the encoder.
    groups:
        Patient identifiers ``(N,)`` used for grouped CV splits.
    alphas:
        Regularization grid passed to Ridge.

    Returns
    -------
    z_resid : np.ndarray
        Covariate-residualized latent matrix ``(N, C)``.
    y_resid : np.ndarray
        Covariate-residualized log-volume vector ``(N,)``.
    """
    z = np.asarray(z, dtype=np.float64)
    log_v = np.asarray(log_v, dtype=np.float64).ravel()
    if z.ndim != 2:
        raise ValueError(f"z must be (N, C); got {z.shape}")

    W = _build_covariate_matrix(covariates)
    n_dims = z.shape[1]
    z_resid = np.empty_like(z)

    for c in range(n_dims):
        logger.debug("FWL residualizing latent dim %d / %d", c + 1, n_dims)
        z_resid[:, c] = _ridge_residualize_vector(W, z[:, c], groups, alphas)

    y_resid = _ridge_residualize_vector(W, log_v, groups, alphas)
    logger.info(
        "fwl_residualize: done. z_resid var=%.4f, y_resid var=%.4f",
        float(z_resid.var()),
        float(y_resid.var()),
    )
    return z_resid, y_resid


def fwl_direction(
    z_resid: np.ndarray,
    y_resid: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Fit a linear probe on FWL residuals and return the unit direction.

    Parameters
    ----------
    z_resid:
        Residualized latent matrix ``(N, C)`` from :func:`fwl_residualize`.
    y_resid:
        Residualized log-volume ``(N,)`` from :func:`fwl_residualize`.
    groups:
        Patient IDs ``(N,)`` for grouped CV in :func:`~experiments.E2._lib.probes.linear.fit_linear_probe`.

    Returns
    -------
    np.ndarray
        Unit vector in latent space, shape ``(C,)``.
    """
    model, cv_r2, best_alpha = fit_linear_probe(z_resid, y_resid, groups)
    logger.info("fwl_direction: CV R²=%.4f at alpha=%g", cv_r2, best_alpha)
    coef = model.coef_.ravel().astype(np.float64)
    norm = float(np.linalg.norm(coef))
    if norm == 0.0:
        raise ValueError("FWL probe coefficient is zero; cannot form a unit direction")
    return coef / norm
