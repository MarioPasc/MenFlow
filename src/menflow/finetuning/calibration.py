"""Test-time metrics for E3.1 volume-conditional FM finetuning.

The full Stage-1 calibration gate runs on Picasso A100. The local smoke
delivers two cheap proxies:

1. **Latent-R² proxy** — sample `n_samples` latents per test scan via the
   FM ODE with CFG, predict log-V from the noiseless latent via a linear
   regressor fit on (z_global, log_v) pairs from e3_train, regress predicted
   on target. Cheap (no decode required).
2. **Soft-volume R² on decoded T1c** — for `n_decoded` test scans, decode the
   sampled latent through the frozen MAISI VAE, count voxels above
   ``μ + 2σ`` of the baseline T1c within a dilated tumor mask, regress voxel
   count vs target log_V. Image-space confirmation; slow.

Both feed a 1000-resample bootstrap CI on the OLS slope and R².

References
----------
E3.1 §8.2 Stage-1 gate.
NV-Generate-CTMR/scripts/diff_model_infer.py:153-212 — sampling loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torchmetrics import MetricCollection
from torchmetrics.regression import R2Score, SpearmanCorrCoef

logger = logging.getLogger(__name__)


# ============================================================================
# Public dataclasses
# ============================================================================


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    """OLS slope, R², per-patient Spearman summary, bootstrap CIs."""

    n: int
    slope: float
    intercept: float
    r2: float
    spearman_mean: float
    spearman_median: float
    pct_spearman_ge_0p9: float
    bootstrap: dict[str, dict[str, float]]  # 'slope': {'lo': , 'hi': }, 'r2': {...}

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "n": int(self.n),
            "slope": float(self.slope),
            "intercept": float(self.intercept),
            "r2": float(self.r2),
            "spearman_mean": float(self.spearman_mean),
            "spearman_median": float(self.spearman_median),
            "pct_spearman_ge_0p9": float(self.pct_spearman_ge_0p9),
            "bootstrap": {
                k: {kk: float(vv) for kk, vv in v.items()} for k, v in self.bootstrap.items()
            },
        }


def per_logv_bin_metrics(
    log_v_pred: np.ndarray,
    log_v_true: np.ndarray,
    *,
    n_bins: int = 5,
    bin_edges: np.ndarray | None = None,
) -> dict[str, Any]:
    """Per-quantile-bin slope / R² / Spearman of (pred, true).

    Useful to spot whether the model overfits to a specific log-volume range.
    Each bin's R² and Spearman are reported alongside the per-bin sample
    count so downstream analysis can correlate calibration with bin density.
    """
    log_v_pred = np.asarray(log_v_pred, dtype=np.float64).ravel()
    log_v_true = np.asarray(log_v_true, dtype=np.float64).ravel()
    if log_v_pred.size != log_v_true.size:
        raise ValueError(f"shape mismatch: {log_v_pred.shape} vs {log_v_true.shape}")
    if bin_edges is None:
        try:
            edges = np.quantile(log_v_true, np.linspace(0.0, 1.0, n_bins + 1))
            for i in range(1, len(edges)):
                if edges[i] <= edges[i - 1]:
                    edges[i] = edges[i - 1] + 1e-9
            bin_edges = edges
        except Exception:
            bin_edges = np.linspace(log_v_true.min(), log_v_true.max(), n_bins + 1)
    bins: list[dict[str, Any]] = []
    last = len(bin_edges) - 2
    for i in range(len(bin_edges) - 1):
        lo, hi = float(bin_edges[i]), float(bin_edges[i + 1])
        mask = (log_v_true >= lo) & ((log_v_true <= hi) if i == last else (log_v_true < hi))
        n_in = int(mask.sum())
        if n_in < 2:
            bins.append(
                {
                    "bin": i,
                    "lo": lo,
                    "hi": hi,
                    "n": n_in,
                    "slope": float("nan"),
                    "r2": float("nan"),
                    "spearman": float("nan"),
                    "pred_mean": float("nan"),
                    "pred_std": float("nan"),
                    "true_mean": float("nan"),
                    "true_std": float("nan"),
                }
            )
            continue
        slope, _, r2 = _ols_with_r2(log_v_true[mask], log_v_pred[mask])
        try:
            sp = float(
                SpearmanCorrCoef()(
                    torch.from_numpy(log_v_pred[mask].copy()),
                    torch.from_numpy(log_v_true[mask].copy()),
                ).item()
            )
        except Exception:
            sp = float("nan")
        bins.append(
            {
                "bin": i,
                "lo": lo,
                "hi": hi,
                "n": n_in,
                "slope": float(slope),
                "r2": float(r2),
                "spearman": sp,
                "pred_mean": float(log_v_pred[mask].mean()),
                "pred_std": float(log_v_pred[mask].std()),
                "true_mean": float(log_v_true[mask].mean()),
                "true_std": float(log_v_true[mask].std()),
            }
        )
    return {
        "bin_edges": [float(x) for x in bin_edges.tolist()],
        "n_bins": int(len(bins)),
        "bins": bins,
    }


# ============================================================================
# Pure-numerics: calibration regression with bootstrap
# ============================================================================


def fit_calibration(
    log_v_pred: np.ndarray,
    log_v_true: np.ndarray,
    *,
    patient_ids: np.ndarray | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> CalibrationMetrics:
    """Fit ``log_v_pred = slope · log_v_true + intercept``; report bootstrap CIs.

    ``patient_ids`` enables per-patient Spearman aggregation. If absent, the
    Spearman summary is computed across all individual (pred, true) pairs and
    treated as one "patient" — the per-patient ratio collapses to 0.0/1.0.
    """
    log_v_pred = np.asarray(log_v_pred, dtype=np.float64).ravel()
    log_v_true = np.asarray(log_v_true, dtype=np.float64).ravel()
    n = log_v_pred.size
    if n != log_v_true.size:
        raise ValueError(f"shape mismatch: {log_v_pred.shape} vs {log_v_true.shape}")

    slope, intercept, r2 = _ols_with_r2(log_v_true, log_v_pred)
    sp_mean, sp_median, frac_ok = _per_patient_spearman(log_v_pred, log_v_true, patient_ids)
    boot = _bootstrap_cis(
        log_v_true,
        log_v_pred,
        n_bootstrap=n_bootstrap,
        seed=seed,
        patient_ids=patient_ids,
    )
    return CalibrationMetrics(
        n=n,
        slope=slope,
        intercept=intercept,
        r2=r2,
        spearman_mean=sp_mean,
        spearman_median=sp_median,
        pct_spearman_ge_0p9=frac_ok,
        bootstrap=boot,
    )


# ============================================================================
# Internals
# ============================================================================


def _ols_with_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    if x.size < 2:
        return float("nan"), float("nan"), float("nan")
    a = np.vstack([x, np.ones_like(x)]).T
    sol, *_ = np.linalg.lstsq(a, y, rcond=None)
    slope, intercept = float(sol[0]), float(sol[1])
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return slope, intercept, r2


def _per_patient_spearman(
    pred: np.ndarray,
    target: np.ndarray,
    patient_ids: np.ndarray | None,
) -> tuple[float, float, float]:
    """Mean / median / fraction ≥ 0.9 of per-patient Spearman correlations.

    Patients with fewer than 2 observations are skipped. If every patient has
    a single observation (cross-sectional cohort), the metric falls back to a
    single global Spearman over all (pred, target) pairs.
    """
    sp_metric = SpearmanCorrCoef()
    if patient_ids is None or patient_ids.size != pred.size:
        if pred.size < 2:
            return float("nan"), float("nan"), 0.0
        v = sp_metric(torch.from_numpy(pred), torch.from_numpy(target)).item()
        return v, v, float(v >= 0.9)
    rhos: list[float] = []
    for pid in np.unique(patient_ids):
        mask = patient_ids == pid
        if mask.sum() < 2:
            continue
        sp_metric.reset()
        rho = sp_metric(
            torch.from_numpy(pred[mask].copy()), torch.from_numpy(target[mask].copy())
        ).item()
        rhos.append(float(rho))
    if not rhos:
        # Cross-sectional fallback: treat the whole cohort as one ordered
        # comparison instead of returning NaN.
        if pred.size < 2:
            return float("nan"), float("nan"), 0.0
        v = sp_metric(torch.from_numpy(pred), torch.from_numpy(target)).item()
        return float(v), float(v), float(v >= 0.9)
    arr = np.asarray(rhos)
    return float(arr.mean()), float(np.median(arr)), float((arr >= 0.9).mean())


def _bootstrap_cis(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    patient_ids: np.ndarray | None,
) -> dict[str, dict[str, float]]:
    """Patient-level bootstrap of slope and R²; returns {'slope': {'lo','hi'}, ...}.

    Falls back to sample-level resampling when patient_ids is absent.
    """
    rng = np.random.default_rng(seed)
    if patient_ids is None or patient_ids.size != x.size:
        n = x.size
        idx_pool = np.arange(n)
        slopes, r2s = [], []
        for _ in range(n_bootstrap):
            idx = rng.choice(idx_pool, size=n, replace=True)
            s, _, r2 = _ols_with_r2(x[idx], y[idx])
            slopes.append(s)
            r2s.append(r2)
    else:
        unique_pids = np.unique(patient_ids)
        slopes, r2s = [], []
        for _ in range(n_bootstrap):
            sampled = rng.choice(unique_pids, size=len(unique_pids), replace=True)
            mask = np.concatenate([np.where(patient_ids == p)[0] for p in sampled])
            s, _, r2 = _ols_with_r2(x[mask], y[mask])
            slopes.append(s)
            r2s.append(r2)
    slopes = np.asarray(slopes)
    r2s = np.asarray(r2s)
    return {
        "slope": {
            "lo": float(np.nanpercentile(slopes, 2.5)),
            "hi": float(np.nanpercentile(slopes, 97.5)),
        },
        "r2": {"lo": float(np.nanpercentile(r2s, 2.5)), "hi": float(np.nanpercentile(r2s, 97.5))},
    }


# ============================================================================
# Streaming val-time metrics (used in the per-step validation loop)
# ============================================================================


def make_val_metric_collection() -> MetricCollection:
    """torchmetrics collection for streaming `val_loss`-side R² + Spearman."""
    return MetricCollection(
        {
            "r2": R2Score(),
            "spearman": SpearmanCorrCoef(),
        }
    )
