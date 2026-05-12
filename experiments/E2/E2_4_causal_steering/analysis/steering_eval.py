"""E2.4 — slope, R², monotonicity, and saturation analysis.

Tolerates Phase A inputs (``decoded_log_v`` all-NaN) by returning a placeholder
result with ``segmenter_completed=False``. After Phase B fills the H5, the
exact same call returns the real :class:`CausalSteeringResult`.

Decision criteria (E2.4 §4.4 / §5):

* slope :math:`\\hat\\beta_1 \\in [0.7, 1.3]`
* :math:`R^2 \\geq 0.6`
* per-anchor Spearman :math:`\\rho \\geq 0.9` for :math:`\\geq 80\\%` of anchors
* drift max :math:`< 0.15`
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CausalSteeringResult:
    """Aggregated metrics for the steering sweep.

    Sentinel values when Phase B has not run:
    ``segmenter_completed=False`` and all post-segmentation fields are NaN.
    """

    segmenter_completed: bool
    n_anchors: int
    n_deltas: int
    slope: float
    slope_ci_lo: float
    slope_ci_hi: float
    intercept: float
    r2: float
    pct_monotone: float
    drift_max: float
    drift_mean: float
    saturation_log_v: float | None
    n_anatomical_pass: int
    n_anatomical_total: int
    decision: str  # "PHASE_A_ONLY" | "PASS" | "FAIL_SLOPE" | "FAIL_R2" | ...


def evaluate_sweep(
    sweep_h5_path: Path,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> CausalSteeringResult:
    """Compute all steering metrics from a sweep_results.h5.

    Returns a Phase-A placeholder if ``decoded_log_v`` is entirely non-finite.
    """
    sweep_h5_path = Path(sweep_h5_path)
    with h5py.File(sweep_h5_path, "r") as f:
        completed = bool(f.attrs.get("segmenter_completed", False))
        predicted = f["predicted_log_v"][:].astype(np.float64)
        decoded = f["decoded_log_v"][:].astype(np.float64)
        drift = f["drift"][:].astype(np.float64)

    a, n_d = predicted.shape
    finite_drift = drift[np.isfinite(drift)]
    drift_max = float(finite_drift.max()) if finite_drift.size else float("nan")
    drift_mean = float(finite_drift.mean()) if finite_drift.size else float("nan")

    if not completed or not np.any(np.isfinite(decoded)):
        return CausalSteeringResult(
            segmenter_completed=False,
            n_anchors=int(a),
            n_deltas=int(n_d),
            slope=float("nan"),
            slope_ci_lo=float("nan"),
            slope_ci_hi=float("nan"),
            intercept=float("nan"),
            r2=float("nan"),
            pct_monotone=float("nan"),
            drift_max=drift_max,
            drift_mean=drift_mean,
            saturation_log_v=None,
            n_anatomical_pass=0,
            n_anatomical_total=0,
            decision="PHASE_A_ONLY",
        )

    # Flatten only the finite (anchor, delta) pairs.
    mask = np.isfinite(predicted) & np.isfinite(decoded)
    x = predicted[mask]
    y = decoded[mask]
    if x.size < 3:
        raise ValueError(f"only {x.size} valid (predicted, decoded) pairs; need >= 3")

    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    r2 = float(r_value**2)

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample_anchors = rng.choice(a, size=a, replace=True)
        xs_list, ys_list = [], []
        for ai in sample_anchors:
            row_mask = np.isfinite(predicted[ai]) & np.isfinite(decoded[ai])
            xs_list.append(predicted[ai, row_mask])
            ys_list.append(decoded[ai, row_mask])
        xs = np.concatenate(xs_list)
        ys = np.concatenate(ys_list)
        if xs.size < 2:
            continue
        boots.append(stats.linregress(xs, ys).slope)
    slope_ci = (
        (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        if boots
        else (float("nan"), float("nan"))
    )

    monotone_count = 0
    for ai in range(a):
        row_mask = np.isfinite(predicted[ai]) & np.isfinite(decoded[ai])
        if row_mask.sum() < 3:
            continue
        rho = stats.spearmanr(predicted[ai, row_mask], decoded[ai, row_mask]).correlation
        if rho is not None and rho >= 0.9:
            monotone_count += 1
    pct_monotone = monotone_count / max(a, 1)

    saturation = _detect_saturation(predicted, decoded)
    decision = _decide(float(slope), r2, pct_monotone, drift_max)

    return CausalSteeringResult(
        segmenter_completed=True,
        n_anchors=int(a),
        n_deltas=int(n_d),
        slope=float(slope),
        slope_ci_lo=slope_ci[0],
        slope_ci_hi=slope_ci[1],
        intercept=float(intercept),
        r2=r2,
        pct_monotone=float(pct_monotone),
        drift_max=drift_max,
        drift_mean=drift_mean,
        saturation_log_v=saturation,
        n_anatomical_pass=0,  # filled by manual_scoring step
        n_anatomical_total=0,
        decision=decision,
    )


def _decide(slope: float, r2: float, pct_monotone: float, drift_max: float) -> str:
    """E2.4 §5 decision rules. Order matters — first failing gate wins."""
    if not (0.7 <= slope <= 1.3):
        return "FAIL_SLOPE"
    if r2 < 0.6:
        return "FAIL_R2"
    if pct_monotone < 0.8:
        return "FAIL_MONOTONICITY"
    if drift_max >= 0.15:
        return "FAIL_DRIFT"
    return "PASS"


def _detect_saturation(predicted: np.ndarray, decoded: np.ndarray) -> float | None:
    """Return the smallest |Δ log V| at which per-step slope departs from 1.

    Heuristic: for each delta column, regress decoded on predicted across
    anchors; the saturation threshold is the smallest |delta| whose per-step
    slope deviates from 1 by more than 0.3.
    """
    deltas = predicted - predicted.mean(axis=1, keepdims=True)
    n_d = predicted.shape[1]
    if n_d < 3:
        return None
    per_step_slope = []
    for di in range(n_d):
        col_mask = np.isfinite(predicted[:, di]) & np.isfinite(decoded[:, di])
        if col_mask.sum() < 3:
            per_step_slope.append(np.nan)
            continue
        s, _ = np.polyfit(predicted[col_mask, di], decoded[col_mask, di], 1)
        per_step_slope.append(float(s))
    per_step_slope_arr = np.asarray(per_step_slope, dtype=np.float64)
    abs_dev = np.abs(per_step_slope_arr - 1.0)
    bad = np.where(abs_dev > 0.3)[0]
    if bad.size == 0:
        return None
    abs_deltas = np.abs(deltas).max(axis=0)
    return float(abs_deltas[bad].min())


def write_result_json(result: CausalSteeringResult, path: Path) -> None:
    """Persist the result as JSON, NaN-safe."""
    payload = asdict(result)
    payload = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in payload.items()}
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
