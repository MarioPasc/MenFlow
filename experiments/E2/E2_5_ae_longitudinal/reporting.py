"""CSV / JSON / bootstrap helpers for E2.5 outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


def bootstrap_ci(
    samples: np.ndarray,
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> dict[str, float]:
    """Return ``{mean, ci_lo, ci_hi}`` for the chosen statistic via the percentile method."""
    arr = np.asarray(samples, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    if arr.size == 1:
        v = float(statistic(arr))
        return {"mean": v, "ci_lo": v, "ci_hi": v, "n": 1}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_resamples, dtype=np.float64)
    n = arr.size
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boots[i] = statistic(arr[idx])
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return {
        "mean": float(statistic(arr)),
        "ci_lo": lo,
        "ci_hi": hi,
        "n": int(arr.size),
    }


def cluster_bootstrap_ci(
    samples: np.ndarray,
    clusters: np.ndarray,
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> dict[str, float]:
    """Patient-cluster bootstrap.

    Resamples *clusters* (patients) with replacement and concatenates all their
    samples. Correct under within-patient correlation, unlike i.i.d. bootstrap
    which underestimates the variance when multiple triples share a patient.
    """
    arr = np.asarray(samples, dtype=np.float64)
    cl = np.asarray(clusters)
    finite = np.isfinite(arr)
    arr = arr[finite]
    cl = cl[finite]
    if arr.size == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0, "n_clusters": 0}
    uniq, idx = np.unique(cl, return_inverse=True)
    if uniq.size < 2:
        v = float(statistic(arr))
        return {"mean": v, "ci_lo": v, "ci_hi": v, "n": int(arr.size), "n_clusters": int(uniq.size)}
    # Pre-bucket sample indices per cluster.
    buckets: list[np.ndarray] = [np.where(idx == ci)[0] for ci in range(uniq.size)]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        chosen = rng.integers(0, uniq.size, size=uniq.size)
        rows = np.concatenate([buckets[c] for c in chosen])
        boots[i] = statistic(arr[rows])
    return {
        "mean": float(statistic(arr)),
        "ci_lo": float(np.quantile(boots, alpha / 2.0)),
        "ci_hi": float(np.quantile(boots, 1.0 - alpha / 2.0)),
        "n": int(arr.size),
        "n_clusters": int(uniq.size),
    }


def bootstrap_paired_diff(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Two-sample percentile bootstrap on the difference of means: ``mean(b) - mean(a)``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        ai = rng.integers(0, a.size, size=a.size)
        bi = rng.integers(0, b.size, size=b.size)
        boots[i] = float(b[bi].mean() - a[ai].mean())
    return {
        "mean": float(b.mean() - a.mean()),
        "ci_lo": float(np.quantile(boots, alpha / 2.0)),
        "ci_hi": float(np.quantile(boots, 1.0 - alpha / 2.0)),
    }


def write_per_patient_csv(records: list[dict], out_path: Path) -> Path:
    df = pd.DataFrame.from_records(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def write_json(payload: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    return out_path


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")
