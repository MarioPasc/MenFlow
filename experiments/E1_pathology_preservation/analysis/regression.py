"""Statistical pipeline for E1 §7.2 (Huber) and §7.3 (breakpoint).

The Huber log–log regression tests whether reconstruction error degrades with
shrinking tumor volume; the piecewise fit estimates the breakpoint volume
``V*`` below which the trend bends.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HuberResult:
    metric: str
    beta0: float
    beta1: float
    n: int
    bootstrap_ci_low: float
    bootstrap_ci_high: float


@dataclass(frozen=True, slots=True)
class BreakpointResult:
    metric: str
    breakpoint_cm3: float | None
    n: int


# ---------------------------------------------------------------------------
# §7.2 Log-log Huber regression
# ---------------------------------------------------------------------------


def fit_log_log_huber(
    df: pd.DataFrame,
    *,
    metric: str,
    volume_field: str = "volume_wt_cm3",
    n_bootstrap: int = 500,
    seed: int = 0,
) -> HuberResult:
    """Fit log(metric) ~ beta0 + beta1 * log(volume) with Huber loss.

    Bootstrap confidence interval for ``beta1`` is computed by resampling rows
    with replacement; residuals are heteroscedastic so a non-parametric CI is
    safer than relying on closed-form Huber se.
    """
    mask = (df[metric] > 0) & (df[volume_field] > 0)
    sub = df.loc[mask, [metric, volume_field]].dropna()
    if len(sub) < 4:
        return HuberResult(metric, float("nan"), float("nan"), len(sub), float("nan"), float("nan"))

    x = np.log(sub[volume_field].to_numpy()).reshape(-1, 1)
    y = np.log(sub[metric].to_numpy())
    model = HuberRegressor().fit(x, y)
    beta0 = float(model.intercept_)
    beta1 = float(model.coef_[0])

    rng = np.random.default_rng(seed)
    coefs = np.empty(n_bootstrap, dtype=np.float64)
    n = len(sub)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        x_b = x[idx]
        y_b = y[idx]
        try:
            coefs[b] = HuberRegressor().fit(x_b, y_b).coef_[0]
        except Exception:  # pragma: no cover — robustness fallback
            coefs[b] = np.nan
    ci = np.nanpercentile(coefs, [2.5, 97.5])

    return HuberResult(metric, beta0, beta1, n, float(ci[0]), float(ci[1]))


# ---------------------------------------------------------------------------
# §7.3 Piecewise breakpoint
# ---------------------------------------------------------------------------


def detect_breakpoint(
    df: pd.DataFrame,
    *,
    metric: str,
    volume_field: str = "volume_wt_cm3",
) -> BreakpointResult:
    """Two-segment piecewise linear fit on (log V, log metric).

    Returns ``None`` if ``pwlf`` is unavailable or the fit fails (small n).
    """
    try:
        import pwlf
    except ImportError:  # pragma: no cover
        logger.warning("pwlf not installed; breakpoint disabled")
        return BreakpointResult(metric, None, 0)

    sub = df.loc[(df[metric] > 0) & (df[volume_field] > 0), [metric, volume_field]].dropna()
    if len(sub) < 8:
        return BreakpointResult(metric, None, len(sub))

    x = np.log(sub[volume_field].to_numpy())
    y = np.log(np.clip(sub[metric].to_numpy(), 1e-9, None))
    try:
        pw = pwlf.PiecewiseLinFit(x, y)
        pw.fit(2)
        log_break = pw.fit_breaks[1]
    except Exception as exc:  # pragma: no cover
        logger.warning("pwlf fit failed for %s: %s", metric, exc)
        return BreakpointResult(metric, None, len(sub))
    return BreakpointResult(metric, float(np.exp(log_break)), len(sub))


# ---------------------------------------------------------------------------
# Bulk runner
# ---------------------------------------------------------------------------


def fit_all(
    df: pd.DataFrame,
    *,
    metrics: Sequence[str],
    volume_field: str = "volume_wt_cm3",
    n_bootstrap: int = 500,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Huber + breakpoint over a list of metrics; return two DataFrames."""
    huber_rows = [
        fit_log_log_huber(
            df,
            metric=m,
            volume_field=volume_field,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        for m in metrics
    ]
    bp_rows = [detect_breakpoint(df, metric=m, volume_field=volume_field) for m in metrics]
    from dataclasses import asdict

    huber_df = pd.DataFrame([asdict(r) for r in huber_rows])
    bp_df = pd.DataFrame([asdict(r) for r in bp_rows])
    return huber_df, bp_df
