"""Volume binning for E1 §5: log-spaced bins motivated by clinical thresholds.

Bins are computed over the *whole-tumor* volume by default (the union of NETC,
SNFH, ET); pass ``volume_field='volume_tc_cm3'`` for tumor-core stratification.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

VOLUME_BIN_EDGES_CM3: tuple[float, ...] = (0.1, 1.0, 5.0, 15.0, 50.0, float("inf"))
VOLUME_BIN_LABELS: tuple[str, ...] = ("B1", "B2", "B3", "B4", "B5")


def stratify_by_volume(df: pd.DataFrame, *, volume_field: str = "volume_wt_cm3") -> pd.DataFrame:
    """Return ``df`` augmented with a categorical ``volume_bin`` column."""
    out = df.copy()
    out["volume_bin"] = pd.cut(
        out[volume_field],
        bins=VOLUME_BIN_EDGES_CM3,
        labels=list(VOLUME_BIN_LABELS),
        right=False,
    )
    return out


def summarize_by_bin(
    df: pd.DataFrame,
    *,
    metric_columns: Sequence[str],
) -> pd.DataFrame:
    """Median + IQR + 5/95 percentile per bin per metric (E1 §7.1)."""
    if "volume_bin" not in df.columns:
        raise KeyError("expected 'volume_bin' column; call stratify_by_volume first")
    rows = []
    for bin_label, group in df.groupby("volume_bin", observed=False):
        for col in metric_columns:
            vals = group[col].dropna()
            if vals.empty:
                rows.append(
                    {"bin": bin_label, "metric": col, "n": 0}
                    | {k: float("nan") for k in ("median", "q1", "q3", "p05", "p95")}
                )
                continue
            rows.append(
                {
                    "bin": bin_label,
                    "metric": col,
                    "n": int(vals.size),
                    "median": float(vals.median()),
                    "q1": float(vals.quantile(0.25)),
                    "q3": float(vals.quantile(0.75)),
                    "p05": float(vals.quantile(0.05)),
                    "p95": float(vals.quantile(0.95)),
                }
            )
    return pd.DataFrame(rows)
