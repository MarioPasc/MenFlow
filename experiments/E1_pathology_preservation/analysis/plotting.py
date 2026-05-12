"""Single-page diagnostic figures for E1 (§11)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for SLURM jobs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.E1_pathology_preservation.analysis.volume_stratification import (
    VOLUME_BIN_LABELS,
)

logger = logging.getLogger(__name__)


def boxplot_metric_by_bin(
    df: pd.DataFrame,
    *,
    metric: str,
    output: Path,
    title: str | None = None,
) -> None:
    """Box plot of ``metric`` stratified by ``volume_bin``."""
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [df.loc[df["volume_bin"] == b, metric].dropna().to_numpy() for b in VOLUME_BIN_LABELS]
    counts = [int(arr.size) for arr in data]
    positions = list(range(1, len(VOLUME_BIN_LABELS) + 1))
    ax.boxplot(data, positions=positions, widths=0.7, showfliers=False)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(VOLUME_BIN_LABELS, counts)])
    ax.set_xlabel("Volume bin (cm³)")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by volume bin")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def loglog_scatter(
    df: pd.DataFrame,
    *,
    metric: str,
    volume_field: str = "volume_wt_cm3",
    huber_intercept: float | None = None,
    huber_slope: float | None = None,
    breakpoint_cm3: float | None = None,
    output: Path,
    title: str | None = None,
) -> None:
    """Log-log scatter of metric vs volume, with optional Huber fit + breakpoint."""
    sub = df[(df[metric] > 0) & (df[volume_field] > 0)].dropna(subset=[metric, volume_field])
    if sub.empty:
        logger.warning("Skipping loglog scatter for %s — no positive samples", metric)
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sub[volume_field], sub[metric], s=12, alpha=0.5, color="steelblue")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"{volume_field} (cm³)")
    ax.set_ylabel(metric)

    if huber_intercept is not None and huber_slope is not None and not np.isnan(huber_slope):
        x_grid = np.geomspace(sub[volume_field].min(), sub[volume_field].max(), 100)
        y_grid = np.exp(huber_intercept + huber_slope * np.log(x_grid))
        ax.plot(
            x_grid,
            y_grid,
            color="darkred",
            linewidth=2,
            label=f"Huber β₁={huber_slope:.3f}",
        )

    if breakpoint_cm3 is not None and not (
        breakpoint_cm3 != breakpoint_cm3  # NaN check
    ):
        ax.axvline(
            breakpoint_cm3,
            color="darkorange",
            linestyle="--",
            linewidth=1.5,
            label=f"V*={breakpoint_cm3:.2f} cm³",
        )

    ax.legend(loc="best")
    ax.set_title(title or metric)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def render_all(
    *,
    metrics_df: pd.DataFrame,
    huber_df: pd.DataFrame,
    breakpoint_df: pd.DataFrame,
    metrics: Sequence[str],
    volume_field: str,
    output_dir: Path,
) -> None:
    """Write one boxplot + one loglog scatter per metric to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    huber_by_metric = huber_df.set_index("metric").to_dict("index")
    bp_by_metric = breakpoint_df.set_index("metric").to_dict("index")

    for metric in metrics:
        boxplot_metric_by_bin(
            metrics_df,
            metric=metric,
            output=output_dir / f"box_{metric}.png",
        )
        h = huber_by_metric.get(metric, {})
        bp = bp_by_metric.get(metric, {})
        loglog_scatter(
            metrics_df,
            metric=metric,
            volume_field=volume_field,
            huber_intercept=h.get("beta0"),
            huber_slope=h.get("beta1"),
            breakpoint_cm3=bp.get("breakpoint_cm3"),
            output=output_dir / f"loglog_{metric}.png",
        )

    logger.info("Wrote E1 figures to %s", output_dir)
