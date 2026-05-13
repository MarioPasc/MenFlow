"""Plots for E2.3 — spatial localization.

Two figures saved to the experiment output directory:
- r2_three_bars.png: 1x2 panel (raw / post-FWL), 3 bars each with 95% CI errorbars.
- delta_r2_summary.png: 2 horizontal bars (raw ΔR², post-FWL ΔR²) with threshold lines.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    pass  # Older matplotlib — default style is fine

logger = logging.getLogger(__name__)

# Import type only for annotation; avoid circular import at module level
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experiments.E2.E2_3_spatial_localization.analysis.spatial import SpatialResult

_POOLING_LABELS = ["global", "tumor", "random"]
_POOLING_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e"]


def _r2_three_bars(out_path: Path, result: SpatialResult, modality: str) -> None:
    """1x2 subplot: raw R² and post-FWL R² for three pooling strategies."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)

    panels = [
        (
            "Raw",
            [result.r2_global, result.r2_tumor, result.r2_random],
            [result.r2_global_ci, result.r2_tumor_ci, result.r2_random_ci],
        ),
        (
            "Post-FWL",
            [result.r2_global_post_fwl, result.r2_tumor_post_fwl, result.r2_random_post_fwl],
            [
                result.r2_global_post_fwl_ci,
                result.r2_tumor_post_fwl_ci,
                result.r2_random_post_fwl_ci,
            ],
        ),
    ]

    for ax, (title, r2s, cis) in zip(axes, panels):
        x = range(len(_POOLING_LABELS))
        yerr_lo = [r - ci[0] for r, ci in zip(r2s, cis)]
        yerr_hi = [ci[1] - r for r, ci in zip(r2s, cis)]
        # Clamp to avoid negative error bar lengths from CI noise
        yerr_lo = [max(0.0, v) for v in yerr_lo]
        yerr_hi = [max(0.0, v) for v in yerr_hi]

        bars = ax.bar(
            x,
            r2s,
            color=_POOLING_COLORS,
            alpha=0.85,
            yerr=[yerr_lo, yerr_hi],
            capsize=5,
            error_kw={"elinewidth": 1.2, "ecolor": "black"},
        )
        for bar, r2 in zip(bars, r2s):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                max(0.0, r2) + 0.02,
                f"{r2:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_xticks(list(x))
        ax.set_xticklabels(_POOLING_LABELS)
        ax.set_ylim(-0.1, 1.0)
        ax.set_ylabel("R²")
        ax.set_title(f"E2.3 — {title} ({modality})")
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _delta_r2_summary(out_path: Path, result: SpatialResult) -> None:
    """Two horizontal bars: raw ΔR² and post-FWL ΔR² with threshold lines."""
    operator_colors = {
        "strongly_local": "#2ca02c",
        "local": "#ff7f0e",
        "global": "#d62728",
    }
    color = operator_colors.get(result.operator, "#1f77b4")

    labels = ["Raw ΔR²", "Post-FWL ΔR²"]
    values = [result.delta_r2_spatial, result.delta_r2_post_fwl]

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    y_pos = range(len(labels))
    ax.barh(list(y_pos), values, color=color, alpha=0.85)

    ax.axvline(0.05, color="#ff7f0e", lw=1.2, ls="--", label="local threshold (0.05)")
    ax.axvline(0.15, color="#2ca02c", lw=1.2, ls="--", label="strongly_local threshold (0.15)")
    ax.axvline(0.0, color="black", lw=0.8, ls="-")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.set_xlabel("ΔR² (tumor − global)")
    ax.set_title(f"E2.3 — Spatial gap (operator: {result.operator})")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def make_all(
    out_dir: Path,
    result: SpatialResult,
    modality: str = "t1c",
) -> None:
    """Produce all E2.3 diagnostic figures.

    Parameters
    ----------
    out_dir:
        Directory where PNG files are saved (must exist).
    result:
        SpatialResult from run_spatial.
    modality:
        Modality label for figure titles.
    """
    _r2_three_bars(out_dir / "r2_three_bars.png", result, modality)
    _delta_r2_summary(out_dir / "delta_r2_summary.png", result)
