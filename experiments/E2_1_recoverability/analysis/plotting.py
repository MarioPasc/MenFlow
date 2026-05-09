"""Figures for E2.1: predicted vs observed scatter, pooling comparison bar."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_pred_vs_obs(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    regime: str,
    r2_text: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(y_true, y_pred, s=12, alpha=0.5, edgecolor="none")
    lim_lo = float(min(y_true.min(), y_pred.min()))
    lim_hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, alpha=0.7)
    ax.set_xlabel(r"observed $\log V$ (cm³)")
    ax.set_ylabel(r"predicted $\log V$ (cm³)")
    ax.set_title(f"Linear probe — {r2_text}  | regime: {regime}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pooling_comparison(
    *,
    labels: tuple[str, ...],
    r2_lin: tuple[float, ...],
    r2_mlp: tuple[float, ...],
    ci_low: tuple[float, ...],
    ci_high: tuple[float, ...],
    out_path: Path,
) -> None:
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    bars_lin = ax.bar(x - width / 2, r2_lin, width, label="ridge", color="#1f77b4")
    ax.bar(x + width / 2, r2_mlp, width, label="MLP", color="#ff7f0e")
    # CI bars on the linear column where available.
    for i, (lo, hi) in enumerate(zip(ci_low, ci_high)):
        if np.isfinite(lo) and np.isfinite(hi):
            ax.errorbar(
                x[i] - width / 2,
                r2_lin[i],
                yerr=[[r2_lin[i] - lo], [hi - r2_lin[i]]],
                fmt="none",
                ecolor="black",
                capsize=4,
                lw=1,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$R^2$ on $\log V$")
    ax.set_ylim(min(0, min(r2_lin + r2_mlp) - 0.05), 1.0)
    ax.axhline(0.6, color="grey", lw=0.8, ls=":")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.set_title("Pooling strategy: mask-anchored vs global")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    _ = bars_lin  # silence unused
