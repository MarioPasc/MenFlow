"""Plots for E2.2 — direction identifiability.

Four figures saved to the experiment output directory:
- bootstrap_cosine_hist.png
- svd_scree.png
- rho_conf_bar.png
- drop_one_sensitivity.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Apply paper style if available; silently fall back to default
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    pass  # Older matplotlib — default style is fine


def _bootstrap_cosine_hist(
    W: np.ndarray,
    rho_mean: float,
    rho_p5: float,
    out_path: Path,
) -> None:
    """Histogram of all pairwise cosines from bootstrap direction matrix."""
    valid_mask = ~np.isnan(W).any(axis=1)
    W_valid = W[valid_mask]
    n = W_valid.shape[0]
    if n < 2:
        logger.warning("bootstrap_cosine_hist: fewer than 2 valid directions; skipping plot")
        return

    G = W_valid @ W_valid.T
    upper_i, upper_j = np.triu_indices(n, k=1)
    cosines = G[upper_i, upper_j]

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.hist(cosines, bins=40, color="#1f77b4", alpha=0.75, edgecolor="none")
    ax.axvline(rho_mean, color="black", lw=1.5, ls="-", label=f"mean={rho_mean:.3f}")
    ax.axvline(rho_p5, color="black", lw=1.2, ls="--", label=f"p5={rho_p5:.3f}")
    ax.axvline(0.70, color="#d62728", lw=1.0, ls=":", label="threshold 0.70")
    ax.axvline(0.85, color="#ff7f0e", lw=1.0, ls=":", label="threshold 0.85")
    ax.set_xlabel("Pairwise cosine similarity")
    ax.set_ylabel("Count")
    ax.set_title("E2.2 — Bootstrap direction stability")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _svd_scree(W: np.ndarray, out_path: Path) -> None:
    """Top-3 singular values of the bootstrap direction matrix (n_boot, C)."""
    valid_mask = ~np.isnan(W).any(axis=1)
    W_valid = W[valid_mask]
    if W_valid.shape[0] < 2:
        logger.warning("svd_scree: not enough valid directions; skipping plot")
        return

    _, s, _ = np.linalg.svd(W_valid, full_matrices=False)
    top_k = min(3, len(s))
    s_top = s[:top_k]

    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    ax.bar(range(1, top_k + 1), s_top, color="#1f77b4", alpha=0.8)
    ax.set_xticks(range(1, top_k + 1))
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")
    ax.set_title("E2.2 — Bootstrap direction SVD scree")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _rho_conf_bar(
    rho_conf_tcav: float,
    rho_conf_fwl: float,
    regime: str,
    out_path: Path,
) -> None:
    """Two-bar chart: rho_conf_tcav vs rho_conf_fwl with threshold lines."""
    # Colour-code bars by regime
    color_map = {
        "stable_unconfounded": "#2ca02c",
        "stable_confounded": "#ff7f0e",
        "subspace": "#d62728",
        "path_d": "#9467bd",
    }
    bar_color = color_map.get(regime, "#1f77b4")

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    x = [0, 1]
    heights = [rho_conf_tcav, rho_conf_fwl]
    ax.bar(x, heights, color=bar_color, alpha=0.8, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["TCAV", "FWL"])
    ax.set_ylabel(r"$|\cos(\mathbf{u}_{\mathrm{naive}}, \mathbf{u}_{\cdot})|$")
    ax.axhline(0.90, color="#2ca02c", lw=1.0, ls="--", label="0.90 (use_naive)")
    ax.axhline(0.60, color="#ff7f0e", lw=1.0, ls=":", label="0.60 (min threshold)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"E2.2 — Confounder cosines (regime: {regime})")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _drop_one_sensitivity(sensitivity: dict[str, float | None], out_path: Path) -> None:
    """Bar per dropped covariate: cosine from full TCAV to drop-one TCAV."""
    labels = list(sensitivity.keys())
    values = [v if v is not None else 0.0 for v in sensitivity.values()]
    missing = [v is None for v in sensitivity.values()]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars = ax.bar(range(len(labels)), values, color="#1f77b4", alpha=0.8)
    # Mark bars with None (insufficient pairs) in grey
    for i, is_missing in enumerate(missing):
        if is_missing:
            bars[i].set_color("#aaaaaa")
            bars[i].set_alpha(0.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15)
    ax.axhline(0.95, color="#d62728", lw=1.0, ls="--", label="0.95 stability")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(r"$|\cos(\mathbf{u}_{\mathrm{TCAV}}, \mathbf{u}_{\mathrm{drop-one}})|$")
    ax.set_title("E2.2 — Drop-one covariate sensitivity")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def make_all(
    output_dir: Path,
    W: np.ndarray,
    rho_boot_mean: float,
    rho_boot_p5: float,
    rho_conf_tcav: float,
    rho_conf_fwl: float,
    regime: str,
    sensitivity: dict[str, float | None],
) -> None:
    """Produce all four E2.2 diagnostic figures.

    Parameters
    ----------
    output_dir:
        Directory where PNG files are saved (must exist).
    W:
        Bootstrap direction matrix ``(n_boot, C)``.
    rho_boot_mean:
        Mean pairwise cosine from the bootstrap.
    rho_boot_p5:
        5th-percentile pairwise cosine.
    rho_conf_tcav:
        Absolute cosine between u_naive and u_TCAV.
    rho_conf_fwl:
        Absolute cosine between u_naive and u_FWL.
    regime:
        One of ``"stable_unconfounded"``, ``"stable_confounded"``,
        ``"subspace"``, ``"path_d"``.
    sensitivity:
        Dict mapping dropped covariate name to cosine (or None if TCAV failed).
    """
    _bootstrap_cosine_hist(W, rho_boot_mean, rho_boot_p5, output_dir / "bootstrap_cosine_hist.png")
    _svd_scree(W, output_dir / "svd_scree.png")
    _rho_conf_bar(rho_conf_tcav, rho_conf_fwl, regime, output_dir / "rho_conf_bar.png")
    _drop_one_sensitivity(sensitivity, output_dir / "drop_one_sensitivity.png")
