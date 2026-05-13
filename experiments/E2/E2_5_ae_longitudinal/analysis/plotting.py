"""Figures for E2.5 — spec §8.3."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: do not require a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)


def plot_rho_lin_distribution(
    rho_lin: np.ndarray,
    rho_lin_null: np.ndarray,
    *,
    rho_pass: float,
    rho_intermediate: float,
    out_path: Path,
) -> Path:
    """Histogram of per-patient ρ_lin overlaid on the null distribution."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0.0, max(1.0, float(np.nanmax([rho_lin.max() if rho_lin.size else 0.0, rho_lin_null.max() if rho_lin_null.size else 0.0]))), 30)
    ax.hist(
        rho_lin_null,
        bins=bins,
        alpha=0.45,
        color="grey",
        density=True,
        label=f"null (n={len(rho_lin_null)})",
    )
    ax.hist(
        rho_lin,
        bins=bins,
        alpha=0.7,
        color="C0",
        density=True,
        label=f"same-patient (n={len(rho_lin)})",
    )
    ax.axvspan(0.0, rho_pass, color="green", alpha=0.08)
    ax.axvspan(rho_pass, rho_intermediate, color="orange", alpha=0.08)
    ax.axvspan(rho_intermediate, ax.get_xlim()[1], color="red", alpha=0.08)
    ax.axvline(rho_pass, color="green", lw=1, ls="--", label=f"R1 ≤ {rho_pass:.2f}")
    ax.axvline(rho_intermediate, color="red", lw=1, ls="--", label=f"R3 > {rho_intermediate:.2f}")
    ax.set_xlabel(r"$\rho_{\mathrm{lin}}$")
    ax.set_ylabel("density")
    ax.set_title("Same-patient vs cross-patient linearity residual")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_beta_residual_scatter(per_patient_df: pd.DataFrame, *, out_path: Path) -> Path:
    """(β_residual, image SSIM at β_vol) scatter; identifies time vs vol mismatch."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(
        per_patient_df["beta_residual"], per_patient_df["ssim_at_beta_vol"], s=24
    )
    axes[0].set_xlabel(r"$|\beta_{\mathrm{vol}} - \beta_{\mathrm{time}}|$")
    axes[0].set_ylabel(r"SSIM($\hat x_{\beta_{\mathrm{vol}}}$, $x^{(2)}$)")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].grid(alpha=0.3)

    axes[1].scatter(
        per_patient_df["beta_residual"],
        per_patient_df.get("ssim_nontumour_mean", np.full(len(per_patient_df), np.nan)),
        s=24,
        color="C1",
    )
    axes[1].set_xlabel(r"$|\beta_{\mathrm{vol}} - \beta_{\mathrm{time}}|$")
    axes[1].set_ylabel(r"mean non-tumour SSIM")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].grid(alpha=0.3)
    fig.suptitle("β residual vs SSIM (per patient)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_anatomy_ssim_curves(
    anatomy_ssim_curves: np.ndarray,  # (n_patients, n_beta)
    *,
    betas: np.ndarray,
    out_path: Path,
) -> Path:
    """Mean ± SE non-tumour SSIM as a function of β; thin per-patient lines underlaid."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for row in anatomy_ssim_curves:
        ax.plot(betas, row, color="C0", alpha=0.18, lw=0.7)
    mean = np.nanmean(anatomy_ssim_curves, axis=0)
    se = np.nanstd(anatomy_ssim_curves, axis=0, ddof=1) / np.sqrt(
        np.sum(~np.isnan(anatomy_ssim_curves), axis=0).clip(min=1)
    )
    ax.plot(betas, mean, color="C0", lw=2.0, label="mean ± SE")
    ax.fill_between(betas, mean - se, mean + se, color="C0", alpha=0.25)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("non-tumour SSIM (vs closest endpoint)")
    ax.set_title(f"Anatomy preservation along the latent line (n={anatomy_ssim_curves.shape[0]})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_rho_variants(
    per_patient_df,
    *,
    out_path: Path,
    rho_pass: float,
    rho_intermediate: float,
) -> Path:
    """Three-column comparison: joint vs masked vs mask-pooled ρ_lin distributions."""
    cols = [
        ("rho_lin_joint", r"$\rho_\mathrm{lin}$ (full latent)"),
        ("rho_lin_masked", r"$\rho_\mathrm{lin}$ (mask-localised)"),
        ("rho_lin_pooled", r"$\rho_\mathrm{lin}$ (mask-pooled, 16-D)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (col, title) in zip(axes, cols):
        if col not in per_patient_df.columns:
            ax.set_axis_off()
            continue
        vals = per_patient_df[col].dropna().values
        if vals.size == 0:
            ax.set_axis_off()
            continue
        ax.hist(vals, bins=20, color="C0", alpha=0.8)
        ax.axvline(rho_pass, color="green", ls="--", lw=1, label=f"R1 ≤ {rho_pass:.2f}")
        ax.axvline(rho_intermediate, color="red", ls="--", lw=1, label=f"R3 > {rho_intermediate:.2f}")
        ax.set_xlabel(title)
        ax.set_ylabel("count")
        if ax is axes[0]:
            ax.legend(fontsize=8)
    fig.suptitle("ρ_lin variants — same-patient triples")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_pca_alignment(pca_dict: dict, *, out_path: Path) -> Path:
    """Per-patient cosine of (z²−z¹) with PC1 of {z³−z¹}, sorted; bar chart."""
    import numpy as np

    cos = np.asarray(pca_dict.get("cos_z2_minus_z1_per_patient", []), dtype=float)
    if cos.size == 0:
        return out_path
    order = np.argsort(cos)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(np.arange(cos.size), cos[order], color="C0")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axhline(0.70, color="green", ls="--", lw=1, label="R1 threshold 0.70")
    ax.axhline(0.40, color="orange", ls="--", lw=1, label="R2 floor 0.40")
    ax.set_xlabel("patient index (sorted)")
    ax.set_ylabel(r"cos(z²−z¹, PC1)")
    var = pca_dict.get("variance_explained_pc1", float("nan"))
    cos_mean = pca_dict.get("cos_z2_minus_z1_mean", float("nan"))
    ax.set_title(
        f"PCA on {{z³−z¹}}: PC1 var = {var:.2f}, mean cos(z²−z¹, PC1) = {cos_mean:.2f}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_qualitative_panel(
    *,
    panels: list[dict],
    out_path: Path,
) -> Path:
    """Grid of (real / decoded β / real) thumbnails for representative patients.

    Each entry of ``panels`` is ``{"patient_id": str, "rho_lin": float,
    "real_t1": (H,W,D) np array, "real_t3": same, "real_t2": same,
    "decoded": list of (β, (H,W,D))}``. Mid-axial slice is shown.
    """
    if not panels:
        logger.warning("plot_qualitative_panel: no panels supplied")
        return out_path
    n_rows = len(panels)
    n_cols = 2 + len(panels[0]["decoded"]) + 1  # real_t1 | decoded β's | real_t3 ; real_t2 inline
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.0 * n_rows))
    if n_rows == 1:
        axes = axes[None]

    for row_i, p in enumerate(panels):
        slc = p["real_t1"].shape[-1] // 2
        cols = (
            [("real t1", p["real_t1"])]
            + [(f"β={b:.2f}", img) for b, img in p["decoded"]]
            + [("real t3", p["real_t3"])]
        )
        # Append real_t2 in the n_cols-1 slot (overwriting), only if it fits.
        if len(cols) > n_cols:
            cols = cols[:n_cols]
        for col_i, (title, vol) in enumerate(cols):
            ax = axes[row_i, col_i]
            ax.imshow(vol[..., slc].T, cmap="gray", origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_i == 0:
                ax.set_title(title, fontsize=9)
            if col_i == 0:
                ax.set_ylabel(
                    f"{p['patient_id']}\nρ_lin={p['rho_lin']:.2f}", fontsize=8
                )
    fig.suptitle("Qualitative latent-line traversal", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
