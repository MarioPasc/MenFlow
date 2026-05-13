"""E2.4 plotting — scatter, drift histogram, intensity proxy, monotonicity panel.

All functions write to ``output_dir`` and return the saved file paths. They
tolerate Phase-A NaN-only ``decoded_log_v`` by emitting a placeholder figure
that announces the missing segmentation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)


def plot_drift_histogram(sweep_h5_path: Path, output_path: Path) -> Path:
    """Histogram of off-manifold drift across all (anchor, delta) cells."""
    with h5py.File(sweep_h5_path, "r") as f:
        drift = f["drift"][:].astype(np.float64).ravel()
    drift = drift[np.isfinite(drift)]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(drift, bins=20, color="#3a86ff", edgecolor="black")
    ax.axvline(0.15, color="crimson", linestyle="--", label="spec threshold (0.15)")
    ax.set_xlabel(r"off-manifold drift $\|z' - z_\mathrm{re}\| / \|z'\|$")
    ax.set_ylabel("count")
    ax.set_title("Off-manifold drift distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_intensity_proxy(sweep_h5_path: Path, output_path: Path) -> Path:
    """Per-anchor intensity-proxy curves vs Δ log V."""
    with h5py.File(sweep_h5_path, "r") as f:
        deltas = f["delta_log_v_grid"][:]
        intensity = f["intensity_proxy"][:]
        scan_ids = f["scan_id"].asstr()[:]
        bins = f["volume_bin"][:]
    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = plt.get_cmap("viridis")
    for ai in range(intensity.shape[0]):
        ax.plot(
            deltas,
            intensity[ai],
            marker="o",
            color=cmap((bins[ai] - 1) / max(bins.max() - 1, 1)),
            label=f"B{int(bins[ai])} {scan_ids[ai]}",
        )
    ax.axvline(0, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel(r"$\Delta \log V$ (nats)")
    ax.set_ylabel("mean intensity inside source-mask region")
    ax.set_title("Intensity proxy vs steering step (Phase A; segmenter pending)")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_scatter_pred_vs_decoded(sweep_h5_path: Path, output_path: Path) -> Path:
    """Scatter of decoded vs predicted log V, OLS line + identity reference."""
    with h5py.File(sweep_h5_path, "r") as f:
        predicted = f["predicted_log_v"][:].astype(np.float64)
        decoded = f["decoded_log_v"][:].astype(np.float64)
        completed = bool(f.attrs.get("segmenter_completed", False))
    fig, ax = plt.subplots(figsize=(5, 5))
    if not completed or not np.any(np.isfinite(decoded)):
        ax.text(
            0.5,
            0.5,
            "PHASE A: segmenter pending\n(no decoded_log_v available)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            color="crimson",
        )
        ax.set_axis_off()
    else:
        m = np.isfinite(predicted) & np.isfinite(decoded)
        x, y = predicted[m], decoded[m]
        ax.scatter(x, y, alpha=0.6, s=30)
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="identity")
        slope, intercept = np.polyfit(x, y, 1)
        ax.plot(
            [lo, hi],
            [slope * lo + intercept, slope * hi + intercept],
            "r-",
            label=f"OLS slope={slope:.2f}",
        )
        ax.set_xlabel(r"predicted $\log V_0 + \Delta\log V$")
        ax.set_ylabel(r"decoded $\log V$")
        ax.set_title("Causal steering: decoded vs predicted log V")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def make_all(sweep_h5_path: Path, output_dir: Path) -> dict[str, Path]:
    """Run every Phase-A-safe plot. Returns a dict of artefact name -> path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "drift_histogram": plot_drift_histogram(sweep_h5_path, output_dir / "drift_histogram.png"),
        "intensity_proxy_vs_delta": plot_intensity_proxy(
            sweep_h5_path, output_dir / "intensity_proxy_vs_delta.png"
        ),
        "scatter_pred_vs_decoded": plot_scatter_pred_vs_decoded(
            sweep_h5_path, output_dir / "scatter_pred_vs_decoded.png"
        ),
    }


# ---------------------------------------------------------------------------
# Phase B headline grid (per-anchor 5 × 8 figure)
# ---------------------------------------------------------------------------

import nibabel as _nib  # noqa: E402  — imported lazily for tests that don't need it

_GRID_MODALITIES: tuple[str, ...] = ("t1c", "t1n", "t2f", "t2w")


def _tumor_centroid_z(seg: np.ndarray) -> int:
    nz = np.argwhere(seg > 0)
    if nz.size == 0:
        return seg.shape[2] // 2
    return int(np.median(nz[:, 2]))


def _percentile_window(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, lo)), float(np.percentile(finite, hi))


def make_modality_grid(
    sweep_h5_path: Path,
    source_h5_path: Path,
    anchor_index_in_sweep: int,
    output_path: Path,
    *,
    decoded_root: Path | None = None,
    h5_modality_order: tuple[str, ...] = ("t1c", "t1n", "t2f", "t2w"),
) -> Path:
    """Build the 5-row × 8-column headline grid for one anchor.

    Rows: t1c, t1n, t2f, t2w, predicted whole-tumor mask. The mask is on its
    own row, **not** overlaid on imaging (per user spec).

    Columns: ground-truth source image of the anchor (col 0) +
    Δ ∈ {−1.5, −1.0, −0.5, 0, +0.5, +1.0, +1.5} (cols 1–7).

    The 4 modality rows show the *decoded* image at each Δ. For Δ=0 the
    decoded image is essentially the autoencoder reconstruction. The
    ground-truth column shows the original BraTS source image (no encoding).
    """
    import h5py  # local — avoids re-importing at module level

    sweep_h5_path = Path(sweep_h5_path)
    if decoded_root is None:
        decoded_root = sweep_h5_path.parent / "decoded"
    with h5py.File(sweep_h5_path, "r") as sw:
        deltas = sw["delta_log_v_grid"][:]
        scan_ids = sw["decoded_nifti_path"].asstr()[:]  # (A, D)
        nifti_paths = sw["decoded_nifti_path"].asstr()[:]
        anchor_rows = sw.attrs["anchor_row_indices"]
        scan_id_arr = sw["scan_id"].asstr()[:]
        scan_id = str(scan_id_arr[anchor_index_in_sweep])
        bin_id = int(sw["volume_bin"][anchor_index_in_sweep])
        log_v0 = float(sw["log_v0"][anchor_index_in_sweep])
    del scan_ids  # not used; kept for clarity

    anchor_row = int(anchor_rows[anchor_index_in_sweep])

    # Source images — read all 4 modalities for the anchor.
    with h5py.File(source_h5_path, "r") as src:
        src_modalities = [
            (m.decode() if isinstance(m, bytes) else str(m)) for m in src.attrs["modalities"]
        ]
        gt_images: dict[str, np.ndarray] = {}
        for m in _GRID_MODALITIES:
            if m not in src_modalities:
                raise KeyError(f"source H5 missing modality {m!r}")
            ch = src_modalities.index(m)
            gt_images[m] = src["images"][anchor_row, ch].astype(np.float32)
        gt_seg = src["segmentations"][anchor_row].astype(np.int32)
    z_slice = _tumor_centroid_z(gt_seg)

    # Decoded images per Δ for each modality. The companion modalities live
    # under <decoded_root>/<scan_id>/companion/<scan_id>-<m>.nii.gz and are
    # constant across Δ; t1c is the steered NIfTI from Phase A.
    anchor_dir = decoded_root / scan_id
    companion_dir = anchor_dir / "companion"
    seg_dir = anchor_dir / "seg"

    decoded_per_delta: dict[str, list[np.ndarray]] = {m: [] for m in _GRID_MODALITIES}
    seg_per_delta: list[np.ndarray] = []
    for di, delta in enumerate(deltas):
        # t1c (steered).
        t1c_path = anchor_dir / f"{scan_id}__delta_{float(delta):+.2f}.nii.gz"
        decoded_per_delta["t1c"].append(
            np.asarray(_nib.load(str(t1c_path)).dataobj, dtype=np.float32)
        )
        # Companion modalities (constant across Δ).
        for m in ("t1n", "t2f", "t2w"):
            comp_path = companion_dir / f"{scan_id}-{m}.nii.gz"
            decoded_per_delta[m].append(
                np.asarray(_nib.load(str(comp_path)).dataobj, dtype=np.float32)
            )
        # Predicted mask.
        wt_path = seg_dir / f"seg_delta_{float(delta):+.2f}.nii.gz"
        if wt_path.is_file():
            seg_per_delta.append(np.asarray(_nib.load(str(wt_path)).dataobj).astype(bool))
        else:
            seg_per_delta.append(np.zeros(gt_seg.shape, dtype=bool))

    # Build the 5 × (1 + n_d) figure.
    n_cols = 1 + len(deltas)
    n_rows = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.0 * n_rows), squeeze=False)

    fig.suptitle(
        f"E2.4 headline grid — bin B{bin_id}, scan {scan_id}, log V₀ = {log_v0:+.2f}",
        fontsize=11,
    )

    # Per-modality intensity windows from the GT (so all Δ images use the
    # same scale — visual change is interpretable).
    windows = {m: _percentile_window(gt_images[m]) for m in _GRID_MODALITIES}

    for r, m in enumerate(_GRID_MODALITIES):
        vmin, vmax = windows[m]
        # Column 0: ground truth source image.
        ax = axes[r, 0]
        ax.imshow(
            gt_images[m][:, :, z_slice].T,
            cmap="gray",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"{m} GT" if r == 0 else f"{m}", fontsize=8)
        ax.set_axis_off()
        for di, delta in enumerate(deltas):
            ax = axes[r, di + 1]
            ax.imshow(
                decoded_per_delta[m][di][:, :, z_slice].T,
                cmap="gray",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
            if r == 0:
                ax.set_title(f"Δ={float(delta):+.2f}", fontsize=8)
            ax.set_axis_off()

    # Mask row (row 4): GT mask in column 0, predicted mask elsewhere.
    ax = axes[4, 0]
    ax.imshow((gt_seg[:, :, z_slice] > 0).T, cmap="gray", origin="lower", vmin=0, vmax=1)
    ax.set_title("GT mask", fontsize=8)
    ax.set_axis_off()
    for di in range(len(deltas)):
        ax = axes[4, di + 1]
        ax.imshow(
            seg_per_delta[di][:, :, z_slice].T,
            cmap="gray",
            origin="lower",
            vmin=0,
            vmax=1,
        )
        ax.set_axis_off()

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def make_all_with_grids(
    sweep_h5_path: Path,
    source_h5_path: Path,
    output_dir: Path,
    *,
    grid_anchor_indices: tuple[int, ...] | None = None,
    decoded_root: Path | None = None,
) -> dict[str, Path]:
    """Phase-A plots + per-anchor headline grids (Phase B figure)."""
    import h5py

    artefacts = make_all(sweep_h5_path, output_dir)
    if grid_anchor_indices is None:
        with h5py.File(sweep_h5_path, "r") as f:
            bins = f["volume_bin"][:]
        small_idx = int(np.argmin(bins))
        big_idx = int(np.argmax(bins))
        grid_anchor_indices = (small_idx, big_idx)
    label_for = {
        idx: ("small" if idx == grid_anchor_indices[0] else "big")
        for idx in set(grid_anchor_indices)
    }
    with h5py.File(sweep_h5_path, "r") as f:
        bins = f["volume_bin"][:]
    for ai in grid_anchor_indices:
        bin_id = int(bins[ai])
        path = output_dir / f"headline_grid_{label_for[ai]}_B{bin_id}.png"
        make_modality_grid(
            sweep_h5_path=sweep_h5_path,
            source_h5_path=source_h5_path,
            anchor_index_in_sweep=ai,
            output_path=path,
            decoded_root=decoded_root,
        )
        artefacts[f"headline_grid_{label_for[ai]}_B{bin_id}"] = path
    return artefacts
