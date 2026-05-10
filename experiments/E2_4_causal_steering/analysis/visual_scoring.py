"""Wide-Δ visual scoring figure for E2.4 Phase B.

Built after the BraTS25_1 segmenter proved incompatible with MAISI-decoded
t1c (Phase B segmenter failure). The user acts as the segmenter via visual
inspection: per anchor, the figure shows the mid-tumor axial slice of the
decoded t1c at every Δ in a horizontal strip, with the Δ=0 column
overlaid with the ground-truth tumor contour as a reference. Tumor growth /
shrinkage along the strip is then a qualitative validation of the steering
operator.

Two outputs per call:

* one **super-figure** with one row per anchor and one column per Δ;
* one **strip per anchor** (small, easier to view).
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)


def _tumor_centroid_z(seg: np.ndarray) -> int:
    nz = np.argwhere(seg > 0)
    if nz.size == 0:
        return seg.shape[2] // 2
    return int(np.median(nz[:, 2]))


def _percentile_window(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, lo)), float(np.percentile(finite, hi))


def _draw_strip(
    ax_row,
    *,
    decoded_per_delta: list[np.ndarray],
    deltas: np.ndarray,
    z_slice: int,
    vmin: float,
    vmax: float,
    gt_mask_slice: np.ndarray | None,
    zero_idx: int,
    title_anchor: str,
) -> None:
    for ci, delta in enumerate(deltas):
        ax = ax_row[ci]
        ax.imshow(
            decoded_per_delta[ci][:, :, z_slice].T,
            cmap="gray",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        if ci == zero_idx and gt_mask_slice is not None:
            # Overlay GT contour as a green outline.
            ax.contour(
                gt_mask_slice.T,
                levels=[0.5],
                colors=["#00ff00"],
                linewidths=1.0,
                origin="lower",
            )
        ax.set_axis_off()
        if ci == 0:
            ax.set_title(title_anchor, fontsize=8, loc="left")
        ax.text(
            0.5,
            -0.05,
            f"Δ={float(delta):+.1f}",
            ha="center",
            va="top",
            fontsize=7,
            color="black",
            transform=ax.transAxes,
        )


def make_wide_visual_grid(
    sweep_h5_path: Path,
    source_h5_path: Path,
    output_path: Path,
    *,
    decoded_root: Path | None = None,
    per_anchor_dir: Path | None = None,
) -> dict[str, Path]:
    """Render the wide-Δ visual grid (5 rows × n_deltas cols) + per-anchor strips.

    Parameters
    ----------
    sweep_h5_path
        ``sweep_results.h5`` from the wide-Δ steer_decode run.
    source_h5_path
        BraTS-MEN unified source H5 (used for the ground-truth mask + slice
        index per anchor).
    output_path
        Path of the super-figure PNG.
    decoded_root
        Root holding ``<scan_id>/<scan_id>__delta_{:+.2f}.nii.gz``. Defaults
        to ``sweep_h5_path.parent / "decoded"``.
    per_anchor_dir
        Optional directory to drop per-anchor strip PNGs into; one file per
        anchor named ``wide_visual_<scan_id>_B<bin>.png``. Defaults to
        ``output_path.parent / "wide_visual_per_anchor"``.

    Returns
    -------
    dict[str, pathlib.Path]
        ``{"super_figure": <path>, "<scan_id>": <strip_path>, ...}``.
    """
    sweep_h5_path = Path(sweep_h5_path)
    output_path = Path(output_path)
    if decoded_root is None:
        decoded_root = sweep_h5_path.parent / "decoded"
    if per_anchor_dir is None:
        per_anchor_dir = output_path.parent / "wide_visual_per_anchor"
    per_anchor_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(sweep_h5_path, "r") as sw:
        deltas = sw["delta_log_v_grid"][:].astype(np.float32)
        scan_ids = list(sw["scan_id"].asstr()[:])
        anchor_rows = [int(x) for x in sw.attrs["anchor_row_indices"]]
        bins = sw["volume_bin"][:]
        log_v0 = sw["log_v0"][:]

    # Sort columns by Δ ascending so left = most negative.
    order = np.argsort(deltas)
    deltas_sorted = deltas[order]
    zero_idx = int(np.argmin(np.abs(deltas_sorted)))

    # Read each anchor's decoded NIfTIs in Δ-order and the GT slice.
    per_anchor: list[dict] = []
    for ai, (scan_id, anchor_row) in enumerate(zip(scan_ids, anchor_rows, strict=True)):
        anchor_dir = decoded_root / scan_id
        with h5py.File(source_h5_path, "r") as src:
            seg = src["segmentations"][anchor_row].astype(np.int32)
            src_modalities = [
                (m.decode() if isinstance(m, bytes) else str(m)) for m in src.attrs["modalities"]
            ]
            t1c_idx = src_modalities.index("t1c")
            real_t1c = src["images"][anchor_row, t1c_idx].astype(np.float32)
        z_slice = _tumor_centroid_z(seg)
        vmin, vmax = _percentile_window(real_t1c)
        decoded_per_delta = []
        for delta in deltas_sorted:
            p = anchor_dir / f"{scan_id}__delta_{float(delta):+.2f}.nii.gz"
            decoded_per_delta.append(np.asarray(nib.load(str(p)).dataobj, dtype=np.float32))
        per_anchor.append(
            {
                "scan_id": scan_id,
                "anchor_row": anchor_row,
                "bin": int(bins[ai]),
                "log_v0": float(log_v0[ai]),
                "z_slice": z_slice,
                "vmin": vmin,
                "vmax": vmax,
                "gt_mask_slice": (seg > 0)[:, :, z_slice].astype(np.uint8),
                "decoded_per_delta": decoded_per_delta,
            }
        )

    n_anchors = len(per_anchor)
    n_cols = len(deltas_sorted)

    # ----- Super-figure -----
    fig_w = max(20.0, 1.3 * n_cols)
    fig_h = max(8.0, 1.5 * n_anchors)
    fig, axes = plt.subplots(n_anchors, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.suptitle(
        "E2.4 Phase B — wide-Δ visual scoring (BraTS25_1 incompatible with MAISI t1c)",
        fontsize=10,
    )
    for ai, info in enumerate(per_anchor):
        title = f"B{info['bin']} {info['scan_id']}\nlog V₀={info['log_v0']:+.2f}"
        _draw_strip(
            axes[ai],
            decoded_per_delta=info["decoded_per_delta"],
            deltas=deltas_sorted,
            z_slice=info["z_slice"],
            vmin=info["vmin"],
            vmax=info["vmax"],
            gt_mask_slice=info["gt_mask_slice"],
            zero_idx=zero_idx,
            title_anchor=title,
        )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    logger.info("Super-figure -> %s", output_path)

    # ----- Per-anchor strips -----
    artefacts: dict[str, Path] = {"super_figure": output_path}
    for info in per_anchor:
        strip_path = per_anchor_dir / f"wide_visual_{info['scan_id']}_B{info['bin']}.png"
        fig_w_strip = max(20.0, 1.3 * n_cols)
        fig_strip, ax_strip = plt.subplots(1, n_cols, figsize=(fig_w_strip, 2.0), squeeze=False)
        title = f"B{info['bin']} {info['scan_id']}, log V₀={info['log_v0']:+.2f}"
        _draw_strip(
            ax_strip[0],
            decoded_per_delta=info["decoded_per_delta"],
            deltas=deltas_sorted,
            z_slice=info["z_slice"],
            vmin=info["vmin"],
            vmax=info["vmax"],
            gt_mask_slice=info["gt_mask_slice"],
            zero_idx=zero_idx,
            title_anchor=title,
        )
        fig_strip.tight_layout()
        fig_strip.savefig(strip_path, dpi=140)
        plt.close(fig_strip)
        logger.info("Per-anchor strip -> %s", strip_path)
        artefacts[info["scan_id"]] = strip_path

    return artefacts
