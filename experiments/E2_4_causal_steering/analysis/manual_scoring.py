"""Anatomical-plausibility montage builder.

For each anchor, lay out three axial slices through the tumor centroid at
``Δ log V ∈ {−1.5, 0, +1.5}`` (or the most extreme available negative / zero /
positive deltas). The resulting PNG is the artefact a radiologist scores
against the three E2.4 §4.5 criteria (location preserved; no contralateral /
distant tumor-like signal; no global brightness change).
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


def _pick_three_deltas(deltas: np.ndarray) -> tuple[int, int, int]:
    """Return indices of the most-negative, zero (or nearest), and most-positive."""
    most_neg = int(np.argmin(deltas))
    most_pos = int(np.argmax(deltas))
    zero_idx = int(np.argmin(np.abs(deltas)))
    return most_neg, zero_idx, most_pos


def _tumor_centroid_z(seg: np.ndarray) -> int:
    """Return the median axial slice index of the tumor (or D//2 if absent)."""
    nz = np.argwhere(seg > 0)
    if nz.size == 0:
        return seg.shape[2] // 2
    return int(np.median(nz[:, 2]))


def build_montage(
    sweep_h5_path: Path,
    output_path: Path,
    source_h5_path: Path | None = None,
) -> Path:
    """Stitch axial slices for every anchor at three deltas.

    Parameters
    ----------
    sweep_h5_path
        Phase-A sweep H5 (used for anchor list, NIfTI paths, mask label set).
    output_path
        PNG destination.
    source_h5_path
        Source H5; if supplied, the tumor centroid for each anchor is taken
        from the original segmentation. Otherwise the central axial slice is
        used.
    """
    sweep_h5_path = Path(sweep_h5_path)
    base_dir = sweep_h5_path.parent
    with h5py.File(sweep_h5_path, "r") as f:
        deltas = f["delta_log_v_grid"][:]
        nifti_paths = f["decoded_nifti_path"].asstr()[:]
        scan_ids = f["scan_id"].asstr()[:]
        bins = f["volume_bin"][:]
        anchor_rows = f.attrs.get("anchor_row_indices", None)

    di_neg, di_zero, di_pos = _pick_three_deltas(deltas)
    chosen_di = [di_neg, di_zero, di_pos]

    a = len(scan_ids)
    fig, axes = plt.subplots(a, 3, figsize=(9, 3 * a), squeeze=False)
    for ai in range(a):
        z_slice = None
        if source_h5_path is not None and anchor_rows is not None:
            with h5py.File(source_h5_path, "r") as src:
                seg_i = src["segmentations"][int(anchor_rows[ai])]
                z_slice = _tumor_centroid_z(seg_i)
        for col, di in enumerate(chosen_di):
            ax = axes[ai, col]
            rel = nifti_paths[ai, di]
            if not rel:
                ax.text(0.5, 0.5, "(no NIfTI)", ha="center", va="center")
                ax.set_axis_off()
                continue
            full = (base_dir / rel) if not Path(rel).is_absolute() else Path(rel)
            arr = np.asarray(nib.load(str(full)).dataobj, dtype=np.float32)
            z = z_slice if z_slice is not None else arr.shape[2] // 2
            z = max(0, min(arr.shape[2] - 1, z))
            ax.imshow(arr[:, :, z].T, cmap="gray", origin="lower")
            ax.set_title(f"B{int(bins[ai])} {scan_ids[ai]}\nΔ={float(deltas[di]):+.2f}", fontsize=8)
            ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
