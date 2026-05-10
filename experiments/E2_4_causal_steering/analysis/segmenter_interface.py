"""Segmenter contract for E2.4 Phase B.

Phase A persists decoded NIfTIs and a ``sweep_results.h5`` whose
``decoded_log_v`` and ``tumor_voxels_decoded`` fields are sentinel-filled (NaN
and ``-1``). Phase B's only job is to:

1. Implement a concrete :class:`Segmenter` (e.g. wrapping a BraTS-MEN nnU-Net
   checkpoint).
2. Call :func:`fill_sweep_results`, which walks every NIfTI listed in the H5,
   runs the segmenter, and writes the measured volume back into the same file.

After Phase B, ``segmenter_completed`` is flipped to ``True`` and downstream
analysis (slope OLS, monotonicity, breakpoint saturation) becomes well-defined.
"""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import h5py
import nibabel as nib
import numpy as np

from menflow.latent_features.volume import voxel_count_to_log_volume_cm3

logger = logging.getLogger(__name__)


@runtime_checkable
class Segmenter(Protocol):
    """A binary tumor segmenter on a single-modality 3D MR image.

    Phase B implementations may wrap nnU-Net, MONAI, or a custom model. The
    contract is intentionally narrow so swapping segmenters is a one-line edit.

    Attributes
    ----------
    modality
        ``"t1c"``, ``"t1n"``, ``"t2f"``, ``"t2w"`` or ``"multi"`` (multimodal).
    expected_shape
        Image shape the segmenter accepts as ``(H, W, D)``. Resampling to this
        shape, if needed, is the segmenter's responsibility.
    """

    modality: str
    expected_shape: tuple[int, int, int]

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Return a binary mask of shape ``(H, W, D)`` aligned to ``image``."""
        ...


def predict_volume_log_cm3(
    segmenter: Segmenter,
    image: np.ndarray,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[float, int]:
    """Run a segmenter and return ``(log V cm³, voxel_count)``.

    The voxel count is from the binary mask post-prediction. Empty masks return
    ``(NaN, 0)`` so downstream analysis can decide how to handle saturation.
    """
    mask = segmenter.predict(image)
    if mask.shape != image.shape:
        raise ValueError(f"segmenter mask shape {mask.shape} != image shape {image.shape}")
    n_vox = int(mask.astype(bool).sum())
    if n_vox == 0:
        return float("nan"), 0
    return float(voxel_count_to_log_volume_cm3(n_vox, spacing_mm)), n_vox


def fill_sweep_results(
    sweep_h5_path: Path,
    segmenter: Segmenter,
    decoded_root: Path,
    spacing_mm: tuple[float, float, float] | None = None,
) -> None:
    """Walk the decoded NIfTIs and fill ``decoded_log_v`` / ``tumor_voxels_decoded``.

    Parameters
    ----------
    sweep_h5_path
        Provisional sweep H5 produced by :class:`SteerDecodeEngine`.
    segmenter
        Phase B implementation of the :class:`Segmenter` protocol.
    decoded_root
        Root directory containing the per-anchor NIfTI tree. NIfTI paths in the
        H5 are stored relative to ``sweep_h5_path.parent``; this argument is
        kept for callers that move the artefacts.
    spacing_mm
        Voxel spacing override. If ``None``, taken from the H5 ``spacing_mm``
        attribute.

    Notes
    -----
    The H5 is opened in append mode; existing fields ``decoded_log_v`` and
    ``tumor_voxels_decoded`` are overwritten in place. ``segmenter_completed``
    is set to ``True`` and ``segmenter_completed_at`` records the timestamp.
    """
    if not isinstance(segmenter, Segmenter):
        raise TypeError("segmenter does not satisfy the Segmenter protocol")
    sweep_h5_path = Path(sweep_h5_path)
    if not sweep_h5_path.is_file():
        raise FileNotFoundError(sweep_h5_path)
    base_dir = sweep_h5_path.parent

    with h5py.File(sweep_h5_path, "a") as f:
        if spacing_mm is None:
            spacing_mm = tuple(float(x) for x in f.attrs["spacing_mm"])  # type: ignore[assignment]
        nifti_paths = f["decoded_nifti_path"].asstr()[:]
        a, n_d = nifti_paths.shape
        decoded_log_v = f["decoded_log_v"][:]
        tumor_voxels = f["tumor_voxels_decoded"][:]

        for ai in range(a):
            for di in range(n_d):
                rel = str(nifti_paths[ai, di])
                if not rel:
                    continue
                full = (base_dir / rel) if not Path(rel).is_absolute() else Path(rel)
                if not full.is_file():
                    logger.warning("missing NIfTI %s; leaving sentinel", full)
                    continue
                img = nib.load(str(full))
                arr = np.asarray(img.dataobj, dtype=np.float32)
                log_v, n_vox = predict_volume_log_cm3(
                    segmenter,
                    arr,
                    spacing_mm=spacing_mm,  # type: ignore[arg-type]
                )
                decoded_log_v[ai, di] = np.float32(log_v)
                tumor_voxels[ai, di] = np.int32(n_vox)

        f["decoded_log_v"][...] = decoded_log_v
        f["tumor_voxels_decoded"][...] = tumor_voxels
        f.attrs["segmenter_completed"] = True
        f.attrs["segmenter_completed_at"] = _dt.datetime.now(_dt.UTC).isoformat()
        f.attrs["segmenter_modality"] = getattr(segmenter, "modality", "?")

    logger.info("Filled sweep_results at %s", sweep_h5_path)
