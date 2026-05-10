"""Phase B orchestrator for E2.4 causal steering.

Combines:

* :func:`menflow.segmentation.companion_decode.decode_companion_modalities`
  to produce the t1n / t2f / t2w companion NIfTIs at Δ=0 (one per anchor),
* :class:`experiments.E2_4_causal_steering.analysis.segmenter_brats.BratsDockerSegmenter`
  to run the BraTS Docker container on a per-(anchor, Δ) staging dir,
* :func:`fill_sweep_results_multimodal` to write the measured ``log V``
  values back into the Phase A ``sweep_results.h5``.

Output layout (extends Phase A's ``decoded/<scan_id>/`` subtree):

::

    decoded/<scan_id>/
    ├── <scan_id>__delta_{:+.2f}.nii.gz   # Phase A: steered t1c (kept)
    ├── companion/<scan_id>-{t1n,t2f,t2w}.nii.gz
    ├── staged/delta_{:+.2f}/<scan_id>-{t1c,t1n,t2f,t2w}.nii.gz   # symlinks
    └── seg/
        ├── seg_delta_{:+.2f}.nii.gz                              # binary WT
        └── raw/seg_delta_{:+.2f}.nii.gz                          # raw multi-class

The orchestrator is idempotent: existing companion NIfTIs and staging dirs
are left in place; only missing pieces are produced.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

from experiments.E2_4_causal_steering.analysis.segmenter_brats import (
    BratsDockerSegmenter,
)
from menflow.latent_features.volume import voxel_count_to_log_volume_cm3
from menflow.maisi_autoencoder.config import MaisiV2Config
from menflow.segmentation.companion_decode import decode_companion_modalities

logger = logging.getLogger(__name__)


COMPANION_MODALITIES: tuple[str, ...] = ("t1n", "t2f", "t2w")
STEERED_MODALITY: str = "t1c"
ALL_MODALITIES: tuple[str, ...] = (STEERED_MODALITY, *COMPANION_MODALITIES)


# Canonical BraTS-MEN affine (LPS, 1×1×1 mm) — matches the raw release used to
# train the BraTS challenge containers. Decoded NIfTIs must be re-headered to
# this orientation or the segmenter mis-orients the volume and predicts
# all-zero. Origin (0, 239, 0) reflects nibabel's convention where the y-flip
# offsets the origin so the world bbox lines up with the canonical raw files.
BRATS_CANONICAL_AFFINE: np.ndarray = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 239.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True, slots=True)
class PhaseBConfig:
    """Configuration for a Phase B run.

    Attributes
    ----------
    sweep_h5
        Phase A ``sweep_results.h5`` (will be modified in place).
    decoded_root
        Root that contains ``<scan_id>/<scan_id>__delta_{:+.2f}.nii.gz`` from
        Phase A. Companion / staged / seg subtrees are created under each
        ``<scan_id>/``.
    latents_h5
        Path of the latents H5 used by Phase A.
    checkpoint
        MAISI-v2 checkpoint path.
    model_id
        BraTS model id (default ``BraTS25_1``).
    work_dir
        Scratch dir for docker outputs (auto-created).
    gpu
        Try GPU passthrough.
    timeout_s
        Per-subject docker-run timeout.
    dtype
        Decode dtype for companion modalities (matches Phase A's t1c dtype).
    device
        Torch device for the companion decode.
    direction_sign
        Optional sign multiplier applied to the Phase A direction.
        Use ``-1`` to invert the steering convention; the routine does NOT
        re-decode t1c — it only re-segments. To actually flip the steered
        latent you must re-run :class:`SteerDecodeEngine` with the negated
        NPZ; this knob is here only to record the convention used in the
        sweep H5 attrs.
    h5_modality_order
        Channel order of ``latents`` axis 1 in the latents H5.
    """

    sweep_h5: Path
    decoded_root: Path
    latents_h5: Path
    checkpoint: Path
    model_id: str = "BraTS25_1"
    work_dir: Path | None = None
    gpu: bool = True
    timeout_s: float | None = 1800.0
    dtype: str = "float16"
    device: str = "cuda"
    direction_sign: int = 1
    h5_modality_order: tuple[str, ...] = ALL_MODALITIES


def run_phase_b(cfg: PhaseBConfig) -> Path:
    """Decode companions, segment every (anchor, Δ), fill the sweep H5.

    Returns the sweep H5 path (now with finite ``decoded_log_v``).
    """
    cfg.decoded_root.mkdir(parents=True, exist_ok=True)
    work_dir = cfg.work_dir if cfg.work_dir is not None else cfg.decoded_root / "_work_docker"
    work_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(cfg.sweep_h5, "r") as f:
        scan_ids = list(f["scan_id"].asstr()[:])
        anchor_rows = [int(x) for x in f.attrs["anchor_row_indices"]]
        deltas = f["delta_log_v_grid"][:]
        spacing = tuple(float(x) for x in f.attrs["spacing_mm"])

    logger.info(
        "Phase B: %d anchors × %d Δ; spacing=%s; model=%s",
        len(scan_ids),
        len(deltas),
        spacing,
        cfg.model_id,
    )

    # Step 1: companion decode per anchor (idempotent).
    companion_indices = _resolve_companion_indices(cfg.h5_modality_order)
    for scan_id, anchor_idx in zip(scan_ids, anchor_rows, strict=True):
        anchor_dir = cfg.decoded_root / scan_id
        companion_dir = anchor_dir / "companion"
        if _all_companions_exist(companion_dir, scan_id):
            logger.info("[%s] companion modalities already present; skipping decode", scan_id)
            continue
        companion_dir.mkdir(parents=True, exist_ok=True)
        decode_companion_modalities(
            latents_h5=cfg.latents_h5,
            anchor_index=anchor_idx,
            scan_id=scan_id,
            modality_indices=[companion_indices[m] for m in COMPANION_MODALITIES],
            modality_names=list(COMPANION_MODALITIES),
            output_dir=companion_dir,
            checkpoint=cfg.checkpoint,
            model_config=MaisiV2Config(),
            dtype=cfg.dtype,
            device=cfg.device,
            intensity_rescale=True,
        )

    # Step 2: stage per (anchor, Δ) and segment.
    segmenter = BratsDockerSegmenter(
        cfg.model_id, work_dir=work_dir, gpu=cfg.gpu, timeout_s=cfg.timeout_s
    )

    decoded_log_v = np.full((len(scan_ids), len(deltas)), np.nan, dtype=np.float32)
    tumor_voxels = np.full((len(scan_ids), len(deltas)), -1, dtype=np.int32)

    for ai, scan_id in enumerate(scan_ids):
        anchor_dir = cfg.decoded_root / scan_id
        seg_dir = anchor_dir / "seg"
        raw_seg_dir = seg_dir / "raw"
        seg_dir.mkdir(parents=True, exist_ok=True)
        raw_seg_dir.mkdir(parents=True, exist_ok=True)

        for di, delta in enumerate(deltas):
            staged_dir = anchor_dir / "staged" / f"delta_{float(delta):+.2f}"
            subject_dir = _stage_subject_dir(
                anchor_dir=anchor_dir,
                staged_dir=staged_dir,
                scan_id=scan_id,
                delta=float(delta),
            )
            wt_path = seg_dir / f"seg_delta_{float(delta):+.2f}.nii.gz"
            if wt_path.is_file():
                logger.info(
                    "[%s Δ=%+0.2f] already segmented; reusing %s",
                    scan_id,
                    float(delta),
                    wt_path,
                )
                wt_arr = np.asarray(nib.load(str(wt_path)).dataobj).astype(bool)
            else:
                wt_arr, raw_seg_path = segmenter.predict_from_subject_dir(
                    subject_dir, keep_raw_dir=raw_seg_dir
                )
                # Move raw to predictable name.
                target_raw = raw_seg_dir / f"seg_delta_{float(delta):+.2f}.nii.gz"
                if raw_seg_path.resolve() != target_raw.resolve():
                    raw_seg_path.replace(target_raw)
                # Save the binary WT mask alongside.
                ref = nib.load(str(target_raw))
                nib.save(
                    nib.Nifti1Image(wt_arr.astype(np.uint8), ref.affine),
                    str(wt_path),
                )

            n_vox = int(wt_arr.sum())
            tumor_voxels[ai, di] = n_vox
            if n_vox == 0:
                decoded_log_v[ai, di] = np.float32(np.nan)
            else:
                decoded_log_v[ai, di] = np.float32(voxel_count_to_log_volume_cm3(n_vox, spacing))

    # Step 3: write back to the sweep H5.
    with h5py.File(cfg.sweep_h5, "a") as f:
        f["decoded_log_v"][...] = decoded_log_v
        f["tumor_voxels_decoded"][...] = tumor_voxels
        f.attrs["segmenter_completed"] = True
        f.attrs["segmenter_completed_at"] = _dt.datetime.now(_dt.UTC).isoformat()
        f.attrs["segmenter_model_id"] = cfg.model_id
        f.attrs["segmenter_companion_strategy"] = "decoded_at_delta_0"
        f.attrs["segmenter_direction_sign"] = np.int64(cfg.direction_sign)
        f.attrs["phase"] = "B_segmenter_completed"

    # Step 4: side-car summary.
    summary = {
        "created_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "config": _jsonable(dataclasses.asdict(cfg)),
        "n_anchors": len(scan_ids),
        "n_deltas": int(len(deltas)),
        "tumor_voxels_decoded": tumor_voxels.tolist(),
        "decoded_log_v": [
            [None if np.isnan(v) else float(v) for v in row] for row in decoded_log_v
        ],
    }
    summary_path = cfg.sweep_h5.parent / "analysis" / "phase_b_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Phase B summary -> %s", summary_path)
    return cfg.sweep_h5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_companion_indices(
    modality_order: tuple[str, ...],
) -> Mapping[str, int]:
    """Return {modality_name: column_index} for the latents H5 layout."""
    out = {}
    for m in ALL_MODALITIES:
        if m not in modality_order:
            raise ValueError(
                f"required modality {m!r} not in latents modality_order={modality_order}"
            )
        out[m] = modality_order.index(m)
    return out


def _all_companions_exist(companion_dir: Path, scan_id: str) -> bool:
    return all((companion_dir / f"{scan_id}-{m}.nii.gz").is_file() for m in COMPANION_MODALITIES)


def _stage_subject_dir(
    *,
    anchor_dir: Path,
    staged_dir: Path,
    scan_id: str,
    delta: float,
) -> Path:
    """Build a per-(anchor, Δ) BraTS-formatted dir; return the subject dir.

    Layout::

        staged_dir/
        └── <scan_id>/
            ├── <scan_id>-t1c.nii.gz   # symlink to steered t1c
            ├── <scan_id>-t1n.nii.gz   # symlink to companion
            ├── <scan_id>-t2f.nii.gz   # symlink to companion
            └── <scan_id>-t2w.nii.gz   # symlink to companion

    BraTS containers mount ``staged_dir`` (the parent) at ``/input`` and
    expect to find a subject sub-directory inside.
    """
    subject_dir = staged_dir / scan_id
    subject_dir.mkdir(parents=True, exist_ok=True)
    steered_t1c = anchor_dir / f"{scan_id}__delta_{delta:+.2f}.nii.gz"
    if not steered_t1c.is_file():
        raise FileNotFoundError(
            f"missing Phase A steered t1c NIfTI for {scan_id} Δ={delta:+.2f}: {steered_t1c}"
        )
    companion_dir = anchor_dir / "companion"
    targets = {
        "t1c": steered_t1c,
        "t1n": companion_dir / f"{scan_id}-t1n.nii.gz",
        "t2f": companion_dir / f"{scan_id}-t2f.nii.gz",
        "t2w": companion_dir / f"{scan_id}-t2w.nii.gz",
    }
    for m, src in targets.items():
        if not src.is_file():
            raise FileNotFoundError(f"missing {m} for {scan_id} (expected at {src})")
        link = subject_dir / f"{scan_id}-{m}.nii.gz"
        if link.is_symlink() or link.exists():
            link.unlink()
        # Materialize a BraTS-canonical NIfTI: re-header to LPS affine and
        # clip negative intensities (MAISI decoder leaks negatives that put
        # the image out of the segmenter's training distribution and cause
        # all-zero predictions). The container needs files actually present
        # inside the bind-mounted staging tree, so we don't symlink.
        arr = np.asarray(nib.load(str(src.resolve())).dataobj, dtype=np.float32)
        arr = np.clip(arr, 0.0, None).astype(np.float32, copy=False)
        nib.save(nib.Nifti1Image(arr, BRATS_CANONICAL_AFFINE), str(link))
    return subject_dir


def _jsonable(obj: object) -> object:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return obj
