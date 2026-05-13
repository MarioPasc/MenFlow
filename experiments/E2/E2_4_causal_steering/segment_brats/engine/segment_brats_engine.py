"""Generic BraTS-MEN segmentation routine.

Two input modes:

* **Directory mode** (``input_dir``): the dir contains one subdirectory per
  subject, each holding ``<scan_id>-{t1c,t1n,t2f,t2w}.nii.gz`` at canonical
  (240, 240, 155). Used for E2.4 staged dirs and for the BraTS-MEN-2023 raw
  release.
* **H5 mode** (``input_h5``): the routine extracts per-subject NIfTIs into a
  scratch dir first (mirrors the BraTS-MEN unified schema). Useful for
  pseudo-labeling the val set straight from the unified H5.

Per-subject the engine:

1. Validates that all 4 modalities are present and shape-canonical.
2. Calls :class:`experiments.E2._lib.segmentation.docker_runner.BratsDockerRunner` against
   a single-subject staging dir.
3. Reduces the raw multi-class prediction to a binary whole-tumor mask via
   :func:`experiments.E2._lib.segmentation.output.wt_mask_from_prediction` and saves it
   alongside the raw prediction.
4. Records duration, voxel count, and paths in a per-run ``manifest.json``.

The routine is the launchable counterpart of E2.4 Phase B but is dataset-
agnostic; the val-set pseudo-labeling pipeline plugs in by pointing at the
raw BraTS-MEN val NIfTI tree.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import nibabel as nib
import numpy as np
import yaml

from experiments.E2._lib.segmentation.docker_runner import BratsDockerRunner
from experiments.E2._lib.segmentation.models import BRATS_MODELS, get_model
from experiments.E2._lib.segmentation.output import wt_mask_from_prediction

logger = logging.getLogger(__name__)


SEGMENT_SCHEMA_VERSION = "0.1"
BRATS_MODALITIES: tuple[str, ...] = ("t1c", "t1n", "t2f", "t2w")
BRATS_CANONICAL_SHAPE: tuple[int, int, int] = (240, 240, 155)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SegmentBratsRoutineConfig:
    """Routine configuration.

    Attributes
    ----------
    output_dir
        Root for ``predictions/<scan_id>/{seg_<scan_id>.nii.gz, raw/}`` and
        ``manifest.json``.
    model
        Model id (key of :data:`experiments.E2._lib.segmentation.models.BRATS_MODELS`).
    input_dir
        Directory of per-subject NIfTI dirs. Mutually exclusive with
        ``input_h5``.
    input_h5
        Unified-schema H5 to extract per-subject NIfTIs from. Mutually
        exclusive with ``input_dir``.
    h5_modality_order
        Channel order of ``images`` axis 1 in ``input_h5``. Required when
        ``input_h5`` is set.
    h5_split
        Optional name of a split to restrict to (e.g. ``"val"``).
    gpu
        Try to pass through GPU (``--gpus all``).
    limit
        Cap subjects for smoke runs.
    include_subject_glob
        Optional glob filter (e.g. ``"BraTS-MEN-*"``).
    timeout_s
        Per-subject docker-run timeout.
    log_level
        Python logging level.
    keep_extracted_h5_dir
        If True (default False), keep the temp NIfTI extraction dir after the
        run for debugging.
    """

    output_dir: Path
    model: str
    input_dir: Path | None = None
    input_h5: Path | None = None
    h5_modality_order: tuple[str, ...] | None = None
    h5_split: str | None = None
    gpu: bool = True
    limit: int | None = None
    include_subject_glob: str | None = None
    timeout_s: float | None = 1800.0
    log_level: str = "INFO"
    keep_extracted_h5_dir: bool = False

    @classmethod
    def from_yaml(cls, path: Path | str) -> SegmentBratsRoutineConfig:
        """Load config from YAML; coerce path-like fields."""
        with open(path) as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        for k in ("slurm",):
            raw.pop(k, None)
        for path_key in ("output_dir", "input_dir", "input_h5"):
            if path_key in raw and raw[path_key] is not None:
                raw[path_key] = Path(str(raw[path_key])).expanduser()
        if "h5_modality_order" in raw and raw["h5_modality_order"] is not None:
            raw["h5_modality_order"] = tuple(raw["h5_modality_order"])
        return cls(**raw)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SegmentBratsEngine:
    """Drive a per-subject BraTS segmentation pass over a directory or H5."""

    def __init__(self, config: SegmentBratsRoutineConfig) -> None:
        self.config = config
        if config.model not in BRATS_MODELS:
            raise ValueError(f"unknown model {config.model!r}; choose from {sorted(BRATS_MODELS)}")

    def run(self) -> Path:
        cfg = self.config
        if (cfg.input_dir is None) == (cfg.input_h5 is None):
            raise ValueError(
                "exactly one of input_dir / input_h5 must be set "
                f"(got input_dir={cfg.input_dir}, input_h5={cfg.input_h5})"
            )
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        predictions_dir = cfg.output_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        spec = get_model(cfg.model)
        runner = BratsDockerRunner(spec, gpu=cfg.gpu)
        runner.ensure_image()

        if cfg.input_h5 is not None:
            scratch_root, subject_dirs = self._extract_h5(cfg.input_h5)
        else:
            assert cfg.input_dir is not None
            scratch_root = None
            subject_dirs = self._discover_input_subjects(cfg.input_dir)

        if cfg.limit is not None:
            subject_dirs = subject_dirs[: cfg.limit]
        logger.info(
            "Running %s on %d subjects; output -> %s",
            cfg.model,
            len(subject_dirs),
            predictions_dir,
        )

        manifest_entries: list[dict[str, Any]] = []
        for i, subject_dir in enumerate(subject_dirs):
            scan_id = subject_dir.name
            logger.info("[%d/%d] %s", i + 1, len(subject_dirs), scan_id)
            entry = self._segment_one(
                scan_id=scan_id,
                subject_dir=subject_dir,
                runner=runner,
                spec=spec,
                predictions_dir=predictions_dir,
            )
            manifest_entries.append(entry)

        if scratch_root is not None and not cfg.keep_extracted_h5_dir:
            shutil.rmtree(scratch_root, ignore_errors=True)

        manifest_path = cfg.output_dir / "manifest.json"
        manifest = {
            "schema_version": SEGMENT_SCHEMA_VERSION,
            "created_at": _dt.datetime.now(_dt.UTC).isoformat(),
            "model": dataclasses.asdict(spec),
            "config": _jsonable(dataclasses.asdict(cfg)),
            "n_subjects": len(manifest_entries),
            "subjects": manifest_entries,
        }
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        logger.info("Wrote manifest -> %s", manifest_path)
        return manifest_path

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _segment_one(
        self,
        *,
        scan_id: str,
        subject_dir: Path,
        runner: BratsDockerRunner,
        spec: Any,
        predictions_dir: Path,
    ) -> dict[str, Any]:
        cfg = self.config
        for m in BRATS_MODALITIES:
            f = subject_dir / f"{scan_id}-{m}.nii.gz"
            if not f.is_file():
                raise FileNotFoundError(f"missing modality {m} for {scan_id}: expected {f}")
        # Shape sanity.
        ref = nib.load(str(subject_dir / f"{scan_id}-t1c.nii.gz"))
        shp = tuple(int(x) for x in ref.shape)
        if shp != BRATS_CANONICAL_SHAPE:
            logger.warning(
                "%s has non-canonical shape %s (expected %s); container may reject",
                scan_id,
                shp,
                BRATS_CANONICAL_SHAPE,
            )

        subject_pred_dir = predictions_dir / scan_id
        raw_dir = subject_pred_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        nifti_paths, elapsed = runner.run(subject_dir, raw_dir, timeout_s=cfg.timeout_s)
        if not nifti_paths:
            raise RuntimeError(f"{scan_id}: container exited 0 but wrote no NIfTI")

        # The container can write multiple files; pick the first as the seg.
        # BraTS25 conventionally writes one file named like
        # `<scan_id>.nii.gz`; we accept whatever shape matches the input.
        raw_seg_path = nifti_paths[0]
        seg_arr = np.asarray(nib.load(str(raw_seg_path)).dataobj)
        wt = wt_mask_from_prediction(seg_arr, spec.label_map_name)
        wt_path = subject_pred_dir / f"seg_{scan_id}.nii.gz"
        nib.save(
            nib.Nifti1Image(wt.astype(np.uint8), nib.load(str(raw_seg_path)).affine),
            str(wt_path),
        )

        n_vox = int(wt.sum())
        n_classes_present = sorted({int(v) for v in np.unique(seg_arr)})
        return {
            "scan_id": scan_id,
            "input_dir": str(subject_dir),
            "raw_seg": str(raw_seg_path),
            "wt_seg": str(wt_path),
            "wt_voxel_count": n_vox,
            "raw_classes": n_classes_present,
            "elapsed_s": float(elapsed),
        }

    def _discover_input_subjects(self, input_dir: Path) -> list[Path]:
        cfg = self.config
        if not input_dir.is_dir():
            raise FileNotFoundError(input_dir)
        candidates = sorted(p for p in input_dir.iterdir() if p.is_dir())
        if cfg.include_subject_glob:
            from fnmatch import fnmatch

            candidates = [p for p in candidates if fnmatch(p.name, cfg.include_subject_glob)]
        return candidates

    def _extract_h5(self, input_h5: Path) -> tuple[Path, list[Path]]:
        cfg = self.config
        if cfg.h5_modality_order is None:
            raise ValueError("h5_modality_order required when input_h5 is set")
        for m in BRATS_MODALITIES:
            if m not in cfg.h5_modality_order:
                raise ValueError(
                    f"BraTS-required modality {m!r} not in h5_modality_order "
                    f"{cfg.h5_modality_order}"
                )
        scratch = Path(tempfile.mkdtemp(prefix="menflow_segment_brats_"))
        scratch_nifti = scratch / "nifti"
        scratch_nifti.mkdir(parents=True, exist_ok=True)
        logger.info("Extracting H5 -> %s", scratch_nifti)
        subject_dirs: list[Path] = []
        with h5py.File(input_h5, "r") as f:
            scan_ids = [_decode(s) for s in f["scan_ids"][:]]
            indices: list[int]
            if cfg.h5_split is not None:
                if "splits" not in f or cfg.h5_split not in f["splits"]:
                    raise KeyError(f"split {cfg.h5_split!r} not in {input_h5}")
                offsets = f["longitudinal/patient_offsets"][:].astype(np.int64)
                patient_idx = f["splits"][cfg.h5_split][:].astype(np.int64)
                idx_set: set[int] = set()
                for p in patient_idx:
                    idx_set.update(range(int(offsets[p]), int(offsets[p + 1])))
                indices = sorted(idx_set)
            else:
                indices = list(range(len(scan_ids)))

            spacing = tuple(float(x) for x in f.attrs.get("spacing_mm", (1.0, 1.0, 1.0)))
            affine = np.eye(4, dtype=np.float64)
            for axis, sp in enumerate(spacing):
                affine[axis, axis] = sp
            modality_order = list(cfg.h5_modality_order)

            for i in indices:
                scan_id = scan_ids[i]
                d = scratch_nifti / scan_id
                d.mkdir(parents=True, exist_ok=True)
                image = f["images"][i]  # (M, H, W, D)
                for m in BRATS_MODALITIES:
                    m_idx = modality_order.index(m)
                    vol = image[m_idx].astype(np.float32)
                    nib.save(
                        nib.Nifti1Image(vol, affine),
                        str(d / f"{scan_id}-{m}.nii.gz"),
                    )
                subject_dirs.append(d)
        return scratch, subject_dirs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind in {"S", "U", "O"}:
        try:
            return _decode(value.item())
        except Exception:  # pragma: no cover
            return str(value)
    return str(value)


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
