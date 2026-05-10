"""MenGrowth → unified MenFlow HDF5.

MenGrowth is the in-house longitudinal meningioma cohort curated by the GICAI/UMA
group: 58 patients with 2-6 scans each, co-registered to the BraTS reference
frame so each scan has shape (240, 240, 155) at 1 mm isotropic spacing. The
modality set and label encoding match BraTS-MEN exactly; the only structural
difference is that some patients contribute multiple timepoints, so the cohort
is *longitudinal* and the CSR layout in :mod:`menflow.data.h5_converter` is
non-trivial.

Directory layout::

    MenGrowth-2025/
        MenGrowth-XXXX/
            MenGrowth-XXXX-YYY/
                {t1c, t1n, t2f, t2w, seg}.nii.gz
                synthseg.nii.gz   (ignored — brain parcellation)

A scan whose ``seg.nii.gz`` is all-zero (no visible tumor at that timepoint)
still counts as *annotated*: the file is present and the radiologist marked the
absence of a lesion. Downstream consumers can filter on ``segmentations.sum() > 0``.

Splits
------

The converter emits the canonical E3.1 splits ``splits/e3_train``,
``splits/e3_val`` and ``splits/e3_test`` (patient-grouped, log-volume
stratified). Patients with every scan reporting a sentinel-floor log-volume
(no visible tumor at any timepoint) are excluded from all splits — there is no
training signal to learn from. The legacy ``splits/{train, val}`` names are
**not** emitted by any MenFlow converter.
"""

from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Mapping
from pathlib import Path

import nibabel as nib
import numpy as np

from menflow.data.h5_converter import ConversionError, H5Converter, ScanData, ScanRecord
from menflow.data.kfold_splits import (
    KFoldLayout,
    compute_log_volume_voxels,
)

logger = logging.getLogger(__name__)


# Cohort-uniform geometry (BraTS reference frame).
_SHAPE: tuple[int, int, int] = (240, 240, 155)
_SPACING_MM: tuple[float, float, float] = (1.0, 1.0, 1.0)
_ORIENTATION = "LAS"

_MODALITIES: tuple[str, ...] = ("t1c", "t1n", "t2f", "t2w")

# Modern BraTS-MEN-RT 2024 label naming (NETC / SNFH / ET).
_LABEL_MAP: dict[int, str] = {
    0: "background",
    1: "non_enhancing_tumor_core",  # NETC
    2: "surrounding_non_enhancing_flair_hyperintensity",  # SNFH
    3: "enhancing_tumor",  # ET
}

_PATIENT_RE = re.compile(r"^MenGrowth-\d{4}$")
_SCAN_RE = re.compile(r"^(MenGrowth-\d{4})-(\d{3})$")


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


class MenGrowthConverter(H5Converter):
    """Converter for the in-house MenGrowth longitudinal meningioma cohort.

    Parameters
    ----------
    e3_split_seed, e3_n_strata, e3_n_folds, e3_test_fold, e3_val_fold
        Control the patient-grouped, log-volume-stratified split layout.
        Defaults match :class:`BraTSMENConverter`. The 58-patient cohort is
        smaller than BraTS-MEN, so the helper auto-clamps `n_strata` to keep
        every stratum bin populated by at least `n_folds` patients.
    """

    def __init__(
        self,
        *,
        kfold_k_values: tuple[int, ...] = (1, 3, 5),  # 10-fold needs 11+ patients
        kfold_test_pct: float = 0.1,
        kfold_seed: int = 42,
        kfold_n_strata: int = 3,
    ) -> None:
        self._scan_log_v: dict[str, float] = {}
        self._kfold_k_values = tuple(int(k) for k in kfold_k_values)
        self._kfold_test_pct = float(kfold_test_pct)
        self._kfold_seed = int(kfold_seed)
        self._kfold_n_strata = int(kfold_n_strata)

    @property
    def dataset_name(self) -> str:
        return "MenGrowth"

    @property
    def modalities(self) -> tuple[str, ...]:
        return _MODALITIES

    @property
    def label_map(self) -> dict[int, str]:
        return dict(_LABEL_MAP)

    @property
    def is_longitudinal(self) -> bool:
        return True

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        return _SHAPE

    @property
    def spacing_mm(self) -> tuple[float, float, float]:
        return _SPACING_MM

    @property
    def orientation(self) -> str:
        return _ORIENTATION

    def metadata_fields(self) -> dict[str, np.dtype]:
        # Per-scan flags: empty_seg (no tumor visible at this timepoint).
        return {"empty_seg": np.dtype("bool")}

    # ----- Discovery -----

    def discover_scans(self, roots: Mapping[str, Path]) -> list[ScanRecord]:
        """Walk patient/scan two-level hierarchy under ``roots['root']``.

        Records are sorted by (patient_id, timepoint) so the CSR longitudinal
        layout produced by the base class places one patient's scans contiguously.
        """
        if "root" not in roots:
            raise KeyError("MenGrowthConverter expects roots={'root': <path>}")
        root = Path(roots["root"])
        if not root.is_dir():
            raise FileNotFoundError(f"MenGrowth root does not exist: {root}")

        records: list[ScanRecord] = []
        for patient_dir in sorted(root.iterdir()):
            if not patient_dir.is_dir() or not _PATIENT_RE.match(patient_dir.name):
                continue
            for scan_dir in sorted(patient_dir.iterdir()):
                if not scan_dir.is_dir():
                    continue
                m = _SCAN_RE.match(scan_dir.name)
                if m is None or m.group(1) != patient_dir.name:
                    continue
                paths = _resolve_scan_files(scan_dir)
                if paths is None:
                    logger.warning("Skipping incomplete scan: %s", scan_dir)
                    continue
                records.append(
                    ScanRecord(
                        scan_id=scan_dir.name,
                        patient_id=patient_dir.name,
                        timepoint=int(m.group(2)),
                        extras={"paths": paths},
                    )
                )

        records.sort(key=lambda r: (r.patient_id, r.timepoint))
        n_pat = len({r.patient_id for r in records})
        logger.info("Discovered %d scans across %d patients", len(records), n_pat)
        return records

    # ----- Loading -----

    def load_scan(self, record: ScanRecord) -> ScanData:
        paths = record.extras["paths"]
        channels: list[np.ndarray] = []
        for modality in _MODALITIES:
            channels.append(_load_nifti(paths[modality]))
        image = np.stack(channels, axis=0).astype(np.float32, copy=False)

        seg: np.ndarray | None = None
        empty = True
        if "seg" in paths:
            seg_arr = _load_nifti(paths["seg"]).astype(np.int8, copy=False)
            seg_arr = _validate_labels(seg_arr, record.scan_id)
            seg = seg_arr
            empty = bool(seg_arr.sum() == 0)
            self._scan_log_v[record.scan_id] = compute_log_volume_voxels(
                seg_arr, label_set=tuple(k for k in _LABEL_MAP if k != 0)
            )
        else:
            self._scan_log_v[record.scan_id] = float(np.log(1e-3))
        return ScanData(image=image, segmentation=seg, metadata={"empty_seg": empty})

    # ----- Splits -----

    def build_kfold_splits(
        self, records: list[ScanRecord], patient_list: list[str]
    ) -> dict[int, KFoldLayout]:
        """Patient-grouped, log-volume-stratified k-fold splits.

        Aggregation is at the patient level: ``log_v_patient = max`` across
        timepoints. MenGrowth has no unannotated cohort, so no patient is
        filtered by subset; only the log_v floor (no-tumor sentinel) excludes
        patients.
        """
        # Local import to dodge isort/formatter stripping a top-level alias.
        from menflow.data.kfold_splits import build_kfold_splits as helper

        return helper(
            patient_list=patient_list,
            scan_to_patient=[(r.scan_id, r.patient_id) for r in records],
            scan_log_v=dict(self._scan_log_v),
            scan_subset={r.scan_id: "train" for r in records},  # all annotated
            k_values=self._kfold_k_values,
            test_pct=self._kfold_test_pct,
            seed=self._kfold_seed,
            n_strata=self._kfold_n_strata,
            excluded_subsets=(),
        )


# ---------------------------------------------------------------------------
# I/O helpers (parallel to brats_men.py — kept local to avoid coupling)
# ---------------------------------------------------------------------------


def _resolve_scan_files(scan_dir: Path) -> dict[str, Path] | None:
    """Return modality+seg paths using bare-name convention with prefixed fallback."""
    paths: dict[str, Path] = {}
    for modality in _MODALITIES:
        bare = scan_dir / f"{modality}.nii.gz"
        prefixed = scan_dir / f"{scan_dir.name}-{modality}.nii.gz"
        if bare.exists():
            paths[modality] = bare
        elif prefixed.exists():
            paths[modality] = prefixed
        else:
            return None
    seg_bare = scan_dir / "seg.nii.gz"
    seg_prefixed = scan_dir / f"{scan_dir.name}-seg.nii.gz"
    if seg_bare.exists():
        paths["seg"] = seg_bare
    elif seg_prefixed.exists():
        paths["seg"] = seg_prefixed
    return paths


def _load_nifti(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    arr = np.asarray(img.dataobj)
    if arr.shape != _SHAPE:
        raise ConversionError(f"Unexpected shape for {path.name}: {arr.shape}, expected {_SHAPE}")
    return arr


def _validate_labels(seg: np.ndarray, scan_id: str) -> np.ndarray:
    valid = set(_LABEL_MAP.keys())
    extra = set(np.unique(seg).tolist()) - valid
    if extra:
        raise ConversionError(
            f"Segmentation for {scan_id} contains unexpected labels: {sorted(extra)}"
        )
    return seg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cli() -> None:
    """Console entry point for ``menflow-convert-mengrowth``."""
    parser = argparse.ArgumentParser(
        description="Convert MenGrowth NIfTI cohort to the unified MenFlow H5 schema."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Path to MenGrowth-2025 directory (containing MenGrowth-XXXX subdirs)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output .h5 file")
    parser.add_argument(
        "--max-scans",
        type=int,
        default=None,
        help="Convert only the first N scans (dry run).",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="gzip compression level [1-9]. Default 4.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    MenGrowthConverter().convert(
        {"root": args.root},
        args.output,
        max_scans=args.max_scans,
        compression_level=args.compression_level,
    )


if __name__ == "__main__":
    cli()
