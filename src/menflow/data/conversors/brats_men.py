"""BraTS-MEN 2023 → unified MenFlow HDF5.

The cohort is preoperative meningioma MRI from the ASNR-MICCAI BraTS 2023
Meningioma sub-challenge (LaBella et al., 2023, arXiv:2305.07642). Each subject
contributes one scan with four co-registered modalities (T1c, T1n, T2f, T2w);
the training subset adds an expert-annotated tumor mask (labels 1=NCR, 2=ED,
3=ET), the validation subset omits it.

Storage policy
--------------

BraTS-MEN data is delivered already preprocessed: skull-stripped, defaced,
co-registered to SRI24 with 1 mm isotropic spacing and uniform 240×240×155
shape. We therefore store the raw arrays at native resolution with original
intensities. This keeps the H5 reversible-by-default; the E1 evaluator (and any
other consumer) applies its own MAISI-v2-specific preprocessing on read.

Cohort fusion
-------------

Training (n≈1000) and validation (n≈141) subjects are concatenated into a
single file; the ``metadata/subset`` field distinguishes ``"train"`` (annotated)
from ``"val"`` (unannotated) on a per-scan basis. ``has_segmentation`` mirrors
the same partition.

Splits
------

The converter emits **only** the patient-grouped, log-volume-stratified splits
required by E3.1 §4.2 — ``splits/e3_train`` (≈80 %), ``splits/e3_val`` (≈10 %)
and ``splits/e3_test`` (≈10 %). All three are computed from the annotated
training cohort exclusively; the unannotated validation cohort is **not**
present in any split. Stratification is by E3.1 volume bin (5 quantiles of log
tumor volume aggregated to patient level) using
``sklearn.model_selection.StratifiedKFold`` with a fixed seed.

The legacy ``splits/{train, val}`` (BraTS challenge cohort partition) is **no
longer emitted**. Files produced before this change must be migrated via
``menflow.data.migrate_e3_splits`` before the e3_* splits are usable downstream.

Optional clinical metadata (grade, age, sex) is parsed from ``*.xlsx`` files
adjacent to the cohort root; missing values are encoded as ``-1`` / ``NaN`` /
``"unknown"``.
"""

from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Mapping
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

from menflow.data.features import (
    FeatureRegistry,
    FeatureSpec,
    laterality_from_mask,
)
from menflow.data.h5_converter import ConversionError, H5Converter, ScanData, ScanRecord
from menflow.data.kfold_splits import (
    KFoldLayout,
    compute_log_volume_voxels,
)

logger = logging.getLogger(__name__)


# Cohort-uniform geometry. SRI24 atlas, BraTS preprocessing.
_BRATS_SHAPE: tuple[int, int, int] = (240, 240, 155)
_BRATS_SPACING_MM: tuple[float, float, float] = (1.0, 1.0, 1.0)
_BRATS_ORIENTATION = "LAS"  # SRI24 default; we do not reorient on write.

_MODALITIES: tuple[str, ...] = ("t1c", "t1n", "t2f", "t2w")

# BraTS-MEN labels (LaBella et al., 2023):
_LABEL_MAP: dict[int, str] = {
    0: "background",
    1: "necrotic_tumor_core",  # NCR
    2: "peritumoral_edema",  # ED
    3: "enhancing_tumor",  # ET
}

_SUBJECT_RE = re.compile(r"^BraTS-MEN-(\d{5})-(\d{3})$")


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


class BraTSMENConverter(H5Converter):
    """Converter for the ASNR-MICCAI BraTS-MEN 2023 challenge cohort.

    Parameters
    ----------
    metadata_xlsx
        Optional path to a clinical metadata sheet. If present, is parsed once
        on construction and joined into per-scan metadata fields.
    """

    def __init__(
        self,
        metadata_xlsx: str | Path | None = None,
        *,
        kfold_k_values: tuple[int, ...] = (1, 3, 5, 10),
        kfold_test_pct: float = 0.1,
        kfold_seed: int = 42,
        kfold_n_strata: int = 5,
    ) -> None:
        self._metadata_xlsx = Path(metadata_xlsx) if metadata_xlsx is not None else None
        self._metadata_table: dict[str, dict[str, object]] = {}
        if self._metadata_xlsx is not None:
            self._metadata_table = _load_metadata_xlsx(self._metadata_xlsx)
        self._train_pids: set[str] = set()
        self._val_pids: set[str] = set()
        # Per-scan log-volume cache, populated in load_scan and consumed in
        # build_kfold_splits to drive patient-level stratification.
        self._scan_log_v: dict[str, float] = {}
        self._kfold_k_values = tuple(int(k) for k in kfold_k_values)
        self._kfold_test_pct = float(kfold_test_pct)
        self._kfold_seed = int(kfold_seed)
        self._kfold_n_strata = int(kfold_n_strata)

    # ----- Static schema metadata -----

    @property
    def dataset_name(self) -> str:
        return "BraTS-MEN-2023"

    @property
    def modalities(self) -> tuple[str, ...]:
        return _MODALITIES

    @property
    def label_map(self) -> dict[int, str]:
        return dict(_LABEL_MAP)

    @property
    def is_longitudinal(self) -> bool:
        return False

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        return _BRATS_SHAPE

    @property
    def spacing_mm(self) -> tuple[float, float, float]:
        return _BRATS_SPACING_MM

    @property
    def orientation(self) -> str:
        return _BRATS_ORIENTATION

    def metadata_fields(self) -> dict[str, np.dtype]:
        return {
            "grade": np.dtype("int8"),
            "age": np.dtype("float32"),
            "sex": np.dtype("O"),
            "subset": np.dtype("O"),  # "train" | "val"
        }

    # ----- One-shot features attached at build time -----

    def feature_registry(self) -> FeatureRegistry:
        """Cheap, dataset-specific features attached to /features/."""
        return FeatureRegistry(
            dataset_name=self.dataset_name,
            features=(
                FeatureSpec(
                    name="n_voxels_tumor",
                    dtype="int64",
                    shape=(),
                    units="voxels",
                    description=(
                        "Number of tumor voxels (labels 1-3) in the expert segmentation; "
                        "0 for unannotated validation scans."
                    ),
                    source="derived:segmentation",
                ),
                FeatureSpec(
                    name="log_volume_cm3",
                    dtype="float32",
                    shape=(),
                    units="log(cm^3)",
                    description=(
                        "log( n_voxels * prod(spacing_mm) / 1000 ) for the tumor mask; "
                        "NaN for unannotated scans."
                    ),
                    source="derived:segmentation",
                ),
                FeatureSpec(
                    name="laterality",
                    dtype="O",
                    shape=(),
                    units="",
                    description=(
                        "L/R/B side classification from the tumor centroid; empty string "
                        "when no segmentation is available."
                    ),
                    source="derived:segmentation",
                ),
                FeatureSpec(
                    name="grade",
                    dtype="int8",
                    shape=(),
                    units="",
                    description="WHO meningioma grade (1/2/3) or -1 if unknown.",
                    source="external:clinical_xlsx",
                ),
                FeatureSpec(
                    name="age",
                    dtype="float32",
                    shape=(),
                    units="years",
                    description="Patient age at imaging in years; NaN if unknown.",
                    source="external:clinical_xlsx",
                ),
                FeatureSpec(
                    name="sex",
                    dtype="O",
                    shape=(),
                    units="",
                    description=(
                        "Patient self-reported sex as recorded in the BraTS-MEN clinical "
                        'sheet; "unknown" if absent.'
                    ),
                    source="external:clinical_xlsx",
                ),
            ),
        )

    def compute_features(
        self, records: list[ScanRecord], h5_file: h5py.File
    ) -> dict[str, np.ndarray]:
        """Compute cheap per-scan features from the just-written H5."""
        n_scans = len(records)
        n_voxels = np.zeros(n_scans, dtype=np.int64)
        log_v = np.full(n_scans, np.nan, dtype=np.float32)
        lat = np.empty(n_scans, dtype=object)
        grade = np.full(n_scans, -1, dtype=np.int8)
        age = np.full(n_scans, np.nan, dtype=np.float32)
        sex = np.empty(n_scans, dtype=object)

        spacing = np.prod(self.spacing_mm)
        tumor_labels = tuple(k for k in _LABEL_MAP if k != 0)

        segs = h5_file["segmentations"]
        has_seg = h5_file["has_segmentation"][:]

        for i, rec in enumerate(records):
            meta = self._metadata_for(rec.patient_id)
            grade[i] = int(meta["grade"])
            age[i] = float(meta["age"])
            sex[i] = str(meta["sex"])
            if has_seg[i]:
                seg = segs[i]
                tumor = np.isin(seg, tumor_labels)
                vox = int(tumor.sum())
                n_voxels[i] = vox
                if vox > 0:
                    log_v[i] = float(np.log(vox * spacing / 1000.0))
                lat[i] = laterality_from_mask(tumor)
            else:
                lat[i] = ""

        return {
            "n_voxels_tumor": n_voxels,
            "log_volume_cm3": log_v,
            "laterality": lat,
            "grade": grade,
            "age": age,
            "sex": sex,
        }

    # ----- Discovery -----

    def discover_scans(self, roots: Mapping[str, Path]) -> list[ScanRecord]:
        """Walk *roots* and emit one record per BraTS-MEN subject directory.

        Recognised keys: ``"train"``, ``"val"``. Either may be omitted. The
        record's ``extras`` carries the file paths of the four modalities and,
        if present, the segmentation; ``subset`` distinguishes train/val for
        downstream split construction.
        """
        records: list[ScanRecord] = []

        for subset_name in ("train", "val"):
            if subset_name not in roots:
                continue
            root = Path(roots[subset_name])
            if not root.is_dir():
                raise FileNotFoundError(f"{subset_name} root does not exist: {root}")
            logger.info("Scanning %s subset at %s", subset_name, root)
            for sub_dir in sorted(root.iterdir()):
                if not sub_dir.is_dir():
                    continue
                m = _SUBJECT_RE.match(sub_dir.name)
                if m is None:
                    continue
                patient_id = f"BraTS-MEN-{m.group(1)}"
                timepoint = int(m.group(2))
                paths = _resolve_subject_files(sub_dir)
                if paths is None:
                    logger.warning("Skipping incomplete subject: %s", sub_dir.name)
                    continue
                records.append(
                    ScanRecord(
                        scan_id=sub_dir.name,
                        patient_id=patient_id,
                        timepoint=timepoint,
                        extras={"paths": paths, "subset": subset_name},
                    )
                )
                if subset_name == "train":
                    self._train_pids.add(patient_id)
                else:
                    self._val_pids.add(patient_id)

        records.sort(key=lambda r: (r.patient_id, r.timepoint))
        logger.info(
            "Discovered %d records (train=%d, val=%d)",
            len(records),
            len(self._train_pids),
            len(self._val_pids),
        )
        return records

    # ----- Loading -----

    def load_scan(self, record: ScanRecord) -> ScanData:
        paths = record.extras["paths"]
        subset = record.extras["subset"]
        image_channels: list[np.ndarray] = []
        for modality in _MODALITIES:
            arr = _load_nifti(paths[modality])
            image_channels.append(arr)
        image = np.stack(image_channels, axis=0).astype(np.float32, copy=False)

        seg: np.ndarray | None
        if "seg" in paths:
            seg_arr = _load_nifti(paths["seg"]).astype(np.int8, copy=False)
            seg_arr = _validate_labels(seg_arr, record.scan_id)
            seg = seg_arr
            self._scan_log_v[record.scan_id] = compute_log_volume_voxels(
                seg, label_set=tuple(k for k in _LABEL_MAP if k != 0)
            )
        else:
            seg = None
            # Sentinel matches routines/compute_features convention so the
            # build_splits filter `log_v > -6.0` excludes the unannotated cohort.
            self._scan_log_v[record.scan_id] = float(np.log(1e-3))

        meta = self._metadata_for(record.patient_id)
        meta["subset"] = subset

        return ScanData(image=image, segmentation=seg, metadata=meta)

    # ----- Splits -----

    def build_kfold_splits(
        self, records: list[ScanRecord], patient_list: list[str]
    ) -> dict[int, KFoldLayout]:
        """Patient-level, log-volume-stratified k-fold splits.

        Produces ``splits/kfold/k{N}/`` for every ``N`` in
        ``self._kfold_k_values``. The unannotated BraTS challenge validation
        cohort (``subset == "val"``) is excluded from all splits.
        """
        # Local import to dodge isort/formatter stripping a top-level alias
        # that would otherwise clash with the method name.
        from menflow.data.kfold_splits import build_kfold_splits as helper

        return helper(
            patient_list=patient_list,
            scan_to_patient=[(r.scan_id, r.patient_id) for r in records],
            scan_log_v=dict(self._scan_log_v),
            scan_subset={r.scan_id: r.extras["subset"] for r in records},
            k_values=self._kfold_k_values,
            test_pct=self._kfold_test_pct,
            seed=self._kfold_seed,
            n_strata=self._kfold_n_strata,
        )

    # ----- Internals -----

    def _metadata_for(self, patient_id: str) -> dict[str, object]:
        row = self._metadata_table.get(patient_id, {})
        return {
            "grade": int(row.get("grade", -1)),
            "age": float(row.get("age", float("nan"))),
            "sex": str(row.get("sex", "unknown")),
        }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _resolve_subject_files(sub_dir: Path) -> dict[str, Path] | None:
    """Return modality+seg paths for a BraTS-MEN subject directory.

    Returns ``None`` if any modality is missing.
    """
    sid = sub_dir.name
    paths: dict[str, Path] = {}
    for modality in _MODALITIES:
        p = sub_dir / f"{sid}-{modality}.nii.gz"
        if not p.exists():
            return None
        paths[modality] = p
    seg_path = sub_dir / f"{sid}-seg.nii.gz"
    if seg_path.exists():
        paths["seg"] = seg_path
    return paths


def _load_nifti(path: Path) -> np.ndarray:
    """Load a NIfTI volume as a numpy array, dtype float32 unless seg."""
    img = nib.load(str(path))
    arr = np.asarray(img.dataobj)
    if arr.shape != _BRATS_SHAPE:
        raise ConversionError(
            f"Unexpected shape for {path.name}: {arr.shape}, expected {_BRATS_SHAPE}"
        )
    return arr


def _validate_labels(seg: np.ndarray, scan_id: str) -> np.ndarray:
    """Ensure segmentation values are within the declared label set."""
    valid = set(_LABEL_MAP.keys())
    unique = set(np.unique(seg).tolist())
    extra = unique - valid
    if extra:
        raise ConversionError(
            f"Segmentation for {scan_id} contains unexpected labels: {sorted(extra)}"
        )
    return seg


def _load_metadata_xlsx(xlsx: Path) -> dict[str, dict[str, object]]:
    """Parse a BraTS-MEN clinical metadata spreadsheet.

    The schema of these sheets is not fixed across releases. We perform a tolerant
    column-name match: the patient ID column must contain "id"/"brats", and the
    fields ``grade``/``age``/``sex`` are matched by case-insensitive substring.
    Missing or unparseable rows fall back to defaults.
    """
    try:
        import openpyxl
    except ImportError:  # pragma: no cover — openpyxl is a hard dep
        logger.warning("openpyxl not installed; metadata ignored")
        return {}

    if not xlsx.exists():
        logger.warning("Metadata file not found: %s", xlsx)
        return {}

    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    try:
        header = [str(h).strip() if h is not None else "" for h in next(rows)]
    except StopIteration:
        return {}

    h_lower = [h.lower() for h in header]
    id_col = next((i for i, h in enumerate(h_lower) if "id" in h or "brats" in h), None)
    grade_col = next((i for i, h in enumerate(h_lower) if "grade" in h), None)
    age_col = next((i for i, h in enumerate(h_lower) if "age" in h), None)
    sex_col = next((i for i, h in enumerate(h_lower) if "sex" in h or "gender" in h), None)
    if id_col is None:
        logger.warning("No ID column found in %s; metadata ignored", xlsx)
        return {}

    out: dict[str, dict[str, object]] = {}
    for row in rows:
        sid = row[id_col]
        if sid is None:
            continue
        sid = str(sid).strip()
        # Some sheets store the scan-id form; collapse to patient id.
        m = _SUBJECT_RE.match(sid)
        patient_id = f"BraTS-MEN-{m.group(1)}" if m else sid
        rec: dict[str, object] = {}
        if grade_col is not None and row[grade_col] is not None:
            try:
                rec["grade"] = int(row[grade_col])
            except (ValueError, TypeError):
                pass
        if age_col is not None and row[age_col] is not None:
            try:
                rec["age"] = float(row[age_col])
            except (ValueError, TypeError):
                pass
        if sex_col is not None and row[sex_col] is not None:
            rec["sex"] = str(row[sex_col]).strip()
        out[patient_id] = rec
    wb.close()
    logger.info("Loaded clinical metadata for %d patients from %s", len(out), xlsx.name)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cli() -> None:
    """Console entry point for ``menflow-convert-brats-men``."""
    parser = argparse.ArgumentParser(
        description="Convert BraTS-MEN 2023 NIfTI cohort to the unified MenFlow H5 schema."
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=None,
        help="Path to ASNR-MICCAI-BraTS2023-MEN-Challenge-TrainingData/BraTS-MEN-Train",
    )
    parser.add_argument(
        "--val-root",
        type=Path,
        default=None,
        help="Path to ASNR-MICCAI-BraTS2023-MEN-Challenge-ValidationData",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output .h5 file")
    parser.add_argument(
        "--metadata-xlsx",
        type=Path,
        default=None,
        help="Optional clinical metadata spreadsheet (grade/age/sex).",
    )
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

    if args.train_root is None and args.val_root is None:
        parser.error("at least one of --train-root / --val-root is required")

    roots: dict[str, Path] = {}
    if args.train_root is not None:
        roots["train"] = args.train_root
    if args.val_root is not None:
        roots["val"] = args.val_root

    converter = BraTSMENConverter(metadata_xlsx=args.metadata_xlsx)
    converter.convert(
        roots,
        args.output,
        max_scans=args.max_scans,
        compression_level=args.compression_level,
    )


if __name__ == "__main__":
    cli()
