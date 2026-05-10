"""Abstract base class for HDF5 dataset conversors.

Concrete converters (BraTS-MEN, BraTS-GLI, MenGrowth, ...) implement a small set
of dataset-specific hooks (discovery, file loading, label semantics) and inherit
the standard write+validate machinery from :class:`H5Converter`. Every conversion
ends with :meth:`H5Converter._write_h5` invoking
:meth:`menflow.data.h5_schema.H5Schema.assert_valid`, so a non-conformant file
cannot reach disk in a "successful" state.

Design notes
------------

The schema does not prescribe runtime preprocessing (resampling, normalization).
Converters should default to *raw, reversible* storage: original BraTS spatial
shape and intensity scale, with metadata describing the orientation and spacing.
Downstream consumers (e.g., the E1 evaluator) apply MAISI-v2-specific
preprocessing themselves so the inverse transform is well-defined.
"""

from __future__ import annotations

import abc
import datetime as _dt
import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

from menflow.data.h5_schema import SCHEMA_VERSION, H5Schema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScanRecord:
    """Lightweight scan descriptor produced by :meth:`H5Converter.discover_scans`.

    Discovery should not load images. ``extras`` carries dataset-specific paths
    or annotations needed by :meth:`H5Converter.load_scan`.
    """

    scan_id: str
    patient_id: str
    timepoint: int
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScanData:
    """Loaded scan payload returned by :meth:`H5Converter.load_scan`."""

    image: np.ndarray  # shape (M, H, W, D), float32
    segmentation: np.ndarray | None  # shape (H, W, D), int8 or None
    metadata: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ConversionError(RuntimeError):
    """Raised when a single-scan conversion fails irrecoverably."""


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class H5Converter(abc.ABC):
    """Abstract base class for cohort-to-H5 converters.

    Subclasses implement
    :meth:`discover_scans`, :meth:`load_scan`, and the ``dataset_name``,
    ``modalities``, ``label_map``, ``is_longitudinal``, ``spatial_shape``,
    ``spacing_mm``, ``orientation`` properties. The :meth:`convert` driver
    handles the rest.
    """

    SCHEMA: H5Schema = H5Schema()

    # ----- Required class-level metadata -----

    @property
    @abc.abstractmethod
    def dataset_name(self) -> str:
        """Cohort identifier stored in /attrs/dataset_name."""

    @property
    @abc.abstractmethod
    def modalities(self) -> tuple[str, ...]:
        """Channel order of /images, e.g. ('t1c','t1n','t2f','t2w')."""

    @property
    @abc.abstractmethod
    def label_map(self) -> dict[int, str]:
        """Mapping from integer label to clinical structure name."""

    @property
    @abc.abstractmethod
    def is_longitudinal(self) -> bool:
        """True iff some patient has more than one scan."""

    @property
    @abc.abstractmethod
    def spatial_shape(self) -> tuple[int, int, int]:
        """Cohort-uniform spatial shape (H, W, D) of every scan."""

    @property
    @abc.abstractmethod
    def spacing_mm(self) -> tuple[float, float, float]:
        """Cohort-uniform voxel spacing in mm."""

    @property
    @abc.abstractmethod
    def orientation(self) -> str:
        """Anatomical orientation code (e.g. 'RAS')."""

    @property
    def intensity_normalized(self) -> bool:
        """Whether intensities are rescaled at write-time. Default False."""
        return False

    @property
    def n_modalities(self) -> int:
        return len(self.modalities)

    # ----- Required behaviour hooks -----

    @abc.abstractmethod
    def discover_scans(self, roots: Mapping[str, Path]) -> list[ScanRecord]:
        """Walk *roots* and return a sorted list of :class:`ScanRecord`.

        Sorting must place all scans of one patient contiguously, in increasing
        timepoint order; this is what enables the CSR longitudinal layout.
        """

    @abc.abstractmethod
    def load_scan(self, record: ScanRecord) -> ScanData:
        """Load one scan from disk and return image + (optional) segmentation."""

    # ----- Optional hook: compose dataset-defined splits -----

    def build_kfold_splits(
        self, records: list[ScanRecord], patient_list: list[str]
    ) -> dict[int, KFoldLayout]:  # noqa: F821 — forward ref to avoid cycle
        """Return ``{k: KFoldLayout}``; written to ``splits/kfold/``.

        Default: empty mapping (no splits). Subclasses that want native
        cross-validation splits should call
        :func:`menflow.data.kfold_splits.build_kfold_splits`.
        """
        return {}

    # ----- Optional hook: per-scan metadata fields to persist -----

    def metadata_fields(self) -> dict[str, np.dtype]:
        """Field name → numpy dtype for /metadata/<field> datasets.

        Returning an empty dict creates an empty /metadata/ group.
        """
        return {}

    # ----- Driver -----

    def convert(
        self,
        roots: Mapping[str, Path],
        output_path: str | Path,
        *,
        max_scans: int | None = None,
        compression: str = "gzip",
        compression_level: int = 4,
    ) -> Path:
        """Discover, load, write, and validate a full cohort.

        Parameters
        ----------
        roots
            Mapping from a converter-defined key (e.g. ``"train"``/``"val"``
            for BraTS-MEN's directory partition, or ``"root"`` for cohorts
            delivered as a single directory) to the root path. These keys
            describe **input directory layout**, not the H5 ``splits/*``
            datasets — those are produced by :meth:`build_splits` and the
            canonical names follow E3.1 (``e3_train``, ``e3_val``, ``e3_test``).
        output_path
            Destination ``.h5`` file. Parent directories are created.
        max_scans
            If set, only the first *max_scans* records are converted. Useful for
            dry runs.
        compression / compression_level
            Passed through to :meth:`h5py.Group.create_dataset`.

        Returns
        -------
        pathlib.Path
            The written output path (absolute).
        """
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records = self.discover_scans({k: Path(v) for k, v in roots.items()})
        logger.info("Discovered %d scans for %s", len(records), self.dataset_name)
        if max_scans is not None:
            records = records[:max_scans]
            logger.info("Limiting to first %d scans (dry run)", len(records))

        if not records:
            raise ConversionError(f"No scans discovered under {dict(roots)}")

        patient_list, patient_offsets, timepoint_indices = _build_csr_layout(records)

        self._write_h5(
            output_path,
            records,
            patient_list,
            patient_offsets,
            timepoint_indices,
            compression=compression,
            compression_level=compression_level,
        )

        from menflow.data.h5_schema import (
            assert_h5_valid,  # local import avoids cycle in re-imports
        )

        assert_h5_valid(output_path, self.SCHEMA)
        logger.info("Schema validation passed for %s", output_path)
        return output_path

    # ----- Internals -----

    def _write_h5(
        self,
        output_path: Path,
        records: list[ScanRecord],
        patient_list: list[str],
        patient_offsets: np.ndarray,
        timepoint_indices: np.ndarray,
        *,
        compression: str,
        compression_level: int,
    ) -> None:
        n_scans = len(records)
        n_patients = len(patient_list)
        H, W, D = self.spatial_shape
        M = self.n_modalities

        vlen_str = h5py.special_dtype(vlen=str)
        meta_fields = self.metadata_fields()
        patient_to_idx = {pid: i for i, pid in enumerate(patient_list)}

        logger.info(
            "Writing %s: %d scans, %d patients, shape=(%d,%d,%d,%d), modalities=%s",
            output_path,
            n_scans,
            n_patients,
            M,
            H,
            W,
            D,
            self.modalities,
        )

        n_errors = 0
        n_with_seg = 0

        with h5py.File(output_path, "w") as f:
            # --- Root attributes ---
            f.attrs["schema_version"] = SCHEMA_VERSION
            f.attrs["dataset_name"] = str(self.dataset_name)
            f.attrs["dataset_type"] = "longitudinal" if self.is_longitudinal else "cross-sectional"
            f.attrs["n_scans"] = np.int64(n_scans)
            f.attrs["n_patients"] = np.int64(n_patients)
            f.attrs["modalities"] = np.asarray(list(self.modalities), dtype=object)
            f.attrs["n_modalities"] = np.int64(M)
            f.attrs["spatial_shape"] = np.asarray([H, W, D], dtype=np.int64)
            f.attrs["spacing_mm"] = np.asarray(self.spacing_mm, dtype=np.float64)
            f.attrs["orientation"] = str(self.orientation)
            f.attrs["label_map"] = json.dumps({int(k): str(v) for k, v in self.label_map.items()})
            f.attrs["intensity_normalized"] = bool(self.intensity_normalized)
            f.attrs["created_at"] = _dt.datetime.now(_dt.UTC).isoformat()
            f.attrs["has_segmentation_any"] = False  # filled below

            # --- Datasets ---
            images = f.create_dataset(
                "images",
                shape=(n_scans, M, H, W, D),
                dtype="float32",
                chunks=(1, M, H, W, D),
                compression=compression,
                compression_opts=compression_level,
            )
            segs = f.create_dataset(
                "segmentations",
                shape=(n_scans, H, W, D),
                dtype="int8",
                chunks=(1, H, W, D),
                compression=compression,
                compression_opts=compression_level,
            )
            has_seg = f.create_dataset("has_segmentation", shape=(n_scans,), dtype="bool")
            f.create_dataset(
                "scan_ids",
                data=np.asarray([r.scan_id for r in records], dtype=object),
                dtype=vlen_str,
            )
            f.create_dataset(
                "patient_ids",
                data=np.asarray([r.patient_id for r in records], dtype=object),
                dtype=vlen_str,
            )
            f.create_dataset("timepoint_idx", data=timepoint_indices.astype(np.int32))

            # --- Longitudinal CSR ---
            long_grp = f.create_group("longitudinal")
            long_grp.create_dataset("patient_offsets", data=patient_offsets.astype(np.int32))
            long_grp.create_dataset(
                "patient_list",
                data=np.asarray(patient_list, dtype=object),
                dtype=vlen_str,
            )

            # --- Metadata ---
            meta_grp = f.create_group("metadata")
            meta_arrays: dict[str, np.ndarray] = {}
            for name, dtype in meta_fields.items():
                if dtype.kind == "O":
                    arr = np.array(["" for _ in range(n_scans)], dtype=object)
                elif dtype.kind == "f":
                    arr = np.full(n_scans, np.nan, dtype=dtype)
                elif dtype.kind in {"i", "u"}:
                    arr = np.full(n_scans, -1, dtype=dtype)
                elif dtype.kind == "b":
                    arr = np.zeros(n_scans, dtype=dtype)
                else:
                    arr = np.zeros(n_scans, dtype=dtype)
                meta_arrays[name] = arr

            # --- Per-scan loop ---
            for i, rec in enumerate(tqdm(records, desc=f"Converting {self.dataset_name}")):
                try:
                    payload = self.load_scan(rec)
                except Exception as exc:  # noqa: BLE001 — log and continue
                    logger.error("Failed to load %s: %s", rec.scan_id, exc)
                    n_errors += 1
                    continue

                _validate_payload(payload, expected_modalities=M, expected_shape=(H, W, D))
                images[i] = payload.image.astype(np.float32, copy=False)

                if payload.segmentation is None:
                    segs[i] = np.zeros((H, W, D), dtype=np.int8)
                    has_seg[i] = False
                else:
                    segs[i] = payload.segmentation.astype(np.int8, copy=False)
                    has_seg[i] = True
                    n_with_seg += 1

                for name, arr in meta_arrays.items():
                    if name in payload.metadata:
                        arr[i] = payload.metadata[name]

            # Persist accumulated metadata arrays after the scan loop
            for name, arr in meta_arrays.items():
                if arr.dtype.kind == "O":
                    meta_grp.create_dataset(name, data=arr, dtype=vlen_str)
                else:
                    meta_grp.create_dataset(name, data=arr)

            # --- Splits (k-fold layout, written under /splits/kfold/) ---
            layouts = self.build_kfold_splits(records, patient_list)
            if layouts:
                from menflow.data.kfold_splits import write_kfold_to_h5

                splits_grp = f.create_group("splits")
                kfold_grp = splits_grp.create_group("kfold")
                write_kfold_to_h5(kfold_grp, layouts)

            # Final flag update
            f.attrs["has_segmentation_any"] = bool(n_with_seg > 0)

        size_gb = output_path.stat().st_size / (1024**3)
        logger.info(
            "Wrote %s (%.2f GB). %d/%d scans had segmentation; %d errors.",
            output_path.name,
            size_gb,
            n_with_seg,
            n_scans,
            n_errors,
        )


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _validate_payload(
    payload: ScanData, *, expected_modalities: int, expected_shape: tuple[int, int, int]
) -> None:
    if payload.image.ndim != 4:
        raise ConversionError(f"image must be (M,H,W,D), got {payload.image.shape}")
    if payload.image.shape[0] != expected_modalities:
        raise ConversionError(
            f"image channel dim {payload.image.shape[0]} != expected {expected_modalities}"
        )
    if tuple(payload.image.shape[1:]) != expected_shape:
        raise ConversionError(
            f"image spatial shape {payload.image.shape[1:]} != expected {expected_shape}"
        )
    if payload.segmentation is not None and tuple(payload.segmentation.shape) != expected_shape:
        raise ConversionError(
            f"segmentation shape {payload.segmentation.shape} != expected {expected_shape}"
        )


def _build_csr_layout(
    records: Iterable[ScanRecord],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Build (patient_list, patient_offsets, timepoint_indices) from ordered records."""
    patient_list: list[str] = []
    patient_offsets: list[int] = [0]
    timepoint_indices: list[int] = []

    current = None
    tp_counter = 0
    n_seen = 0
    for rec in records:
        if rec.patient_id != current:
            if current is not None:
                patient_offsets.append(n_seen)
            patient_list.append(rec.patient_id)
            current = rec.patient_id
            tp_counter = 0
        timepoint_indices.append(tp_counter)
        tp_counter += 1
        n_seen += 1
    patient_offsets.append(n_seen)

    return (
        patient_list,
        np.asarray(patient_offsets, dtype=np.int32),
        np.asarray(timepoint_indices, dtype=np.int32),
    )
