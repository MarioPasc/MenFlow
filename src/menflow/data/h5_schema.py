"""Declarative HDF5 schema for the unified MenFlow cohort format.

The schema is **dataset-agnostic**. Each concrete converter (BraTS-MEN, BraTS-GLI,
MenGrowth, ...) writes a file conforming to this contract and is validated against
it on creation. The schema captures:

- *what* modalities are present and their channel order,
- *how* labels map to clinical structures,
- whether the cohort is *cross-sectional* or *longitudinal* (CSR-encoded),
- per-scan provenance (scan id, patient id, timepoint),
- optional metadata and dataset-defined splits.

A single source of truth for the schema simplifies downstream consumers (E1
evaluator, flow-predictor dataloader, segmentation training): they only need to
trust :class:`H5Schema`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

logger = logging.getLogger(__name__)


SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Specification primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AttrSpec:
    """One required (or optional) root attribute of the H5 file."""

    name: str
    dtype: str  # "string" | "int" | "float" | "bool" | "string_array" | "int_array" | "float_array"
    required: bool = True
    description: str = ""


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """One required (or optional) dataset under the root or a subgroup."""

    path: str  # e.g. "images" or "longitudinal/patient_offsets"
    dtype_kind: str  # numpy dtype.kind: 'f','i','u','b','O' (string vlen)
    rank: int
    leading_dim: str | None = None  # which dim must equal n_scans / n_patients / n_patients+1
    required: bool = True
    description: str = ""


@dataclass(frozen=True, slots=True)
class GroupSpec:
    """One required group (e.g. ``metadata/``)."""

    path: str
    required: bool = True
    description: str = ""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class H5SchemaError(Exception):
    """Raised when an HDF5 file violates the unified MenFlow schema."""


@dataclass(frozen=True, slots=True)
class SchemaViolation:
    """One field-level violation encountered during validation."""

    path: str
    kind: str  # "missing", "dtype", "shape", "value"
    detail: str

    def __str__(self) -> str:  # pragma: no cover — trivial
        return f"[{self.kind}] {self.path}: {self.detail}"


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------


_REQUIRED_ATTRS: tuple[AttrSpec, ...] = (
    AttrSpec("schema_version", "string", description="Version of the MenFlow H5 schema."),
    AttrSpec("dataset_name", "string", description="Human-readable cohort name (e.g. BraTS-MEN)."),
    AttrSpec("dataset_type", "string", description="'cross-sectional' or 'longitudinal'."),
    AttrSpec("n_scans", "int", description="Total number of scans (rows of /images)."),
    AttrSpec("n_patients", "int", description="Total number of unique patients."),
    AttrSpec("modalities", "string_array", description="Channel-order modality names."),
    AttrSpec("n_modalities", "int", description="Number of modality channels per scan."),
    AttrSpec("spatial_shape", "int_array", description="(H, W, D) spatial shape of each scan."),
    AttrSpec("spacing_mm", "float_array", description="Cohort-level voxel spacing in mm (3,)."),
    AttrSpec("orientation", "string", description="Anatomical orientation code (e.g. RAS)."),
    AttrSpec(
        "label_map",
        "string",
        description="JSON-encoded mapping from int label to semantic name.",
    ),
    AttrSpec(
        "intensity_normalized",
        "bool",
        description="True if intensities are z-scored or rescaled at write-time.",
    ),
    AttrSpec("created_at", "string", description="ISO-8601 timestamp."),
    AttrSpec(
        "has_segmentation_any",
        "bool",
        description="True iff at least one scan in /segmentations is annotated.",
    ),
)


_REQUIRED_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec("images", "f", rank=5, leading_dim="n_scans"),
    DatasetSpec("segmentations", "i", rank=4, leading_dim="n_scans"),
    DatasetSpec("has_segmentation", "b", rank=1, leading_dim="n_scans"),
    DatasetSpec("scan_ids", "O", rank=1, leading_dim="n_scans"),
    DatasetSpec("patient_ids", "O", rank=1, leading_dim="n_scans"),
    DatasetSpec("timepoint_idx", "i", rank=1, leading_dim="n_scans"),
    DatasetSpec(
        "longitudinal/patient_offsets",
        "i",
        rank=1,
        leading_dim="n_patients_plus_one",
    ),
    DatasetSpec(
        "longitudinal/patient_list",
        "O",
        rank=1,
        leading_dim="n_patients",
    ),
)


_REQUIRED_GROUPS: tuple[GroupSpec, ...] = (
    GroupSpec("longitudinal"),
    GroupSpec("metadata"),  # may be empty (no patient covariates) but must exist
)


@dataclass(frozen=True, slots=True)
class H5Schema:
    """The unified MenFlow HDF5 schema. Pure data — no I/O state."""

    version: str = SCHEMA_VERSION
    required_attrs: tuple[AttrSpec, ...] = field(default_factory=lambda: _REQUIRED_ATTRS)
    required_datasets: tuple[DatasetSpec, ...] = field(
        default_factory=lambda: _REQUIRED_DATASETS
    )
    required_groups: tuple[GroupSpec, ...] = field(default_factory=lambda: _REQUIRED_GROUPS)

    # ----- Attribute helpers -----

    def attr_names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.required_attrs)

    def dataset_paths(self) -> tuple[str, ...]:
        return tuple(d.path for d in self.required_datasets)

    # ----- Validation -----

    def validate(self, h5_file: h5py.File) -> list[SchemaViolation]:
        """Return all schema violations of *h5_file*. Empty list iff fully compliant."""
        violations: list[SchemaViolation] = []
        violations.extend(self._validate_attrs(h5_file))
        violations.extend(self._validate_groups(h5_file))
        violations.extend(self._validate_datasets(h5_file))
        violations.extend(self._validate_cross_field(h5_file))
        return violations

    def assert_valid(self, h5_file: h5py.File) -> None:
        """Raise :class:`H5SchemaError` listing every violation, or return silently."""
        violations = self.validate(h5_file)
        if violations:
            joined = "\n  - ".join(str(v) for v in violations)
            raise H5SchemaError(
                f"H5 file does not satisfy MenFlow schema v{self.version}:\n  - {joined}"
            )

    # ----- Internals -----

    def _validate_attrs(self, f: h5py.File) -> list[SchemaViolation]:
        out: list[SchemaViolation] = []
        for spec in self.required_attrs:
            if spec.name not in f.attrs:
                if spec.required:
                    out.append(SchemaViolation(spec.name, "missing", "required attribute absent"))
                continue
            value = f.attrs[spec.name]
            if not _attr_dtype_ok(value, spec.dtype):
                out.append(
                    SchemaViolation(
                        spec.name,
                        "dtype",
                        f"expected {spec.dtype}, got {type(value).__name__}={value!r}",
                    )
                )
        # extra: validate label_map is JSON-decodable
        if "label_map" in f.attrs:
            raw = f.attrs["label_map"]
            try:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                json.loads(raw)
            except (ValueError, TypeError, AttributeError) as exc:
                out.append(SchemaViolation("label_map", "value", f"not valid JSON: {exc}"))
        # extra: dataset_type whitelist
        if "dataset_type" in f.attrs:
            v = _decode(f.attrs["dataset_type"])
            if v not in {"cross-sectional", "longitudinal"}:
                out.append(
                    SchemaViolation(
                        "dataset_type",
                        "value",
                        f"must be 'cross-sectional' or 'longitudinal', got {v!r}",
                    )
                )
        return out

    def _validate_groups(self, f: h5py.File) -> list[SchemaViolation]:
        out: list[SchemaViolation] = []
        for spec in self.required_groups:
            if spec.path not in f or not isinstance(f[spec.path], h5py.Group):
                if spec.required:
                    out.append(SchemaViolation(spec.path, "missing", "required group absent"))
        return out

    def _validate_datasets(self, f: h5py.File) -> list[SchemaViolation]:
        out: list[SchemaViolation] = []
        for spec in self.required_datasets:
            if spec.path not in f:
                if spec.required:
                    out.append(SchemaViolation(spec.path, "missing", "required dataset absent"))
                continue
            ds = f[spec.path]
            if not isinstance(ds, h5py.Dataset):
                out.append(SchemaViolation(spec.path, "missing", "expected Dataset, got Group"))
                continue
            if ds.dtype.kind != spec.dtype_kind:
                out.append(
                    SchemaViolation(
                        spec.path,
                        "dtype",
                        f"expected dtype.kind={spec.dtype_kind!r}, got {ds.dtype.kind!r}",
                    )
                )
            if ds.ndim != spec.rank:
                out.append(
                    SchemaViolation(
                        spec.path,
                        "shape",
                        f"expected rank {spec.rank}, got {ds.ndim} (shape={ds.shape})",
                    )
                )
        return out

    def _validate_cross_field(self, f: h5py.File) -> list[SchemaViolation]:
        """Cross-field invariants: leading dims, label range, modality count."""
        out: list[SchemaViolation] = []

        def _attr(name: str) -> Any | None:
            return f.attrs[name] if name in f.attrs else None

        n_scans = _attr("n_scans")
        n_patients = _attr("n_patients")
        n_modalities = _attr("n_modalities")
        spatial_shape = _attr("spatial_shape")
        modalities = _attr("modalities")

        if n_scans is None or n_patients is None:
            return out  # already reported as missing

        n_scans = int(n_scans)
        n_patients = int(n_patients)

        for spec in self.required_datasets:
            if spec.path not in f:
                continue
            ds = f[spec.path]
            if not isinstance(ds, h5py.Dataset):
                continue
            expected_lead: int | None
            if spec.leading_dim == "n_scans":
                expected_lead = n_scans
            elif spec.leading_dim == "n_patients":
                expected_lead = n_patients
            elif spec.leading_dim == "n_patients_plus_one":
                expected_lead = n_patients + 1
            else:
                expected_lead = None
            if expected_lead is not None and ds.shape[0] != expected_lead:
                out.append(
                    SchemaViolation(
                        spec.path,
                        "shape",
                        f"leading dim {ds.shape[0]} != {spec.leading_dim}={expected_lead}",
                    )
                )

        # /images shape check
        if "images" in f and isinstance(f["images"], h5py.Dataset):
            img = f["images"]
            if n_modalities is not None and img.shape[1] != int(n_modalities):
                out.append(
                    SchemaViolation(
                        "images",
                        "shape",
                        f"channel dim {img.shape[1]} != n_modalities={int(n_modalities)}",
                    )
                )
            if spatial_shape is not None and tuple(img.shape[2:]) != tuple(int(x) for x in spatial_shape):
                out.append(
                    SchemaViolation(
                        "images",
                        "shape",
                        f"spatial dims {img.shape[2:]} != spatial_shape={tuple(spatial_shape)}",
                    )
                )

        if modalities is not None and n_modalities is not None:
            mod_list = [_decode(m) for m in modalities]
            if len(mod_list) != int(n_modalities):
                out.append(
                    SchemaViolation(
                        "modalities",
                        "value",
                        f"len(modalities)={len(mod_list)} != n_modalities={int(n_modalities)}",
                    )
                )

        # CSR consistency: offsets monotonic, last == n_scans
        if "longitudinal/patient_offsets" in f:
            offsets = f["longitudinal/patient_offsets"][:]
            if len(offsets) >= 1:
                if offsets[0] != 0:
                    out.append(
                        SchemaViolation(
                            "longitudinal/patient_offsets",
                            "value",
                            f"first offset must be 0, got {offsets[0]}",
                        )
                    )
                if offsets[-1] != n_scans:
                    out.append(
                        SchemaViolation(
                            "longitudinal/patient_offsets",
                            "value",
                            f"last offset must equal n_scans={n_scans}, got {offsets[-1]}",
                        )
                    )
                if np.any(np.diff(offsets) < 0):
                    out.append(
                        SchemaViolation(
                            "longitudinal/patient_offsets",
                            "value",
                            "offsets are not non-decreasing",
                        )
                    )

        return out


# ---------------------------------------------------------------------------
# Top-level helpers
# ---------------------------------------------------------------------------


def validate_h5(path: str | Path, schema: H5Schema | None = None) -> list[SchemaViolation]:
    """Open *path* read-only and return the list of schema violations."""
    schema = schema or H5Schema()
    with h5py.File(path, "r") as f:
        return schema.validate(f)


def assert_h5_valid(path: str | Path, schema: H5Schema | None = None) -> None:
    """Raise :class:`H5SchemaError` if *path* does not satisfy the schema."""
    schema = schema or H5Schema()
    with h5py.File(path, "r") as f:
        schema.assert_valid(f)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind in {"S", "U"}:
        return str(value)
    return str(value)


def _attr_dtype_ok(value: Any, kind: str) -> bool:
    """Check that a stored H5 attribute matches the abstract type code."""
    if kind == "string":
        return isinstance(value, (str, bytes, np.bytes_, np.str_))
    if kind == "int":
        return isinstance(value, (int, np.integer)) or (
            isinstance(value, np.ndarray) and value.shape == () and value.dtype.kind in {"i", "u"}
        )
    if kind == "float":
        return isinstance(value, (float, np.floating)) or (
            isinstance(value, np.ndarray) and value.shape == () and value.dtype.kind == "f"
        )
    if kind == "bool":
        # h5py stores numpy bool scalars
        if isinstance(value, (bool, np.bool_)):
            return True
        if isinstance(value, np.ndarray) and value.shape == () and value.dtype.kind == "b":
            return True
        # numpy scalar 0/1 written by some writers: accept
        return False
    if kind == "string_array":
        if isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype.kind in {"O", "S", "U"}:
            return True
        return isinstance(value, Sequence) and all(isinstance(v, (str, bytes)) for v in value)
    if kind == "int_array":
        return isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype.kind in {"i", "u"}
    if kind == "float_array":
        return isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype.kind == "f"
    return False
