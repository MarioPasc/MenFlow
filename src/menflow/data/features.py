"""Per-cohort feature registry for the unified MenFlow HDF5.

Each converter (BraTS-MEN, MenGrowth, ...) declares which derived features it
attaches to the unified H5 at build time, and how to compute them. The registry
is dataset-specific but the storage layout, attribute schema, and validator are
shared so downstream consumers can introspect any cohort's features without
hard-coding cohort knowledge.

Storage layout
--------------

Features live under the optional ``/features/`` group of the unified H5.
Each feature is one dataset whose path is ``/features/<name>``. The dataset
carries the following attributes describing its semantic content:

- ``units``: physical or counting units (``"cm^3"``, ``"voxels"``, ``""`` for
  dimensionless / categorical).
- ``description``: one-sentence semantic meaning.
- ``source``: provenance tag, typically ``"derived:segmentation"``,
  ``"derived:image"``, or ``"external:<filename>"``.
- ``dtype``: stored numpy dtype string (informational; the HDF5 dataset's own
  dtype is authoritative).

The ``/features/`` group itself stores the registry JSON in
``attrs['registry_json']`` so the file is self-describing without external
context. ``attrs['schema_version']`` matches :data:`FEATURES_SCHEMA_VERSION`.

Validation
----------

:func:`validate_features` / :func:`assert_features_valid` enforce that every
feature declared in the registry exists with the declared dtype and leading
dimension. Unknown ``/features/<name>`` datasets without a registry entry are
flagged. The validator is invoked automatically by the H5 converter when a
feature registry is declared.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from menflow.data.h5_schema import H5SchemaError, SchemaViolation

logger = logging.getLogger(__name__)


FEATURES_SCHEMA_VERSION = "1.0"
FEATURES_GROUP = "features"


# ---------------------------------------------------------------------------
# Specification primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One feature attached to a unified MenFlow HDF5.

    Parameters
    ----------
    name
        Dataset basename written under ``/features/<name>``.
    dtype
        NumPy dtype string (e.g. ``"float32"``, ``"int32"``, ``"bool"``).
    shape
        Per-row shape excluding the leading dimension. Use ``()`` for scalar
        per-scan features (e.g. ``log_volume_cm3``) and ``(K,)`` for fixed-
        width vector features.
    units
        Physical units (``"cm^3"``, ``"voxels"``, ``"rad"``, ``"mm"``) or the
        empty string for dimensionless / categorical features.
    description
        One-sentence semantic meaning. Persisted in the dataset attrs so the
        cohort is self-describing.
    source
        Provenance tag — typically ``"derived:segmentation"``,
        ``"derived:image"``, ``"derived:metadata"``, or
        ``"external:<source>"``.
    leading_dim
        Which schema-level dimension drives the dataset's leading axis. The
        common case is ``"n_scans"``; ``"n_patients"`` is allowed for
        per-patient summaries.
    required
        If False, validators tolerate the absence of this feature.
    """

    name: str
    dtype: str
    shape: tuple[int, ...]
    units: str
    description: str
    source: str
    leading_dim: str = "n_scans"
    required: bool = True

    def numpy_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "units": self.units,
            "description": self.description,
            "source": self.source,
            "leading_dim": self.leading_dim,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FeatureSpec:
        return cls(
            name=str(data["name"]),
            dtype=str(data["dtype"]),
            shape=tuple(int(s) for s in data.get("shape", ())),
            units=str(data.get("units", "")),
            description=str(data.get("description", "")),
            source=str(data.get("source", "")),
            leading_dim=str(data.get("leading_dim", "n_scans")),
            required=bool(data.get("required", True)),
        )


@dataclass(frozen=True, slots=True)
class FeatureRegistry:
    """Cohort-specific declaration of every feature in ``/features/``."""

    dataset_name: str
    features: tuple[FeatureSpec, ...]
    schema_version: str = FEATURES_SCHEMA_VERSION

    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features)

    def spec(self, name: str) -> FeatureSpec:
        for f in self.features:
            if f.name == name:
                return f
        raise KeyError(f"Feature {name!r} not in registry for {self.dataset_name!r}")

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "dataset_name": self.dataset_name,
                "features": [f.to_dict() for f in self.features],
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> FeatureRegistry:
        data = json.loads(text)
        return cls(
            dataset_name=str(data["dataset_name"]),
            schema_version=str(data.get("schema_version", FEATURES_SCHEMA_VERSION)),
            features=tuple(FeatureSpec.from_dict(f) for f in data["features"]),
        )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_features(
    h5_file: h5py.File,
    registry: FeatureRegistry,
    data: Mapping[str, np.ndarray],
    *,
    overwrite: bool = False,
) -> None:
    """Write every declared feature into ``/features/`` of *h5_file*.

    Parameters
    ----------
    h5_file
        Open H5 file in write/append mode (``"r+"`` or ``"a"``).
    registry
        Cohort feature registry. Each entry's dtype, shape, and leading dim
        must match the corresponding array in *data*.
    data
        Mapping ``feature_name -> ndarray``. Must contain at least every
        ``required=True`` feature; extra keys are ignored with a warning.
    overwrite
        If True and ``/features/`` already exists, recreate it from scratch.

    Raises
    ------
    KeyError
        A required feature has no array in *data*.
    ValueError
        An array's dtype or shape does not match its registry entry, or the
        leading dim disagrees with the file's ``n_scans`` / ``n_patients``.
    """
    n_scans = int(h5_file.attrs["n_scans"])
    n_patients = int(h5_file.attrs["n_patients"])

    if FEATURES_GROUP in h5_file:
        if not overwrite:
            raise ValueError(f"/{FEATURES_GROUP} already exists; pass overwrite=True to replace")
        del h5_file[FEATURES_GROUP]

    grp = h5_file.create_group(FEATURES_GROUP)
    grp.attrs["schema_version"] = registry.schema_version
    grp.attrs["registry_json"] = registry.to_json()

    declared = set(registry.names())
    extras = set(data.keys()) - declared
    if extras:
        logger.warning(
            "Ignoring %d feature arrays not in the registry: %s",
            len(extras),
            sorted(extras),
        )

    for spec in registry.features:
        if spec.name not in data:
            if spec.required:
                raise KeyError(f"Required feature {spec.name!r} missing from data dict")
            continue
        arr = np.asarray(data[spec.name])
        expected_dtype = spec.numpy_dtype()
        if arr.dtype != expected_dtype:
            try:
                arr = arr.astype(expected_dtype)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Feature {spec.name!r}: cannot cast {arr.dtype} -> {expected_dtype}: {exc}"
                ) from exc
        expected_lead = _resolve_leading(spec.leading_dim, n_scans, n_patients)
        if expected_lead is None:
            raise ValueError(f"Feature {spec.name!r}: unknown leading_dim {spec.leading_dim!r}")
        expected_shape = (expected_lead, *spec.shape)
        if arr.shape != expected_shape:
            raise ValueError(
                f"Feature {spec.name!r}: shape {arr.shape} != expected {expected_shape}"
            )

        # vlen-string features get a special h5py dtype.
        if expected_dtype.kind == "O":
            ds = grp.create_dataset(
                spec.name,
                shape=arr.shape,
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            ds[...] = arr
        else:
            ds = grp.create_dataset(spec.name, data=arr)
        ds.attrs["units"] = spec.units
        ds.attrs["description"] = spec.description
        ds.attrs["source"] = spec.source
        ds.attrs["dtype"] = spec.dtype
        ds.attrs["leading_dim"] = spec.leading_dim


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def validate_features(h5_path: str | Path) -> list[SchemaViolation]:
    """Return any violations of the in-file feature registry.

    The validator only fires when ``/features/`` is present; absence is not
    itself a violation (the group is optional).
    """
    out: list[SchemaViolation] = []
    with h5py.File(h5_path, "r") as f:
        if FEATURES_GROUP not in f:
            return out
        grp = f[FEATURES_GROUP]
        if not isinstance(grp, h5py.Group):
            out.append(
                SchemaViolation(
                    FEATURES_GROUP, "missing", "expected /features as Group, got Dataset"
                )
            )
            return out
        if "registry_json" not in grp.attrs:
            out.append(
                SchemaViolation(
                    f"{FEATURES_GROUP}/registry_json",
                    "missing",
                    "feature group lacks self-describing registry_json attr",
                )
            )
            return out
        try:
            registry = FeatureRegistry.from_json(_decode(grp.attrs["registry_json"]))
        except (ValueError, KeyError) as exc:
            out.append(
                SchemaViolation(
                    f"{FEATURES_GROUP}/registry_json",
                    "value",
                    f"registry_json not parseable: {exc}",
                )
            )
            return out

        n_scans = int(f.attrs["n_scans"]) if "n_scans" in f.attrs else None
        n_patients = int(f.attrs["n_patients"]) if "n_patients" in f.attrs else None

        declared = set(registry.names())
        present = set(grp.keys())
        for spec in registry.features:
            path = f"{FEATURES_GROUP}/{spec.name}"
            if spec.name not in present:
                if spec.required:
                    out.append(SchemaViolation(path, "missing", "feature dataset absent"))
                continue
            ds = grp[spec.name]
            if not isinstance(ds, h5py.Dataset):
                out.append(SchemaViolation(path, "missing", "expected Dataset, got Group"))
                continue
            if ds.dtype != spec.numpy_dtype() and not (
                spec.numpy_dtype().kind == "O" and h5py.check_string_dtype(ds.dtype) is not None
            ):
                out.append(
                    SchemaViolation(
                        path,
                        "dtype",
                        f"expected {spec.dtype}, got {ds.dtype}",
                    )
                )
            expected_lead = _resolve_leading(spec.leading_dim, n_scans, n_patients)
            if expected_lead is not None:
                expected_shape = (expected_lead, *spec.shape)
                if ds.shape != expected_shape:
                    out.append(
                        SchemaViolation(
                            path,
                            "shape",
                            f"expected {expected_shape}, got {ds.shape}",
                        )
                    )
            for attr_name in ("units", "description", "source"):
                if attr_name not in ds.attrs:
                    out.append(
                        SchemaViolation(
                            path, "missing", f"feature dataset lacks attr {attr_name!r}"
                        )
                    )

        # Datasets present in the file but not in the registry are a
        # documentation gap, not a hard failure.
        for extra in sorted(present - declared):
            out.append(
                SchemaViolation(
                    f"{FEATURES_GROUP}/{extra}",
                    "value",
                    "feature present but not declared in registry_json",
                )
            )
    return out


def assert_features_valid(h5_path: str | Path) -> None:
    """Raise :class:`H5SchemaError` if ``/features/`` is malformed."""
    violations = validate_features(h5_path)
    if violations:
        joined = "\n  - ".join(str(v) for v in violations)
        raise H5SchemaError(
            f"H5 /features/ group does not satisfy registry contract:\n  - {joined}"
        )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def laterality_from_mask(mask: np.ndarray) -> str:
    """Classify a 3-D binary tumor mask as ``"L"``, ``"R"``, or ``"B"``.

    The convention assumes the cohort orientation places the patient's left
    side at higher indices along axis 0 (right-anatomical first; this matches
    the BraTS-MEN and MenGrowth converters which store data RAS-aligned). The
    classification is based on the laterality of the mask centroid in voxel
    units relative to the midline ``H // 2``.

    Parameters
    ----------
    mask
        3-D array of shape ``(H, W, D)``. Treated as boolean (``mask > 0``).

    Returns
    -------
    str
        One of ``"L"``, ``"R"``, ``"B"``, or ``""`` (empty mask).
    """
    mask = np.asarray(mask)
    if mask.ndim != 3:
        raise ValueError(f"expected 3-D mask, got shape {mask.shape}")
    bool_mask = mask > 0
    if not bool_mask.any():
        return ""
    n_left = int(bool_mask[: bool_mask.shape[0] // 2].sum())
    n_right = int(bool_mask[bool_mask.shape[0] // 2 :].sum())
    if n_left == 0:
        return "R"
    if n_right == 0:
        return "L"
    # Both sides contain mask voxels — bilateral if neither side dominates.
    total = n_left + n_right
    left_frac = n_left / total
    if left_frac >= 0.85:
        return "L"
    if left_frac <= 0.15:
        return "R"
    return "B"


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _resolve_leading(name: str, n_scans: int | None, n_patients: int | None) -> int | None:
    if name == "n_scans":
        return n_scans
    if name == "n_patients":
        return n_patients
    return None


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


__all__ = [
    "FEATURES_GROUP",
    "FEATURES_SCHEMA_VERSION",
    "FeatureRegistry",
    "FeatureSpec",
    "assert_features_valid",
    "laterality_from_mask",
    "validate_features",
    "write_features",
]
