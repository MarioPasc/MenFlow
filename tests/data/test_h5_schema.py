"""Schema-level unit tests for the unified MenFlow H5 layout.

Builds synthetic fixtures with :mod:`h5py` and exercises every violation kind
(missing attribute, wrong dtype, wrong shape, broken cross-field invariant).
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from menflow.data.h5_schema import (
    H5Schema,
    H5SchemaError,
    SCHEMA_VERSION,
    validate_h5,
)


# ---------------------------------------------------------------------------
# Fixture: a minimal valid file
# ---------------------------------------------------------------------------


def _write_minimal_valid(path: Path, n_scans: int = 2, n_patients: int = 2) -> None:
    H, W, D, M = 8, 8, 6, 2
    vlen = h5py.special_dtype(vlen=str)
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["dataset_name"] = "ToyCohort"
        f.attrs["dataset_type"] = "cross-sectional"
        f.attrs["n_scans"] = np.int64(n_scans)
        f.attrs["n_patients"] = np.int64(n_patients)
        f.attrs["modalities"] = np.asarray(["m1", "m2"], dtype=object)
        f.attrs["n_modalities"] = np.int64(M)
        f.attrs["spatial_shape"] = np.asarray([H, W, D], dtype=np.int64)
        f.attrs["spacing_mm"] = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
        f.attrs["orientation"] = "RAS"
        f.attrs["label_map"] = json.dumps({0: "background", 1: "tumor"})
        f.attrs["intensity_normalized"] = bool(False)
        f.attrs["created_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        f.attrs["has_segmentation_any"] = bool(True)

        f.create_dataset("images", data=np.zeros((n_scans, M, H, W, D), dtype=np.float32))
        f.create_dataset("segmentations", data=np.zeros((n_scans, H, W, D), dtype=np.int8))
        f.create_dataset("has_segmentation", data=np.ones(n_scans, dtype=bool))
        f.create_dataset(
            "scan_ids",
            data=np.asarray([f"s{i}" for i in range(n_scans)], dtype=object),
            dtype=vlen,
        )
        f.create_dataset(
            "patient_ids",
            data=np.asarray([f"p{i}" for i in range(n_scans)], dtype=object),
            dtype=vlen,
        )
        f.create_dataset("timepoint_idx", data=np.zeros(n_scans, dtype=np.int32))

        long_grp = f.create_group("longitudinal")
        long_grp.create_dataset(
            "patient_offsets",
            data=np.arange(n_patients + 1, dtype=np.int32),
        )
        long_grp.create_dataset(
            "patient_list",
            data=np.asarray([f"p{i}" for i in range(n_patients)], dtype=object),
            dtype=vlen,
        )

        f.create_group("metadata")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSchemaValidates:
    def test_minimal_valid_file_passes(self, tmp_path: Path) -> None:
        path = tmp_path / "valid.h5"
        _write_minimal_valid(path)
        violations = validate_h5(path)
        assert violations == [], "\n".join(str(v) for v in violations)

    def test_assert_valid_silent_on_clean_file(self, tmp_path: Path) -> None:
        path = tmp_path / "valid.h5"
        _write_minimal_valid(path)
        with h5py.File(path, "r") as f:
            H5Schema().assert_valid(f)  # no exception expected


class TestMissingAttrs:
    @pytest.mark.parametrize(
        "attr",
        [
            "schema_version",
            "dataset_name",
            "dataset_type",
            "n_scans",
            "n_patients",
            "modalities",
            "spatial_shape",
            "spacing_mm",
            "orientation",
            "label_map",
        ],
    )
    def test_missing_required_attr_raises(self, tmp_path: Path, attr: str) -> None:
        path = tmp_path / "broken.h5"
        _write_minimal_valid(path)
        with h5py.File(path, "a") as f:
            del f.attrs[attr]
        with pytest.raises(H5SchemaError) as exc:
            with h5py.File(path, "r") as f:
                H5Schema().assert_valid(f)
        assert attr in str(exc.value)


class TestMissingDatasets:
    @pytest.mark.parametrize(
        "ds_path",
        [
            "images",
            "segmentations",
            "has_segmentation",
            "scan_ids",
            "patient_ids",
            "timepoint_idx",
            "longitudinal/patient_offsets",
            "longitudinal/patient_list",
        ],
    )
    def test_missing_dataset_raises(self, tmp_path: Path, ds_path: str) -> None:
        path = tmp_path / "broken.h5"
        _write_minimal_valid(path)
        with h5py.File(path, "a") as f:
            del f[ds_path]
        violations = validate_h5(path)
        assert any(v.path == ds_path and v.kind == "missing" for v in violations), violations


class TestCrossFieldInvariants:
    def test_images_channel_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.h5"
        _write_minimal_valid(path)
        with h5py.File(path, "a") as f:
            del f["images"]
            f.create_dataset(
                "images",
                data=np.zeros((2, 99, 8, 8, 6), dtype=np.float32),  # wrong M
            )
        violations = validate_h5(path)
        assert any("channel dim" in v.detail for v in violations), violations

    def test_offsets_must_terminate_at_n_scans(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.h5"
        _write_minimal_valid(path)
        with h5py.File(path, "a") as f:
            del f["longitudinal/patient_offsets"]
            f["longitudinal"].create_dataset(
                "patient_offsets",
                data=np.array([0, 1, 999], dtype=np.int32),
            )
        violations = validate_h5(path)
        assert any("last offset" in v.detail for v in violations), violations

    def test_label_map_must_be_json(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.h5"
        _write_minimal_valid(path)
        with h5py.File(path, "a") as f:
            f.attrs["label_map"] = "not a json"
        violations = validate_h5(path)
        assert any(v.path == "label_map" and v.kind == "value" for v in violations)

    def test_dataset_type_whitelist(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.h5"
        _write_minimal_valid(path)
        with h5py.File(path, "a") as f:
            f.attrs["dataset_type"] = "weekly"
        violations = validate_h5(path)
        assert any(v.path == "dataset_type" and v.kind == "value" for v in violations)


class TestDtype:
    def test_segmentation_must_be_int_kind(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.h5"
        _write_minimal_valid(path)
        with h5py.File(path, "a") as f:
            del f["segmentations"]
            f.create_dataset(
                "segmentations",
                data=np.zeros((2, 8, 8, 6), dtype=np.float32),  # wrong kind
            )
        violations = validate_h5(path)
        assert any(v.path == "segmentations" and v.kind == "dtype" for v in violations)
