"""Unit tests for the per-cohort feature registry."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from menflow.data.features import (
    FEATURES_GROUP,
    FeatureRegistry,
    FeatureSpec,
    assert_features_valid,
    laterality_from_mask,
    validate_features,
    write_features,
)
from menflow.data.h5_schema import H5SchemaError


def _make_unified_h5(tmp_path: Path, n_scans: int = 3, n_patients: int = 2) -> Path:
    """Build a minimal H5 with the root attrs the feature module needs."""
    path = tmp_path / "cohort.h5"
    with h5py.File(path, "w") as f:
        f.attrs["n_scans"] = n_scans
        f.attrs["n_patients"] = n_patients
    return path


def _toy_registry() -> FeatureRegistry:
    return FeatureRegistry(
        dataset_name="ToyCohort",
        features=(
            FeatureSpec(
                name="log_volume_cm3",
                dtype="float32",
                shape=(),
                units="cm^3",
                description="log of tumor volume in cm^3",
                source="derived:segmentation",
            ),
            FeatureSpec(
                name="laterality",
                dtype="O",
                shape=(),
                units="",
                description="L/R/B side classification",
                source="derived:segmentation",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# FeatureSpec / FeatureRegistry round-trip
# ---------------------------------------------------------------------------


def test_feature_spec_to_dict_round_trip() -> None:
    spec = FeatureSpec(
        name="x",
        dtype="float32",
        shape=(3,),
        units="mm",
        description="d",
        source="derived:image",
    )
    assert FeatureSpec.from_dict(spec.to_dict()) == spec


def test_registry_to_json_round_trip() -> None:
    reg = _toy_registry()
    rebuilt = FeatureRegistry.from_json(reg.to_json())
    assert rebuilt == reg


def test_registry_spec_lookup_and_missing() -> None:
    reg = _toy_registry()
    assert reg.spec("log_volume_cm3").units == "cm^3"
    with pytest.raises(KeyError):
        reg.spec("does_not_exist")


# ---------------------------------------------------------------------------
# write_features
# ---------------------------------------------------------------------------


def test_write_features_creates_documented_datasets(tmp_path: Path) -> None:
    h5 = _make_unified_h5(tmp_path, n_scans=3)
    reg = _toy_registry()
    data = {
        "log_volume_cm3": np.array([0.5, 1.0, 2.0], dtype=np.float32),
        "laterality": np.array(["L", "R", "B"], dtype=object),
    }
    with h5py.File(h5, "r+") as f:
        write_features(f, reg, data)
    with h5py.File(h5, "r") as f:
        grp = f[FEATURES_GROUP]
        assert grp.attrs["schema_version"]
        assert "log_volume_cm3" in grp
        ds = grp["log_volume_cm3"]
        assert ds.shape == (3,)
        assert ds.dtype == np.float32
        assert ds.attrs["units"] == "cm^3"
        assert ds.attrs["description"]
        assert ds.attrs["source"] == "derived:segmentation"
        assert ds.attrs["leading_dim"] == "n_scans"
        # vlen-string laterality round-trips
        lat = grp["laterality"][:]
        assert [s.decode() if isinstance(s, bytes) else s for s in lat] == ["L", "R", "B"]


def test_write_features_rejects_wrong_shape(tmp_path: Path) -> None:
    h5 = _make_unified_h5(tmp_path, n_scans=3)
    reg = _toy_registry()
    data = {
        "log_volume_cm3": np.array([0.5, 1.0], dtype=np.float32),  # wrong leading dim
        "laterality": np.array(["L", "R", "B"], dtype=object),
    }
    with h5py.File(h5, "r+") as f, pytest.raises(ValueError, match="shape"):
        write_features(f, reg, data)


def test_write_features_required_missing(tmp_path: Path) -> None:
    h5 = _make_unified_h5(tmp_path, n_scans=3)
    reg = _toy_registry()
    data = {"log_volume_cm3": np.array([0.0, 0.0, 0.0], dtype=np.float32)}
    with h5py.File(h5, "r+") as f, pytest.raises(KeyError, match="laterality"):
        write_features(f, reg, data)


def test_write_features_overwrite_replaces_group(tmp_path: Path) -> None:
    h5 = _make_unified_h5(tmp_path, n_scans=3)
    reg = _toy_registry()
    data = {
        "log_volume_cm3": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "laterality": np.array(["L", "R", "B"], dtype=object),
    }
    with h5py.File(h5, "r+") as f:
        write_features(f, reg, data)
        with pytest.raises(ValueError, match="already exists"):
            write_features(f, reg, data)
        write_features(f, reg, data, overwrite=True)


# ---------------------------------------------------------------------------
# validate_features
# ---------------------------------------------------------------------------


def test_validate_features_returns_empty_when_group_absent(tmp_path: Path) -> None:
    h5 = _make_unified_h5(tmp_path, n_scans=3)
    assert validate_features(h5) == []


def test_assert_features_valid_passes_on_good_file(tmp_path: Path) -> None:
    h5 = _make_unified_h5(tmp_path, n_scans=3)
    reg = _toy_registry()
    data = {
        "log_volume_cm3": np.array([0.5, 1.0, 2.0], dtype=np.float32),
        "laterality": np.array(["L", "R", "B"], dtype=object),
    }
    with h5py.File(h5, "r+") as f:
        write_features(f, reg, data)
    assert_features_valid(h5)


def test_validate_features_detects_missing_attr(tmp_path: Path) -> None:
    h5 = _make_unified_h5(tmp_path, n_scans=3)
    reg = _toy_registry()
    data = {
        "log_volume_cm3": np.array([0.5, 1.0, 2.0], dtype=np.float32),
        "laterality": np.array(["L", "R", "B"], dtype=object),
    }
    with h5py.File(h5, "r+") as f:
        write_features(f, reg, data)
        del f[f"{FEATURES_GROUP}/log_volume_cm3"].attrs["description"]
    violations = validate_features(h5)
    assert any("description" in v.detail for v in violations)
    with pytest.raises(H5SchemaError):
        assert_features_valid(h5)


def test_validate_features_detects_undeclared_dataset(tmp_path: Path) -> None:
    h5 = _make_unified_h5(tmp_path, n_scans=3)
    reg = _toy_registry()
    data = {
        "log_volume_cm3": np.array([0.5, 1.0, 2.0], dtype=np.float32),
        "laterality": np.array(["L", "R", "B"], dtype=object),
    }
    with h5py.File(h5, "r+") as f:
        write_features(f, reg, data)
        f[FEATURES_GROUP].create_dataset("rogue", data=np.zeros(3, dtype=np.float32))
    violations = validate_features(h5)
    assert any("rogue" in v.path for v in violations)


# ---------------------------------------------------------------------------
# laterality_from_mask
# ---------------------------------------------------------------------------


def test_laterality_from_mask_empty_returns_empty_string() -> None:
    assert laterality_from_mask(np.zeros((4, 4, 4))) == ""


def test_laterality_from_mask_left_only_classified_left() -> None:
    m = np.zeros((10, 4, 4))
    m[:4, :, :] = 1
    assert laterality_from_mask(m) == "L"


def test_laterality_from_mask_right_only_classified_right() -> None:
    m = np.zeros((10, 4, 4))
    m[6:, :, :] = 1
    assert laterality_from_mask(m) == "R"


def test_laterality_from_mask_balanced_classified_bilateral() -> None:
    m = np.zeros((10, 4, 4))
    m[:5, :, :] = 1
    m[5:, :, :] = 1
    assert laterality_from_mask(m) == "B"


def test_laterality_from_mask_rejects_non_3d() -> None:
    with pytest.raises(ValueError):
        laterality_from_mask(np.zeros((4, 4)))
