"""Tests for :mod:`menflow.data.migrate_e3_splits`.

Builds two synthetic H5 files (source unified-schema-shaped + features) that
share scan_ids, runs the migration, and verifies the new
``splits/kfold/k{N}/...`` layout is written, the legacy splits are removed,
the migration is reproducible, and the per-file provenance JSON is produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from menflow.data.migrate_e3_splits import migrate_kfold_splits


def _write_synthetic_source(path: Path, n_train: int = 30, n_val: int = 5) -> None:
    n = n_train + n_val
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "1.0"
        f.attrs["dataset_name"] = "synthetic"
        f.attrs["dataset_type"] = "cross-sectional"
        f.attrs["n_scans"] = np.int64(n)
        f.attrs["n_patients"] = np.int64(n)
        scan_ids = [f"S{i:03d}" for i in range(n)]
        patient_ids = [f"P{i:03d}" for i in range(n)]
        subsets = ["train"] * n_train + ["val"] * n_val
        f.create_dataset("scan_ids", data=np.asarray(scan_ids, dtype=h5py.string_dtype()))
        f.create_dataset("patient_ids", data=np.asarray(patient_ids, dtype=h5py.string_dtype()))
        f.create_dataset(
            "longitudinal/patient_list", data=np.asarray(patient_ids, dtype=h5py.string_dtype())
        )
        f.create_dataset("longitudinal/patient_offsets", data=np.arange(n + 1, dtype=np.int32))
        f.create_dataset("metadata/subset", data=np.asarray(subsets, dtype=h5py.string_dtype()))
        # Pre-existing legacy splits (intermediate e3_* + BraTS challenge train/val).
        f.create_dataset("splits/train", data=np.arange(n_train, dtype=np.int32))
        f.create_dataset("splits/val", data=np.arange(n_train, n, dtype=np.int32))
        f.create_dataset("splits/e3_train", data=np.arange(n_train // 2, dtype=np.int32))
        f.create_dataset(
            "splits/e3_val",
            data=np.arange(n_train // 2, n_train // 2 + 3, dtype=np.int32),
        )
        f.create_dataset("rng_seed", data=np.int64(0))  # unrelated; must survive


def _write_synthetic_features(path: Path, n_train: int = 30, n_val: int = 5) -> None:
    n = n_train + n_val
    log_v = np.concatenate(
        [
            np.linspace(1.0, 5.0, n_train, dtype=np.float32),
            np.full(n_val, np.log(1e-3), dtype=np.float32),
        ]
    )
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "0.1"
        scan_ids = [f"S{i:03d}" for i in range(n)]
        f.create_dataset("scan_ids", data=np.asarray(scan_ids, dtype=h5py.string_dtype()))
        f.create_dataset("log_volume", data=log_v)


def _write_synthetic_latent(path: Path, n_train: int = 30, n_val: int = 5) -> None:
    n = n_train + n_val
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "0.2-provisional"
        scan_ids = [f"S{i:03d}" for i in range(n)]
        f.create_dataset("scan_ids", data=np.asarray(scan_ids, dtype=h5py.string_dtype()))
        f.create_dataset("splits/train", data=np.arange(n_train, dtype=np.int32))
        f.create_dataset("splits/val", data=np.arange(n_train, n, dtype=np.int32))
        f.create_dataset("dummy_latents", data=np.zeros((n, 1), dtype=np.float16))


@pytest.fixture
def synthetic_files(tmp_path: Path) -> dict:
    src = tmp_path / "src.h5"
    feat = tmp_path / "feat.h5"
    lat = tmp_path / "lat.h5"
    _write_synthetic_source(src)
    _write_synthetic_features(feat)
    _write_synthetic_latent(lat)
    return {"source": src, "features": feat, "latent": lat}


def test_migrate_writes_kfold_into_both_files(synthetic_files: dict) -> None:
    src = synthetic_files["source"]
    lat = synthetic_files["latent"]
    manifest = migrate_kfold_splits(
        source_h5=src,
        latent_h5=lat,
        features_h5=synthetic_files["features"],
        k_values=(1, 3, 5),
        seed=42,
        n_strata=3,
        test_pct=0.1,
    )
    assert manifest["status"] == "written"
    for path in (src, lat):
        with h5py.File(path, "r") as f:
            for k in (1, 3, 5):
                assert f"splits/kfold/k{k}/test" in f
                assert f"splits/kfold/k{k}/fold_0/train" in f
                assert f"splits/kfold/k{k}/fold_0/val" in f
            # Legacy splits removed.
            top = list(f["splits"].keys())
            assert "train" not in top
            assert "val" not in top
            assert "e3_train" not in top
            assert "e3_val" not in top
            assert "kfold" in top


def test_migrate_check_only_passes(synthetic_files: dict) -> None:
    src = synthetic_files["source"]
    lat = synthetic_files["latent"]
    migrate_kfold_splits(
        source_h5=src,
        latent_h5=lat,
        features_h5=synthetic_files["features"],
        k_values=(1, 3),
        seed=42,
        n_strata=3,
    )
    out = migrate_kfold_splits(
        source_h5=src,
        latent_h5=lat,
        features_h5=synthetic_files["features"],
        k_values=(1, 3),
        seed=42,
        n_strata=3,
        check_only=True,
    )
    assert out["status"] == "checked_ok"


def test_migrate_writes_provenance_json(synthetic_files: dict) -> None:
    src = synthetic_files["source"]
    migrate_kfold_splits(
        source_h5=src,
        latent_h5=None,
        features_h5=synthetic_files["features"],
        k_values=(1, 3),
        seed=42,
        n_strata=3,
    )
    prov = src.with_suffix(src.suffix + ".splits_provenance.json")
    assert prov.exists()
    with open(prov) as f:
        manifest = json.load(f)
    assert manifest["seed"] == 42
    assert manifest["k_values"] == [1, 3]
    assert "log_v_distribution" in manifest
    assert "k1" in manifest["log_v_distribution"]
    assert "k3" in manifest["log_v_distribution"]


def test_migrate_keeps_unrelated_datasets(synthetic_files: dict) -> None:
    src = synthetic_files["source"]
    migrate_kfold_splits(
        source_h5=src,
        latent_h5=None,
        features_h5=synthetic_files["features"],
        k_values=(1, 3),
        seed=42,
        n_strata=3,
    )
    with h5py.File(src, "r") as f:
        assert "rng_seed" in f
        assert int(f["rng_seed"][()]) == 0
        assert "longitudinal/patient_list" in f


def test_migrate_test_set_is_shared_across_k(synthetic_files: dict) -> None:
    src = synthetic_files["source"]
    migrate_kfold_splits(
        source_h5=src,
        latent_h5=None,
        features_h5=synthetic_files["features"],
        k_values=(1, 3, 5),
        seed=42,
        n_strata=3,
    )
    with h5py.File(src, "r") as f:
        t1 = set(f["splits/kfold/k1/test"][:].tolist())
        t3 = set(f["splits/kfold/k3/test"][:].tolist())
        t5 = set(f["splits/kfold/k5/test"][:].tolist())
    assert t1 == t3 == t5
