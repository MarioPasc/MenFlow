"""Tests for JointLatentVolumeDataset against synthetic H5 fixtures."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from routines.finetune_fm_volume.engine.data import (
    DatasetConfig,
    JointLatentVolumeDataset,
    collate_samples,
)


def _write_synthetic_latent_h5(path: Path, n_scans: int = 6, n_patients: int = 4) -> None:
    rng = np.random.default_rng(0)
    M, C, H, W, D = 4, 4, 8, 8, 8
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "0.2-provisional"
        f.attrs["dataset_name"] = "synthetic"
        f.attrs["dataset_type"] = "cross-sectional"
        f.attrs["n_scans"] = n_scans
        f.attrs["n_patients"] = n_patients
        f.attrs["modalities"] = np.array(["t1c", "t1n", "t2f", "t2w"], dtype=h5py.string_dtype())
        f.attrs["n_modalities"] = M
        f.attrs["latent_channels"] = C
        f.attrs["latent_spatial_shape"] = np.array([H, W, D], dtype=np.int32)
        f.attrs["spacing_mm"] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        f.attrs["orientation"] = "LAS"
        f.create_dataset(
            "latents", data=rng.standard_normal((n_scans, M, C, H, W, D)).astype(np.float16)
        )
        f.create_dataset(
            "scan_ids",
            data=np.array([f"S{i:03d}-000" for i in range(n_scans)], dtype=h5py.string_dtype()),
        )
        f.create_dataset(
            "patient_ids",
            data=np.array(
                [f"P{i % n_patients:03d}" for i in range(n_scans)], dtype=h5py.string_dtype()
            ),
        )
        f.create_dataset("timepoint_idx", data=np.zeros(n_scans, dtype=np.int32))
        # Patients: P0=2 scans, P1=1, P2=2, P3=1  → offsets [0,2,3,5,6]
        offsets = np.array([0, 2, 3, 5, 6], dtype=np.int32)
        f.create_dataset("longitudinal/patient_offsets", data=offsets)
        f.create_dataset(
            "longitudinal/patient_list",
            data=np.array([f"P{i:03d}" for i in range(n_patients)], dtype=h5py.string_dtype()),
        )
        # Splits at the patient level: train=[0,2], val=[1,3]
        f.create_dataset("splits/train", data=np.array([0, 2], dtype=np.int32))
        f.create_dataset("splits/val", data=np.array([1, 3], dtype=np.int32))


def _write_synthetic_features_h5(path: Path, scan_ids: list[str], log_v: np.ndarray) -> None:
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "0.1"
        f.attrs["dataset_name"] = "synthetic"
        f.create_dataset("scan_ids", data=np.array(scan_ids, dtype=h5py.string_dtype()))
        f.create_dataset("log_volume", data=log_v.astype(np.float32))


@pytest.fixture
def synthetic_files(tmp_path: Path) -> dict:
    latent_h5 = tmp_path / "lat.h5"
    feat_dir = tmp_path / "feat"
    feat_dir.mkdir()
    _write_synthetic_latent_h5(latent_h5)
    scan_ids = [f"S{i:03d}-000" for i in range(6)]
    log_v = np.array([-7.0, 1.0, 2.0, -7.0, 0.5, 1.5])  # two sentinels
    for m in ("t1c", "t1n", "t2f", "t2w"):
        _write_synthetic_features_h5(feat_dir / f"brats_men_features_{m}.h5", scan_ids, log_v)
    return {"latent_h5": latent_h5, "feat_dir": feat_dir}


def test_dataset_train_split_filters_sentinel(synthetic_files: dict) -> None:
    cfg = DatasetConfig(
        latent_h5=synthetic_files["latent_h5"],
        features_h5_dir=synthetic_files["feat_dir"],
        modalities=("t1c",),
        split="train",
        log_v_floor=-6.0,
    )
    ds = JointLatentVolumeDataset(cfg)
    # train patients = [P0, P2]; P0 owns rows [0,1], P2 owns rows [3,4]
    # log_v sentinels at rows 0 and 3; usable: rows 1, 4.
    assert len(ds) == 2
    sample = ds[0]
    # Fixture latent shape (4,8,8,8) is padded to /16 by the dataset's
    # pad_multiple=16 default; padding is required so the FM U-Net's three
    # downsample steps see dims divisible by 16.
    assert sample["z"].shape == (4, 16, 16, 16)
    assert sample["modality_idx"].item() == 17  # t1c
    assert sample["log_v"].item() in (1.0, 0.5)


def test_dataset_multi_modality(synthetic_files: dict) -> None:
    cfg = DatasetConfig(
        latent_h5=synthetic_files["latent_h5"],
        features_h5_dir=synthetic_files["feat_dir"],
        modalities=("t1c", "t2w"),
        split="train",
        log_v_floor=-6.0,
    )
    ds = JointLatentVolumeDataset(cfg)
    assert len(ds) == 4  # 2 usable scans × 2 modalities
    s0 = ds[0]  # first modality block (t1c)
    s_last = ds[len(ds) - 1]
    assert s0["modality"] == "t1c"
    assert s_last["modality"] == "t2w"
    assert s_last["modality_idx"].item() == 10  # t2w


def test_dataset_val_split(synthetic_files: dict) -> None:
    cfg = DatasetConfig(
        latent_h5=synthetic_files["latent_h5"],
        features_h5_dir=synthetic_files["feat_dir"],
        modalities=("t1c",),
        split="val",
        log_v_floor=-6.0,
    )
    ds = JointLatentVolumeDataset(cfg)
    # val patients = [P1, P3]; P1 owns row 2, P3 owns row 5. Both have log_v > -6.
    assert len(ds) == 2


def test_collate(synthetic_files: dict) -> None:
    cfg = DatasetConfig(
        latent_h5=synthetic_files["latent_h5"],
        features_h5_dir=synthetic_files["feat_dir"],
        modalities=("t1c",),
        split="train",
    )
    ds = JointLatentVolumeDataset(cfg)
    batch = collate_samples([ds[0], ds[1]])
    assert batch["z"].shape == (2, 4, 16, 16, 16)
    assert batch["spacing"].shape == (2, 3)
    assert batch["log_v"].shape == (2,)
    assert isinstance(batch["scan_id"], list) and len(batch["scan_id"]) == 2
