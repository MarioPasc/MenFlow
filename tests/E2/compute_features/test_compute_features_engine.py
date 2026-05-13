"""End-to-end test of the compute-features routine on synthetic latent + source H5."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from experiments.E2.compute_features.engine.compute_features_engine import (
    ComputeFeaturesEngine,
    ComputeFeaturesRoutineConfig,
)


def _write_synthetic_pair(
    src_path: Path,
    lat_path: Path,
    *,
    n: int = 3,
    spatial: tuple[int, int, int] = (16, 16, 16),
    stride: int = 4,
    c: int = 4,
    m: int = 4,
) -> None:
    h, w, d = spatial
    h_l, w_l, d_l = h // stride, w // stride, d // stride
    rng = np.random.default_rng(0)

    # --- source unified MenFlow H5 ---
    with h5py.File(src_path, "w") as f:
        f.attrs["schema_version"] = "1.0"
        f.attrs["dataset_name"] = "Synthetic"
        f.attrs["dataset_type"] = "cross-sectional"
        f.attrs["n_scans"] = np.int64(n)
        f.attrs["n_patients"] = np.int64(n)
        f.attrs["modalities"] = np.asarray(["t1c", "t1n", "t2f", "t2w"], dtype=object)
        f.attrs["n_modalities"] = np.int64(m)
        f.attrs["spatial_shape"] = np.asarray(spatial, dtype=np.int64)
        f.attrs["spacing_mm"] = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
        f.attrs["orientation"] = "RAS"
        f.attrs["label_map"] = json.dumps({"1": "tumor"})
        f.attrs["intensity_normalized"] = False
        f.attrs["created_at"] = "1970-01-01T00:00:00+00:00"
        f.attrs["has_segmentation_any"] = True

        f.create_dataset("images", data=rng.standard_normal((n, m, h, w, d)).astype(np.float32))
        seg = np.zeros((n, h, w, d), dtype=np.int8)
        # Place a small lesion in each scan; size grows with index.
        for i in range(n):
            r = i + 1
            seg[i, 4 : 4 + r, 4 : 4 + r, 4 : 4 + r] = 1
        f.create_dataset("segmentations", data=seg)
        f.create_dataset("has_segmentation", data=np.ones(n, dtype=bool))
        vlen = h5py.special_dtype(vlen=str)
        ds = f.create_dataset("scan_ids", shape=(n,), dtype=vlen)
        ds[...] = np.array([f"S{i:03d}" for i in range(n)], dtype=object)
        ds = f.create_dataset("patient_ids", shape=(n,), dtype=vlen)
        ds[...] = np.array([f"P{i:03d}" for i in range(n)], dtype=object)
        f.create_dataset("timepoint_idx", data=np.zeros(n, dtype=np.int32))
        long_grp = f.create_group("longitudinal")
        long_grp.create_dataset("patient_offsets", data=np.arange(n + 1, dtype=np.int32))
        ds = long_grp.create_dataset("patient_list", shape=(n,), dtype=vlen)
        ds[...] = np.array([f"P{i:03d}" for i in range(n)], dtype=object)
        f.create_group("metadata")

    # --- latent H5 (provisional schema) ---
    with h5py.File(lat_path, "w") as f:
        f.attrs["schema_version"] = "0.2-provisional"
        f.attrs["source_h5"] = str(src_path)
        f.attrs["encoded_at"] = "1970-01-01T00:00:00+00:00"
        f.attrs["encoder_name"] = "synthetic"
        f.attrs["encoder_checkpoint"] = "synthetic"
        f.attrs["encoder_config"] = "{}"
        f.attrs["inference_config"] = "{}"
        f.attrs["deterministic"] = True
        f.attrs["dataset_name"] = "Synthetic"
        f.attrs["dataset_type"] = "cross-sectional"
        f.attrs["n_scans"] = np.int64(n)
        f.attrs["n_patients"] = np.int64(n)
        f.attrs["modalities"] = np.asarray(["t1c", "t1n", "t2f", "t2w"], dtype=object)
        f.attrs["n_modalities"] = np.int64(m)
        f.attrs["orientation"] = "RAS"
        f.attrs["spacing_mm"] = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
        f.attrs["source_spatial_shape"] = np.asarray(spatial, dtype=np.int64)
        f.attrs["working_spatial_shape"] = np.asarray(spatial, dtype=np.int64)
        f.attrs["padded_spatial_shape"] = np.asarray(spatial, dtype=np.int64)
        f.attrs["spatial_op"] = "none"
        f.attrs["crop_offset"] = np.asarray([0, 0, 0], dtype=np.int64)
        f.attrs["latent_spatial_shape"] = np.asarray([h_l, w_l, d_l], dtype=np.int64)
        f.attrs["latent_channels"] = np.int64(c)

        latents = rng.standard_normal((n, m, c, h_l, w_l, d_l)).astype(np.float16)
        f.create_dataset("latents", data=latents)
        vlen = h5py.special_dtype(vlen=str)
        ds = f.create_dataset("scan_ids", shape=(n,), dtype=vlen)
        ds[...] = np.array([f"S{i:03d}" for i in range(n)], dtype=object)
        ds = f.create_dataset("patient_ids", shape=(n,), dtype=vlen)
        ds[...] = np.array([f"P{i:03d}" for i in range(n)], dtype=object)
        f.create_dataset("timepoint_idx", data=np.zeros(n, dtype=np.int32))


def test_compute_features_writes_expected_artifact(tmp_path: Path) -> None:
    src = tmp_path / "src.h5"
    lat = tmp_path / "latents.h5"
    out = tmp_path / "features.h5"
    _write_synthetic_pair(src, lat)

    cfg = ComputeFeaturesRoutineConfig(
        latent_h5=lat,
        source_h5=src,
        output_h5=out,
        modality="t1c",
        mask_label_set=(1,),
        random_region_seed=0,
        compute_z_full=False,
        log_level="WARNING",
    )
    out_path = ComputeFeaturesEngine(cfg).run()
    assert out_path == out
    with h5py.File(out_path, "r") as f:
        assert f["z_tumor"].shape == (3, 4)
        assert f["z_global"].shape == (3, 4)
        assert f["z_random"].shape == (3, 4)
        assert f["log_volume"].shape == (3,)
        assert bool(f["mask_lat_present"][:].all())
        assert f.attrs["modality"] == "t1c"
        assert f.attrs["compute_z_full"] in (False, np.bool_(False))
        # Larger lesions → larger log V
        log_v = f["log_volume"][:]
        assert log_v[0] < log_v[1] < log_v[2]


def test_compute_features_with_z_full_toggle(tmp_path: Path) -> None:
    src = tmp_path / "src.h5"
    lat = tmp_path / "latents.h5"
    out = tmp_path / "features.h5"
    _write_synthetic_pair(src, lat, n=2)

    cfg = ComputeFeaturesRoutineConfig(
        latent_h5=lat,
        source_h5=src,
        output_h5=out,
        modality="t1c",
        mask_label_set=(1,),
        compute_z_full=True,
        log_level="WARNING",
    )
    ComputeFeaturesEngine(cfg).run()
    with h5py.File(out, "r") as f:
        assert "z_full" in f
        assert "mask_lat" in f
        assert f["z_full"].shape == (2, 4, 4, 4, 4)
        assert f["mask_lat"].shape == (2, 1, 4, 4, 4)
