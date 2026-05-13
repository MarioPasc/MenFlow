"""Unit tests for null_baseline.py — determinism + volume-order constraint."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from experiments.E2.E2_5_ae_longitudinal.analysis.null_baseline import (
    compute_null_baseline,
)


def _make_synthetic_dataset(tmp_path: Path, n_patients: int, scans_per_patient: int) -> tuple[Path, Path]:
    """Build a tiny pair (unified, latents) of H5s with valid metadata."""
    rng = np.random.default_rng(0)
    n_scans = n_patients * scans_per_patient
    M, C, H, W, D = 1, 2, 3, 3, 3
    offsets = np.arange(n_patients + 1) * scans_per_patient
    patient_list = [f"P{i:02d}" for i in range(n_patients)]
    scan_ids = [
        f"{patient_list[i]}-{t:03d}" for i in range(n_patients) for t in range(scans_per_patient)
    ]
    log_vol = rng.uniform(-1.0, 3.0, size=n_scans).astype(np.float32)

    unified = tmp_path / "unified.h5"
    with h5py.File(unified, "w") as f:
        f.attrs["dataset_type"] = "longitudinal"
        f.attrs["n_scans"] = np.int64(n_scans)
        f.attrs["n_patients"] = np.int64(n_patients)
        f.attrs["spatial_shape"] = np.array([8, 8, 8], dtype=np.int64)
        vlen = h5py.special_dtype(vlen=str)
        f.create_dataset("scan_ids", data=np.asarray(scan_ids, dtype=object), dtype=vlen)
        f.create_dataset("has_segmentation", data=np.ones(n_scans, dtype=bool))
        feat = f.create_group("features")
        feat.create_dataset("log_volume_cm3", data=log_vol)
        long_grp = f.create_group("longitudinal")
        long_grp.create_dataset("patient_offsets", data=offsets.astype(np.int32))
        long_grp.create_dataset(
            "patient_list", data=np.asarray(patient_list, dtype=object), dtype=vlen
        )

    latents = tmp_path / "latents.h5"
    with h5py.File(latents, "w") as f:
        f.attrs["n_scans"] = np.int64(n_scans)
        f.attrs["n_modalities"] = np.int64(M)
        f.attrs["modalities"] = np.asarray(["t1c"], dtype=object)
        f.attrs["latent_channels"] = np.int64(C)
        f.attrs["latent_spatial_shape"] = np.array([H, W, D], dtype=np.int64)
        f.attrs["padded_spatial_shape"] = np.array([H * 4, W * 4, D * 4], dtype=np.int64)
        z = rng.standard_normal(size=(n_scans, M, C, H, W, D)).astype(np.float32)
        f.create_dataset("latents", data=z)
        vlen = h5py.special_dtype(vlen=str)
        f.create_dataset("scan_ids", data=np.asarray(scan_ids, dtype=object), dtype=vlen)
    return unified, latents


def test_null_baseline_deterministic(tmp_path):
    unified, latents = _make_synthetic_dataset(tmp_path, n_patients=6, scans_per_patient=3)
    r1 = compute_null_baseline(
        latents_h5_path=str(latents),
        unified_h5_path=str(unified),
        n_triples=20,
        seed=123,
    )
    r2 = compute_null_baseline(
        latents_h5_path=str(latents),
        unified_h5_path=str(unified),
        n_triples=20,
        seed=123,
    )
    assert r1.n_drawn > 0
    np.testing.assert_allclose(r1.rho_lin, r2.rho_lin)


def test_null_baseline_returns_finite_rhos(tmp_path):
    unified, latents = _make_synthetic_dataset(tmp_path, n_patients=8, scans_per_patient=2)
    r = compute_null_baseline(
        latents_h5_path=str(latents),
        unified_h5_path=str(unified),
        n_triples=30,
        seed=7,
    )
    assert r.n_drawn > 0
    assert np.all(np.isfinite(r.rho_lin))
    assert np.all(r.rho_lin >= 0)
