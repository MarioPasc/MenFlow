"""Round-trip test for the v0.2-provisional latent HDF5 writer/reader."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from menflow.maisi_autoencoder.config import InferenceConfig, MaisiV2Config
from menflow.maisi_autoencoder.latents_h5 import (
    LATENT_SCHEMA_VERSION,
    LatentH5Writer,
    read_latent_h5,
)


def _common_kwargs(
    n_scans: int,
    modalities: tuple[str, ...],
    *,
    n_patients: int,
) -> dict:
    """Default kwargs for a synthetic latent file (cross-sectional, no crop)."""
    patient_offsets = list(range(n_patients + 1))
    while patient_offsets[-1] < n_scans:
        patient_offsets[-1] += 1
    return dict(
        dataset_name="ToyCohort",
        dataset_type="cross-sectional",
        modalities=modalities,
        orientation="RAS",
        spacing_mm=(1.0, 1.0, 1.0),
        patient_list=[f"patient-{i}" for i in range(n_patients)],
        patient_offsets=patient_offsets,
        n_scans=n_scans,
        source_spatial_shape=(32, 32, 24),
        working_spatial_shape=(32, 32, 24),
        padded_spatial_shape=(32, 32, 32),
        latent_spatial_shape=(8, 8, 8),
        latent_channels=4,
        spatial_op="none",
        crop_offset=(0, 0, 0),
        source_h5="/tmp/source.h5",
        encoder_checkpoint="/tmp/ae.pt",
        encoder_config=MaisiV2Config(),
        inference_config=InferenceConfig(),
        deterministic=True,
    )


def test_latent_h5_round_trip(tmp_path: Path) -> None:
    n_scans = 3
    modalities = ("t1c", "t2f")
    latent_spatial = (8, 8, 8)
    latent_channels = 4
    output = tmp_path / "latents.h5"

    rng = np.random.default_rng(0)
    latents = rng.normal(size=(n_scans, len(modalities), latent_channels, *latent_spatial)).astype(
        np.float32
    )
    lower = rng.uniform(0, 50, size=(n_scans, len(modalities))).astype(np.float32)
    upper = rng.uniform(500, 1000, size=(n_scans, len(modalities))).astype(np.float32)

    kwargs = _common_kwargs(n_scans, modalities, n_patients=n_scans)
    with LatentH5Writer(output, latent_dtype="float32", **kwargs) as writer:
        for i in range(n_scans):
            writer.write_scan(
                i,
                scan_id=f"scan-{i}",
                patient_id=f"patient-{i}",
                timepoint=0,
                latent=latents[i],
                intensity_lower=lower[i],
                intensity_upper=upper[i],
            )

    with h5py.File(output, "r") as f:
        assert f.attrs["schema_version"] == LATENT_SCHEMA_VERSION
        assert f.attrs["dataset_name"] == "ToyCohort"
        assert f.attrs["dataset_type"] == "cross-sectional"
        assert int(f.attrs["n_scans"]) == n_scans
        assert int(f.attrs["n_modalities"]) == 2
        assert int(f.attrs["latent_channels"]) == latent_channels
        assert tuple(int(x) for x in f.attrs["latent_spatial_shape"]) == latent_spatial
        assert tuple(int(x) for x in f.attrs["working_spatial_shape"]) == (32, 32, 24)
        assert f.attrs["spatial_op"] == "none"
        assert bool(f.attrs["deterministic"]) is True

        cfg_round = json.loads(f.attrs["encoder_config"])
        assert cfg_round["latent_channels"] == 4

        np.testing.assert_allclose(f["latents"][:], latents, atol=1e-6)
        np.testing.assert_allclose(f["intensity_lower"][:], lower)
        np.testing.assert_allclose(f["intensity_upper"][:], upper)

        # CSR layout present
        assert "longitudinal/patient_offsets" in f
        assert "longitudinal/patient_list" in f
        assert int(f["longitudinal/patient_offsets"][0]) == 0
        assert int(f["longitudinal/patient_offsets"][-1]) == n_scans

    out = read_latent_h5(output)
    assert out["latents"].shape == (n_scans, 2, latent_channels, *latent_spatial)
    assert out["scan_ids"].shape == (n_scans,)
    assert "patient_offsets" in out and "patient_list" in out


def test_latent_writer_persists_splits(tmp_path: Path) -> None:
    output = tmp_path / "splits.h5"
    n_scans = 2
    modalities = ("t1c",)
    kwargs = _common_kwargs(n_scans, modalities, n_patients=n_scans)
    with LatentH5Writer(output, **kwargs) as writer:
        for i in range(n_scans):
            writer.write_scan(
                i,
                scan_id=f"s{i}",
                patient_id=f"patient-{i}",
                timepoint=0,
                latent=np.zeros((1, 4, 8, 8, 8), dtype=np.float16),
                intensity_lower=[0.0],
                intensity_upper=[1.0],
            )
        writer.write_split("train", np.array([0]))
        writer.write_split("val", np.array([1]))

    with h5py.File(output, "r") as f:
        assert "splits/train" in f
        assert "splits/val" in f
        assert f["splits/train"][:].tolist() == [0]


def test_latent_writer_rejects_wrong_rank(tmp_path: Path) -> None:
    kwargs = _common_kwargs(1, ("t1c",), n_patients=1)
    with LatentH5Writer(tmp_path / "bad.h5", **kwargs) as writer:
        bad = np.zeros((4, 8, 8, 8), dtype=np.float32)  # missing M dim
        with pytest.raises(ValueError):
            writer.write_scan(
                0,
                scan_id="s",
                patient_id="patient-0",
                timepoint=0,
                latent=bad,
                intensity_lower=[0.0],
                intensity_upper=[1.0],
            )


def test_latent_writer_rejects_bad_csr(tmp_path: Path) -> None:
    kwargs = _common_kwargs(2, ("t1c",), n_patients=2)
    kwargs["patient_offsets"] = [0, 1, 99]  # last != n_scans
    with pytest.raises(ValueError, match="patient_offsets"):
        LatentH5Writer(tmp_path / "bad.h5", **kwargs)


def test_latent_writer_rejects_bad_spatial_op(tmp_path: Path) -> None:
    kwargs = _common_kwargs(1, ("t1c",), n_patients=1)
    kwargs["spatial_op"] = "warp"
    with pytest.raises(ValueError, match="spatial_op"):
        LatentH5Writer(tmp_path / "bad.h5", **kwargs)
