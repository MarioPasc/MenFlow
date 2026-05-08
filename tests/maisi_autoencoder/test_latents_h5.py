"""Round-trip test for the provisional latent HDF5 writer/reader."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from menflow.maisi_autoencoder.config import InferenceConfig, MaisiV2Config
from menflow.maisi_autoencoder.latents_h5 import (
    LATENT_SCHEMA_VERSION,
    LatentH5Writer,
    read_latent_h5,
)


def test_latent_h5_round_trip(tmp_path: Path) -> None:
    n_scans = 3
    modalities = ("t1c", "t2f")
    latent_shape = (4, 8, 8, 6)  # (C, H', W', D')
    output = tmp_path / "latents.h5"

    rng = np.random.default_rng(0)
    latents = rng.normal(size=(n_scans, len(modalities), *latent_shape)).astype(np.float32)
    lower = rng.uniform(0, 50, size=(n_scans, len(modalities))).astype(np.float32)
    upper = rng.uniform(500, 1000, size=(n_scans, len(modalities))).astype(np.float32)

    with LatentH5Writer(
        output,
        n_scans=n_scans,
        modalities=modalities,
        latent_shape=latent_shape,
        source_h5="/tmp/source.h5",
        source_dataset_name="ToyCohort",
        source_spatial_shape=(32, 32, 24),
        encoder_checkpoint="/tmp/ae.pt",
        encoder_config=MaisiV2Config(),
        inference_config=InferenceConfig(),
        deterministic=True,
        latent_dtype="float32",
    ) as writer:
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

    # Direct h5 inspection
    with h5py.File(output, "r") as f:
        assert f.attrs["schema_version"] == LATENT_SCHEMA_VERSION
        assert f.attrs["source_dataset_name"] == "ToyCohort"
        assert int(f.attrs["n_scans"]) == n_scans
        assert int(f.attrs["n_modalities"]) == 2
        assert tuple(int(x) for x in f.attrs["latent_shape"]) == latent_shape
        assert bool(f.attrs["deterministic"]) is True

        cfg_round = json.loads(f.attrs["encoder_config"])
        assert cfg_round["latent_channels"] == 4

        np.testing.assert_allclose(f["latents"][:], latents, atol=1e-6)
        np.testing.assert_allclose(f["intensity_lower"][:], lower)
        np.testing.assert_allclose(f["intensity_upper"][:], upper)

    # Reader API
    out = read_latent_h5(output)
    assert out["latents"].shape == (n_scans, 2, *latent_shape)
    assert out["scan_ids"].shape == (n_scans,)


def test_latent_writer_rejects_wrong_rank(tmp_path: Path) -> None:
    output = tmp_path / "bad.h5"
    with LatentH5Writer(
        output,
        n_scans=1,
        modalities=("t1c",),
        latent_shape=(4, 8, 8, 6),
        source_h5="x",
        source_dataset_name="x",
        source_spatial_shape=(32, 32, 24),
        encoder_checkpoint="x",
        encoder_config={},
        inference_config={},
        deterministic=True,
    ) as writer:
        bad = np.zeros((4, 8, 8, 6), dtype=np.float32)  # missing M dim
        try:
            writer.write_scan(
                0,
                scan_id="s",
                patient_id="p",
                timepoint=0,
                latent=bad,
                intensity_lower=[0.0],
                intensity_upper=[1.0],
            )
        except ValueError:
            return
        raise AssertionError("expected ValueError")
