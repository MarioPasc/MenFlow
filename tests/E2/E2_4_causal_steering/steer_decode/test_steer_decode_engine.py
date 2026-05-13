"""Smoke test for SteerDecodeEngine — mocks decoder + encoder.

The engine is exercised end-to-end on a synthetic latents H5 + source H5 with
fake MAISI (identity decode + identity encode). The test verifies:

* Sweep H5 has the expected datasets and shapes.
* ``decoded_log_v`` and ``tumor_voxels_decoded`` carry sentinel values
  (NaN / -1) — Phase A invariant.
* NIfTI files appear at the announced paths.
* drift values are finite (zero in the identity-encoder case is also fine).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import h5py
import nibabel as nib
import numpy as np
import pytest
import torch

from experiments.E2.E2_4_causal_steering.steer_decode.engine.steer_decode_engine import (
    SteerDecodeEngine,
    SteerDecodeRoutineConfig,
)

# ---------------------------------------------------------------------------
# Fake encoder/decoder + checkpoint loader
# ---------------------------------------------------------------------------


class _FakeAutoencoder:
    """Identity-like decode (zero-pads channel dim back to 1) + identity encode."""

    def __init__(self, latent_channels: int, padded_shape: tuple[int, int, int]):
        self.latent_channels = latent_channels
        self.padded_shape = padded_shape
        self._params: list[torch.Tensor] = [torch.zeros(1, dtype=torch.float32)]

    def parameters(self):
        return iter(self._params)

    def to(self, *args, **kwargs):  # noqa: D401
        return self

    def eval(self):
        return self

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Map latent (B, C, H', W', D') -> image (B, 1, H, W, D) by upsampling
        # via F.interpolate then averaging across the channel axis.
        b = z.shape[0]
        out = torch.nn.functional.interpolate(
            z.float(), size=self.padded_shape, mode="trilinear", align_corners=False
        )
        # Average channels -> single-channel image; keep a (B,1,H,W,D) layout.
        img = out.mean(dim=1, keepdim=True)
        return img

    def encode(self, x: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        # (B, 1, H, W, D) -> (B, C, H/4, W/4, D/4) by avg-pool then channel tile.
        pooled = torch.nn.functional.avg_pool3d(x.float(), kernel_size=4)
        return pooled.expand(-1, self.latent_channels, -1, -1, -1).contiguous()


@pytest.fixture
def fake_data(tmp_path: Path):
    """Build a minimal latents H5 + source H5 + direction NPZ + dummy checkpoint."""
    latent_channels = 4
    source_shape = (32, 32, 24)
    padded_shape = (32, 32, 32)  # source's last axis padded up by 8
    latent_spatial = (8, 8, 8)
    n_scans = 6
    modalities = ["t1c", "t1n", "t2f", "t2w"]
    spacing = (1.0, 1.0, 1.0)
    rng = np.random.default_rng(0)

    latents_h5 = tmp_path / "latents.h5"
    source_h5 = tmp_path / "source.h5"
    direction_npz = tmp_path / "direction.npz"
    checkpoint = tmp_path / "fake_ckpt.pt"

    # Source H5 — minimal unified-schema subset the engine actually reads.
    with h5py.File(source_h5, "w") as f:
        f.attrs["schema_version"] = "1.0"
        f.attrs["dataset_name"] = "TEST"
        f.attrs["dataset_type"] = "cross-sectional"
        f.attrs["n_scans"] = n_scans
        f.attrs["n_patients"] = n_scans
        f.attrs["modalities"] = np.array(modalities, dtype=object)
        f.attrs["n_modalities"] = len(modalities)
        f.attrs["spatial_shape"] = np.asarray(source_shape, dtype=np.int64)
        f.attrs["spacing_mm"] = np.asarray(spacing, dtype=np.float64)
        f.attrs["orientation"] = "RAS"
        f.attrs["label_map"] = '{"1":"tumor"}'
        f.attrs["intensity_normalized"] = False
        f.attrs["has_segmentation_any"] = True
        # Build segmentations with varying voxel counts that span the bin grid.
        segs = np.zeros((n_scans,) + source_shape, dtype=np.int8)
        log_v_targets = np.linspace(-1.5, 3.5, n_scans)
        for i, log_v in enumerate(log_v_targets):
            n_vox = max(1, int(round(np.exp(log_v) * 1000)))  # 1000 mm3 per cm3
            n_side = max(2, int(np.cbrt(n_vox)))
            n_side = min(n_side, source_shape[0] - 2)
            segs[i, 2 : 2 + n_side, 2 : 2 + n_side, 2 : 2 + n_side] = 1
        f.create_dataset("segmentations", data=segs)
        f.create_dataset(
            "images",
            data=rng.standard_normal((n_scans, len(modalities)) + source_shape, dtype=np.float32),
        )
        f.create_dataset(
            "scan_ids", data=np.array([f"S{i:03d}" for i in range(n_scans)], dtype=object)
        )
        f.create_dataset(
            "patient_ids", data=np.array([f"P{i:03d}" for i in range(n_scans)], dtype=object)
        )
        f.create_dataset("timepoint_idx", data=np.zeros(n_scans, dtype=np.int32))
        f.create_dataset("has_segmentation", data=np.ones(n_scans, dtype=bool))
        long_grp = f.create_group("longitudinal")
        long_grp.create_dataset("patient_offsets", data=np.arange(n_scans + 1, dtype=np.int32))
        long_grp.create_dataset(
            "patient_list", data=np.array([f"P{i:03d}" for i in range(n_scans)], dtype=object)
        )

    # Latents H5 — what the engine reads.
    with h5py.File(latents_h5, "w") as f:
        f.attrs["schema_version"] = "0.2-provisional"
        f.attrs["dataset_name"] = "TEST"
        f.attrs["dataset_type"] = "cross-sectional"
        f.attrs["n_scans"] = n_scans
        f.attrs["n_patients"] = n_scans
        f.attrs["modalities"] = np.array(modalities, dtype=object)
        f.attrs["latent_channels"] = latent_channels
        f.attrs["latent_spatial_shape"] = np.asarray(latent_spatial, dtype=np.int64)
        f.attrs["source_spatial_shape"] = np.asarray(source_shape, dtype=np.int64)
        f.attrs["working_spatial_shape"] = np.asarray(source_shape, dtype=np.int64)
        f.attrs["padded_spatial_shape"] = np.asarray(padded_shape, dtype=np.int64)
        f.attrs["spatial_op"] = "none"
        f.attrs["crop_offset"] = np.zeros(3, dtype=np.int64)
        f.attrs["encoder_name"] = "FAKE"
        f.attrs["deterministic"] = True
        f.create_dataset(
            "latents",
            data=rng.standard_normal(
                (n_scans, len(modalities), latent_channels) + latent_spatial,
                dtype=np.float32,
            ).astype(np.float16),
        )
        f.create_dataset(
            "intensity_lower",
            data=np.zeros((n_scans, len(modalities)), dtype=np.float32),
        )
        f.create_dataset(
            "intensity_upper",
            data=np.ones((n_scans, len(modalities)), dtype=np.float32),
        )
        f.create_dataset(
            "scan_ids", data=np.array([f"S{i:03d}" for i in range(n_scans)], dtype=object)
        )
        f.create_dataset(
            "patient_ids", data=np.array([f"P{i:03d}" for i in range(n_scans)], dtype=object)
        )
        f.create_dataset("timepoint_idx", data=np.zeros(n_scans, dtype=np.int32))
        long_grp = f.create_group("longitudinal")
        long_grp.create_dataset("patient_offsets", data=np.arange(n_scans + 1, dtype=np.int32))
        long_grp.create_dataset(
            "patient_list", data=np.array([f"P{i:03d}" for i in range(n_scans)], dtype=object)
        )
        splits_grp = f.create_group("splits")
        # Put all patients into "val" so anchor sampling has the entire pool.
        splits_grp.create_dataset("val", data=np.arange(n_scans, dtype=np.int32))

    # Direction NPZ.
    direction = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    direction = direction / np.linalg.norm(direction)
    np.savez(direction_npz, direction=direction, direction_norm=np.float32(20.846))

    # Dummy checkpoint placeholder (file existence only — patched away).
    checkpoint.write_bytes(b"")

    return {
        "latents_h5": latents_h5,
        "source_h5": source_h5,
        "direction_npz": direction_npz,
        "checkpoint": checkpoint,
        "padded_shape": padded_shape,
        "latent_channels": latent_channels,
        "n_scans": n_scans,
    }


def _patched_from_checkpoint(latent_channels: int, padded_shape: tuple[int, int, int]):
    """Return a callable replacing MaisiAutoencoder.from_checkpoint with a stub."""

    def _factory(*args, **kwargs):
        return _FakeAutoencoder(latent_channels, padded_shape)

    return _factory


def test_engine_writes_phase_a_h5(fake_data, tmp_path: Path):
    out_dir = tmp_path / "out"
    cfg = SteerDecodeRoutineConfig(
        latents_h5=fake_data["latents_h5"],
        source_h5=fake_data["source_h5"],
        direction_npz=fake_data["direction_npz"],
        checkpoint=fake_data["checkpoint"],
        output_dir=out_dir,
        modality="t1c",
        modality_index=0,
        operator="local",
        deltas=(-1.0, 0.0, 1.0),
        n_anchors_per_bin=1,
        # Three bins covering the 6-scan synthetic log V range.
        volume_bin_edges_log_cm3=(-2.0, 0.0, 2.0, 4.0),
        splits_to_use=("val",),
        dtype="float32",
        device="cpu",
        seed=0,
        save_decoded_nifti=True,
        intensity_rescale=True,
    )

    with patch(
        "experiments.E2.E2_4_causal_steering.steer_decode.engine.steer_decode_engine.MaisiAutoencoder.from_checkpoint",
        side_effect=_patched_from_checkpoint(
            fake_data["latent_channels"], fake_data["padded_shape"]
        ),
    ):
        sweep_h5 = SteerDecodeEngine(cfg).run()

    assert sweep_h5.is_file()
    with h5py.File(sweep_h5, "r") as f:
        assert f.attrs["phase"] == "A_decoded_only"
        assert (
            f.attrs["segmenter_completed"] is np.False_ or f.attrs["segmenter_completed"] == False
        )  # noqa: E712
        scan_ids = f["scan_id"].asstr()[:]
        decoded = f["decoded_log_v"][:]
        tvox = f["tumor_voxels_decoded"][:]
        drift = f["drift"][:]
        intensity = f["intensity_proxy"][:]
        nifti_paths = f["decoded_nifti_path"].asstr()[:]
        deltas = f["delta_log_v_grid"][:]
        alphas = f["alpha_grid"][:]
        predicted = f["predicted_log_v"][:]

    n_anchors = scan_ids.size
    assert n_anchors >= 1
    assert decoded.shape == (n_anchors, len(deltas))
    assert np.all(np.isnan(decoded))
    assert np.all(tvox == -1)
    assert np.all(np.isfinite(drift))
    assert np.all(np.isfinite(intensity))
    assert predicted.shape == decoded.shape
    # alpha_grid = delta / 20.846
    np.testing.assert_allclose(alphas, deltas / 20.846, rtol=1e-5)

    # NIfTI files exist on disk.
    for ai in range(n_anchors):
        for di in range(len(deltas)):
            rel = nifti_paths[ai, di]
            assert rel != ""
            full = out_dir / rel
            assert full.is_file()
            # Cheap shape check.
            arr = np.asarray(nib.load(str(full)).dataobj)
            assert arr.shape == (32, 32, 24)
