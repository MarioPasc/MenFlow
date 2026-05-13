"""Phase B handoff round-trip: feed a fake segmenter, verify the H5 fills."""

from __future__ import annotations

from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

from experiments.E2.E2_4_causal_steering.analysis.segmenter_interface import (
    Segmenter,
    fill_sweep_results,
)


class _FakeSegmenter:
    """Returns a fixed-size box mask whose volume scales with mean intensity.

    The test only cares that ``decoded_log_v`` becomes finite and
    ``tumor_voxels_decoded`` becomes non-negative — the segmenter does not need
    to match any biology.
    """

    modality = "t1c"
    expected_shape = (16, 16, 12)

    def predict(self, image: np.ndarray) -> np.ndarray:
        m = np.zeros_like(image, dtype=np.uint8)
        side = max(1, int(round(abs(float(image.mean())) * 4 + 1)))
        side = min(side, image.shape[0])
        m[:side, :side, :side] = 1
        return m


def _make_minimal_sweep(tmp_path: Path) -> Path:
    """Write a tiny sweep H5 + 2 NIfTIs, all sentinels for Phase B fields."""
    sweep = tmp_path / "sweep_results.h5"
    decoded_root = tmp_path / "decoded"
    decoded_root.mkdir(parents=True)
    n_anchors, n_deltas = 2, 3
    affine = np.eye(4)
    paths = np.empty((n_anchors, n_deltas), dtype=object)
    for ai in range(n_anchors):
        for di in range(n_deltas):
            arr = np.random.randn(16, 16, 12).astype(np.float32) + ai + di * 0.5
            rel = f"decoded/anchor_{ai}_delta_{di}.nii.gz"
            full = tmp_path / rel
            nib.save(nib.Nifti1Image(arr, affine), str(full))
            paths[ai, di] = rel

    vlen = h5py.special_dtype(vlen=str)
    with h5py.File(sweep, "w") as f:
        f.attrs["schema_version"] = "0.1-provisional"
        f.attrs["phase"] = "A_decoded_only"
        f.attrs["segmenter_completed"] = False
        f.attrs["spacing_mm"] = np.asarray([1.0, 1.0, 1.0])
        f.create_dataset(
            "decoded_log_v", data=np.full((n_anchors, n_deltas), np.nan, dtype=np.float32)
        )
        f.create_dataset(
            "tumor_voxels_decoded", data=np.full((n_anchors, n_deltas), -1, dtype=np.int32)
        )
        f.create_dataset("decoded_nifti_path", data=paths, dtype=vlen)
    return sweep


def test_segmenter_protocol_satisfied():
    seg = _FakeSegmenter()
    assert isinstance(seg, Segmenter)


def test_fill_sweep_results_writes_finite_values(tmp_path: Path):
    sweep = _make_minimal_sweep(tmp_path)
    fill_sweep_results(sweep, _FakeSegmenter(), tmp_path / "decoded")
    with h5py.File(sweep, "r") as f:
        decoded_log_v = f["decoded_log_v"][:]
        tumor_voxels = f["tumor_voxels_decoded"][:]
        assert bool(f.attrs["segmenter_completed"]) is True
        assert "segmenter_completed_at" in f.attrs
    assert np.all(np.isfinite(decoded_log_v))
    assert np.all(tumor_voxels > 0)
