"""Tests for the E2.4 Phase B orchestrator (mocks docker + companion decode)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import h5py
import nibabel as nib
import numpy as np
import pytest

from experiments.E2_4_causal_steering.analysis.phase_b_runner import (
    ALL_MODALITIES,
    COMPANION_MODALITIES,
    PhaseBConfig,
    _stage_subject_dir,
    run_phase_b,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_phase_a_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    """Build a tiny Phase A sweep H5 + decoded NIfTIs for 2 anchors x 3 deltas."""
    out = tmp_path / "out"
    decoded = out / "decoded"
    decoded.mkdir(parents=True)

    affine = np.eye(4)
    scan_ids = ["BraTS-MEN-00001-000", "BraTS-MEN-00002-000"]
    deltas = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    a, n_d = len(scan_ids), len(deltas)
    nifti_paths = np.empty((a, n_d), dtype=object)
    for ai, sid in enumerate(scan_ids):
        d = decoded / sid
        d.mkdir(parents=True)
        for di, dv in enumerate(deltas):
            arr = np.zeros((32, 32, 24), dtype=np.float32)
            rel = f"decoded/{sid}/{sid}__delta_{float(dv):+.2f}.nii.gz"
            nib.save(nib.Nifti1Image(arr, affine), str(out / rel))
            nifti_paths[ai, di] = rel

    sweep = out / "sweep_results.h5"
    vlen = h5py.special_dtype(vlen=str)
    with h5py.File(sweep, "w") as f:
        f.attrs["schema_version"] = "0.1-provisional"
        f.attrs["phase"] = "A_decoded_only"
        f.attrs["segmenter_completed"] = False
        f.attrs["spacing_mm"] = np.asarray([1.0, 1.0, 1.0])
        f.attrs["anchor_row_indices"] = np.asarray([10, 20], dtype=np.int64)
        f.create_dataset("scan_id", data=np.asarray(scan_ids, dtype=object), dtype=vlen)
        f.create_dataset("delta_log_v_grid", data=deltas)
        f.create_dataset("decoded_log_v", data=np.full((a, n_d), np.nan, dtype=np.float32))
        f.create_dataset("tumor_voxels_decoded", data=np.full((a, n_d), -1, dtype=np.int32))
        f.create_dataset("decoded_nifti_path", data=nifti_paths, dtype=vlen)
        f.create_dataset("volume_bin", data=np.array([1, 5], dtype=np.int8))
    return out, sweep


def _fake_decode_companions(*args, output_dir: Path, scan_id: str, **kwargs):  # noqa: ARG001
    output_dir.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4)
    out = {}
    for m in ("t1n", "t2f", "t2w"):
        p = output_dir / f"{scan_id}-{m}.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((32, 32, 24), dtype=np.float32), affine), str(p))
        out[m] = p
    return out


class _FakeSegmenter:
    spec = type("Spec", (), {"label_map_name": "brats25", "model_id": "BraTS25_1"})

    def __init__(self, *args, **kwargs):  # noqa: ARG002
        self.calls = 0

    def predict_from_subject_dir(self, subject_dir: Path, *, keep_raw_dir=None):
        self.calls += 1
        # Synthesize a binary WT of size proportional to a small constant so log V is finite.
        mask = np.zeros((32, 32, 24), dtype=bool)
        mask[5:10, 5:10, 5:10] = True  # 125 voxels
        out_dir = keep_raw_dir if keep_raw_dir is not None else subject_dir.parent / "raw"
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = out_dir / "pred.nii.gz"
        seg_arr = mask.astype(np.int8) * 2  # label 2 = ET in BraTS25 -> WT
        nib.save(nib.Nifti1Image(seg_arr, np.eye(4)), str(raw_path))
        return mask, raw_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_phase_b_fills_h5_and_writes_artifacts(tmp_path: Path):
    out, sweep = _make_phase_a_artifacts(tmp_path)
    cfg = PhaseBConfig(
        sweep_h5=sweep,
        decoded_root=out / "decoded",
        latents_h5=tmp_path / "fake_lat.h5",
        checkpoint=tmp_path / "fake_ckpt.pt",
        model_id="BraTS25_1",
        gpu=False,
    )

    with (
        patch(
            "experiments.E2_4_causal_steering.analysis.phase_b_runner.decode_companion_modalities",
            side_effect=_fake_decode_companions,
        ),
        patch(
            "experiments.E2_4_causal_steering.analysis.phase_b_runner.BratsDockerSegmenter",
            _FakeSegmenter,
        ),
    ):
        result_path = run_phase_b(cfg)

    assert result_path == sweep
    with h5py.File(sweep, "r") as f:
        decoded = f["decoded_log_v"][:]
        n_vox = f["tumor_voxels_decoded"][:]
        assert bool(f.attrs["segmenter_completed"]) is True
        assert str(f.attrs["segmenter_model_id"]) == "BraTS25_1"
    assert np.all(np.isfinite(decoded))
    assert np.all(n_vox == 125)

    # Staging dirs created with a per-subject subdirectory holding 4 link-or-copy files.
    for sid in ["BraTS-MEN-00001-000", "BraTS-MEN-00002-000"]:
        for d in (-0.5, 0.0, 0.5):
            staged = out / "decoded" / sid / "staged" / f"delta_{d:+.2f}"
            subject_dir = staged / sid
            assert subject_dir.is_dir()
            for m in ALL_MODALITIES:
                f = subject_dir / f"{sid}-{m}.nii.gz"
                assert f.is_file()  # hardlink or copy

    summary = json.loads((out / "analysis" / "phase_b_summary.json").read_text())
    assert summary["n_anchors"] == 2
    assert summary["n_deltas"] == 3


def test_phase_b_idempotent_skips_already_segmented(tmp_path: Path):
    out, sweep = _make_phase_a_artifacts(tmp_path)
    cfg = PhaseBConfig(
        sweep_h5=sweep,
        decoded_root=out / "decoded",
        latents_h5=tmp_path / "fake_lat.h5",
        checkpoint=tmp_path / "fake_ckpt.pt",
        model_id="BraTS25_1",
        gpu=False,
    )

    seg = _FakeSegmenter()

    def _seg_factory(*args, **kwargs):  # noqa: ARG001
        return seg

    with (
        patch(
            "experiments.E2_4_causal_steering.analysis.phase_b_runner.decode_companion_modalities",
            side_effect=_fake_decode_companions,
        ),
        patch(
            "experiments.E2_4_causal_steering.analysis.phase_b_runner.BratsDockerSegmenter",
            side_effect=_seg_factory,
        ),
    ):
        run_phase_b(cfg)
        first_calls = seg.calls
        run_phase_b(cfg)  # second run — should reuse
        assert seg.calls == first_calls  # no new docker calls


def test_stage_subject_dir_missing_t1c_raises(tmp_path: Path):
    anchor_dir = tmp_path / "anchor"
    anchor_dir.mkdir()
    staged = anchor_dir / "staged" / "delta_+0.00"
    with pytest.raises(FileNotFoundError, match="steered t1c"):
        _stage_subject_dir(anchor_dir=anchor_dir, staged_dir=staged, scan_id="X", delta=0.0)


def test_companion_modalities_constant():
    assert tuple(COMPANION_MODALITIES) == ("t1n", "t2f", "t2w")
    assert tuple(ALL_MODALITIES) == ("t1c", "t1n", "t2f", "t2w")
