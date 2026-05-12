"""Smoke test for SegmentBratsEngine — mocks BratsDockerRunner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from experiments.E2.E2_4_causal_steering.segment_brats.engine.segment_brats_engine import (
    BRATS_CANONICAL_SHAPE,
    SegmentBratsEngine,
    SegmentBratsRoutineConfig,
)


def _make_subject(input_dir: Path, scan_id: str) -> Path:
    d = input_dir / scan_id
    d.mkdir(parents=True)
    affine = np.eye(4)
    arr = np.zeros(BRATS_CANONICAL_SHAPE, dtype=np.float32)
    for m in ("t1c", "t1n", "t2f", "t2w"):
        nib.save(nib.Nifti1Image(arr, affine), str(d / f"{scan_id}-{m}.nii.gz"))
    return d


class _FakeRunner:
    def __init__(self, *args, **kwargs):  # noqa: ARG002
        self.calls = 0

    def ensure_image(self) -> None:
        return None

    def run(self, input_dir: Path, output_dir: Path, *, timeout_s=None):  # noqa: ARG002
        # Write a fake multi-class seg with labels {0, 1, 2} into output_dir.
        scan_id = input_dir.name
        seg = np.zeros(BRATS_CANONICAL_SHAPE, dtype=np.int8)
        seg[40:60, 40:60, 40:60] = 2  # 8000 ET voxels
        seg[60:65, 60:65, 60:65] = 1  # 125 SNFH voxels
        out_file = output_dir / f"{scan_id}.nii.gz"
        nib.save(nib.Nifti1Image(seg, np.eye(4)), str(out_file))
        self.calls += 1
        return [out_file], 0.123


def test_engine_directory_mode(tmp_path: Path):
    in_dir = tmp_path / "subjects"
    _make_subject(in_dir, "BraTS-MEN-00001-000")
    _make_subject(in_dir, "BraTS-MEN-00002-000")
    out_dir = tmp_path / "out"

    cfg = SegmentBratsRoutineConfig(
        output_dir=out_dir, model="BraTS25_1", input_dir=in_dir, gpu=False
    )

    with patch(
        "experiments.E2.E2_4_causal_steering.segment_brats.engine.segment_brats_engine.BratsDockerRunner",
        _FakeRunner,
    ):
        manifest_path = SegmentBratsEngine(cfg).run()

    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["n_subjects"] == 2
    for entry in manifest["subjects"]:
        assert entry["wt_voxel_count"] > 0
        wt = nib.load(entry["wt_seg"])
        assert tuple(wt.shape) == BRATS_CANONICAL_SHAPE
        # BraTS25 maps {1,2} -> WT, so all 8125 voxels should be in mask.
        assert int(np.asarray(wt.dataobj).sum()) == 8125


def test_engine_xor_validation(tmp_path: Path):
    cfg = SegmentBratsRoutineConfig(
        output_dir=tmp_path / "out",
        model="BraTS25_1",
        input_dir=None,
        input_h5=None,
    )
    import pytest

    with pytest.raises(ValueError, match="exactly one"):
        SegmentBratsEngine(cfg).run()


def test_engine_unknown_model():
    import pytest

    with pytest.raises(ValueError, match="unknown model"):
        SegmentBratsEngine(
            SegmentBratsRoutineConfig(
                output_dir=Path("/tmp"),
                model="NotARealModel",
                input_dir=Path("/tmp"),
            )
        )
