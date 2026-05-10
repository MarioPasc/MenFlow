"""Tests for BratsDockerSegmenter — mocks the Docker runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

from experiments.E2_4_causal_steering.analysis.segmenter_brats import (
    BratsDockerSegmenter,
)


def _make_subject_dir(tmp_path: Path, scan_id: str) -> Path:
    d = tmp_path / scan_id
    d.mkdir(parents=True)
    affine = np.eye(4)
    for m in ("t1c", "t1n", "t2f", "t2w"):
        nib.save(
            nib.Nifti1Image(np.zeros((240, 240, 155), dtype=np.float32), affine),
            str(d / f"{scan_id}-{m}.nii.gz"),
        )
    return d


class _FakeRunner:
    def __init__(self, *args, **kwargs):  # noqa: ARG002
        pass

    def ensure_image(self):
        return None

    def run(self, input_dir: Path, output_dir: Path, *, timeout_s=None):  # noqa: ARG002
        seg = np.zeros((240, 240, 155), dtype=np.int8)
        seg[100:120, 100:120, 70:90] = 2  # 8000 voxels under BraTS25 -> WT
        out = output_dir / "pred.nii.gz"
        nib.save(nib.Nifti1Image(seg, np.eye(4)), str(out))
        return [out], 0.05


def test_predict_from_subject_dir_returns_wt_mask(tmp_path: Path):
    subj = _make_subject_dir(tmp_path / "in", "BraTS-MEN-12345-000")
    work = tmp_path / "work"
    keep_raw = tmp_path / "raw"

    with patch(
        "experiments.E2_4_causal_steering.analysis.segmenter_brats.BratsDockerRunner",
        _FakeRunner,
    ):
        seg = BratsDockerSegmenter("BraTS25_1", work_dir=work, gpu=False)
        mask, raw_path = seg.predict_from_subject_dir(subj, keep_raw_dir=keep_raw)

    assert mask.shape == (240, 240, 155)
    assert mask.dtype == bool
    assert int(mask.sum()) == 20 * 20 * 20
    assert raw_path.is_file()
    assert raw_path.parent == keep_raw


def test_predict_protocol_path_raises(tmp_path: Path):
    work = tmp_path / "work"
    with patch(
        "experiments.E2_4_causal_steering.analysis.segmenter_brats.BratsDockerRunner",
        _FakeRunner,
    ):
        seg = BratsDockerSegmenter("BraTS25_1", work_dir=work, gpu=False)
        with pytest.raises(NotImplementedError, match="multimodal"):
            seg.predict(np.zeros((10, 10, 10), dtype=np.float32))
