"""End-to-end test for :class:`menflow.data.conversors.mengrowth.MenGrowthConverter`.

Builds a synthetic two-patient longitudinal cohort (one with two timepoints, one
with three) and verifies the H5 schema, CSR offsets, ``empty_seg`` flagging,
and ``has_segmentation_any`` cohort-level flag.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pytest

from menflow.data.conversors.mengrowth import MenGrowthConverter
from menflow.data.h5_schema import validate_h5

_MODALITIES = ("t1c", "t1n", "t2f", "t2w")
_SHAPE = (240, 240, 155)


def _write_synthetic_scan(
    scan_dir: Path, *, seg: np.ndarray | None, rng: np.random.Generator
) -> None:
    scan_dir.mkdir(parents=True, exist_ok=True)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    for modality in _MODALITIES:
        arr = rng.uniform(0, 500, size=_SHAPE).astype(np.float32)
        nib.save(nib.Nifti1Image(arr, affine), str(scan_dir / f"{modality}.nii.gz"))
    if seg is not None:
        nib.save(nib.Nifti1Image(seg, affine), str(scan_dir / "seg.nii.gz"))


@pytest.fixture(scope="module")
def synthetic_mengrowth(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(7)
    root = tmp_path_factory.mktemp("mengrowth_synth")

    # Patient 0001: two timepoints, second has empty seg (no tumor visible).
    pat1 = root / "MenGrowth-0001"
    seg_t0 = np.zeros(_SHAPE, dtype=np.int16)
    seg_t0[100:110, 100:110, 70:75] = 1
    seg_t0[110:115, 105:115, 75:80] = 3
    _write_synthetic_scan(pat1 / "MenGrowth-0001-000", seg=seg_t0, rng=rng)
    _write_synthetic_scan(
        pat1 / "MenGrowth-0001-001", seg=np.zeros(_SHAPE, dtype=np.int16), rng=rng
    )

    # Patient 0002: three timepoints, all with tumor.
    pat2 = root / "MenGrowth-0002"
    for tp in range(3):
        seg = np.zeros(_SHAPE, dtype=np.int16)
        seg[120 + tp : 125 + tp, 120 : 130, 80 : 85] = 2
        seg[125 + tp : 130 + tp, 125 : 135, 85 : 90] = 3
        _write_synthetic_scan(pat2 / f"MenGrowth-0002-{tp:03d}", seg=seg, rng=rng)

    return root


@pytest.mark.integration
def test_mengrowth_writes_schema_compliant_h5(
    synthetic_mengrowth: Path, tmp_path: Path
) -> None:
    output = tmp_path / "mengrowth.h5"
    MenGrowthConverter().convert({"root": synthetic_mengrowth}, output)
    violations = validate_h5(output)
    assert violations == [], "\n".join(str(v) for v in violations)


@pytest.mark.integration
def test_mengrowth_csr_layout_is_correct(
    synthetic_mengrowth: Path, tmp_path: Path
) -> None:
    output = tmp_path / "mengrowth.h5"
    MenGrowthConverter().convert({"root": synthetic_mengrowth}, output)

    with h5py.File(output, "r") as f:
        assert f.attrs["dataset_name"] == "MenGrowth"
        assert f.attrs["dataset_type"] == "longitudinal"
        assert int(f.attrs["n_scans"]) == 5
        assert int(f.attrs["n_patients"]) == 2

        offsets = f["longitudinal/patient_offsets"][:]
        # Patient 0001 owns scans [0,1]; patient 0002 owns [2,3,4].
        assert offsets.tolist() == [0, 2, 5]

        plist = [s.decode() if isinstance(s, bytes) else s for s in f["longitudinal/patient_list"][:]]
        assert plist == ["MenGrowth-0001", "MenGrowth-0002"]

        tp = f["timepoint_idx"][:].tolist()
        assert tp == [0, 1, 0, 1, 2]

        empty = f["metadata/empty_seg"][:]
        # Only the second scan of patient 0001 is empty.
        assert empty.tolist() == [False, True, False, False, False]

        label_map = json.loads(f.attrs["label_map"])
        assert label_map["1"] == "non_enhancing_tumor_core"
        assert label_map["2"] == "surrounding_non_enhancing_flair_hyperintensity"
        assert label_map["3"] == "enhancing_tumor"

        assert bool(f.attrs["has_segmentation_any"]) is True
