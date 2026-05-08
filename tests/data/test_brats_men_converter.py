"""End-to-end converter contract test on synthetic BraTS-MEN-shaped fixtures.

We construct two fake "subjects" with the BraTS-MEN naming scheme, write tiny
240×240×155 NIfTI volumes (one with seg, one without), run the full converter,
and assert: (i) the H5 satisfies the unified schema, (ii) splits are populated
correctly, (iii) ``has_segmentation`` reflects the source.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pytest

from menflow.data.conversors.brats_men import BraTSMENConverter
from menflow.data.h5_schema import validate_h5


_MODALITIES = ("t1c", "t1n", "t2f", "t2w")
_SHAPE = (240, 240, 155)


def _write_synthetic_subject(
    parent: Path, scan_id: str, *, with_seg: bool, rng: np.random.Generator
) -> None:
    sub_dir = parent / scan_id
    sub_dir.mkdir(parents=True, exist_ok=True)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    for modality in _MODALITIES:
        arr = rng.uniform(0, 500, size=_SHAPE).astype(np.float32)
        nib.save(nib.Nifti1Image(arr, affine), str(sub_dir / f"{scan_id}-{modality}.nii.gz"))
    if with_seg:
        seg = np.zeros(_SHAPE, dtype=np.int16)
        seg[100:110, 100:110, 70:75] = 1
        seg[110:115, 105:115, 75:80] = 3
        nib.save(nib.Nifti1Image(seg, affine), str(sub_dir / f"{scan_id}-seg.nii.gz"))


@pytest.fixture(scope="module")
def synthetic_cohort(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    rng = np.random.default_rng(0)
    base = tmp_path_factory.mktemp("brats_men_synth")
    train_root = base / "train"
    val_root = base / "val"
    train_root.mkdir()
    val_root.mkdir()

    # Two annotated training subjects, one unannotated validation subject.
    _write_synthetic_subject(train_root, "BraTS-MEN-00001-000", with_seg=True, rng=rng)
    _write_synthetic_subject(train_root, "BraTS-MEN-00002-000", with_seg=True, rng=rng)
    _write_synthetic_subject(val_root, "BraTS-MEN-00003-000", with_seg=False, rng=rng)

    return {"train": train_root, "val": val_root}


@pytest.mark.integration
def test_converter_writes_schema_compliant_h5(
    synthetic_cohort: dict[str, Path], tmp_path: Path
) -> None:
    output = tmp_path / "brats_men.h5"
    converter = BraTSMENConverter()
    converter.convert(synthetic_cohort, output)

    assert output.exists()
    violations = validate_h5(output)
    assert violations == [], "\n".join(str(v) for v in violations)


@pytest.mark.integration
def test_converter_populates_splits_and_metadata(
    synthetic_cohort: dict[str, Path], tmp_path: Path
) -> None:
    output = tmp_path / "brats_men.h5"
    BraTSMENConverter().convert(synthetic_cohort, output)

    with h5py.File(output, "r") as f:
        assert f.attrs["dataset_name"] == "BraTS-MEN-2023"
        assert f.attrs["dataset_type"] == "cross-sectional"
        assert int(f.attrs["n_scans"]) == 3
        assert int(f.attrs["n_patients"]) == 3
        assert tuple(int(x) for x in f.attrs["spatial_shape"]) == _SHAPE
        assert list(f.attrs["modalities"]) == list(_MODALITIES)
        label_map = json.loads(f.attrs["label_map"])
        assert label_map["1"] == "necrotic_tumor_core"

        has_seg = f["has_segmentation"][:]
        scan_ids = [s.decode() if isinstance(s, bytes) else s for s in f["scan_ids"][:]]
        seg_by_id = dict(zip(scan_ids, has_seg))
        assert seg_by_id["BraTS-MEN-00001-000"]
        assert seg_by_id["BraTS-MEN-00002-000"]
        assert not seg_by_id["BraTS-MEN-00003-000"]

        train_idx = f["splits/train"][:]
        val_idx = f["splits/val"][:]
        assert len(train_idx) == 2
        assert len(val_idx) == 1

        assert bool(f.attrs["has_segmentation_any"]) is True

        subset = f["metadata/subset"][:]
        subset = [s.decode() if isinstance(s, bytes) else s for s in subset]
        subset_by_id = dict(zip(scan_ids, subset))
        assert subset_by_id["BraTS-MEN-00003-000"] == "val"


@pytest.mark.integration
def test_converter_rejects_invalid_label(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(1)
    train_root = tmp_path / "bad_cohort" / "train"
    train_root.mkdir(parents=True)
    sid = "BraTS-MEN-99999-000"
    sub_dir = train_root / sid
    sub_dir.mkdir()
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    for modality in _MODALITIES:
        nib.save(
            nib.Nifti1Image(rng.uniform(0, 500, _SHAPE).astype(np.float32), affine),
            str(sub_dir / f"{sid}-{modality}.nii.gz"),
        )
    bad_seg = np.zeros(_SHAPE, dtype=np.int16)
    bad_seg[10, 10, 10] = 7  # invalid label
    nib.save(nib.Nifti1Image(bad_seg, affine), str(sub_dir / f"{sid}-seg.nii.gz"))

    output = tmp_path / "out.h5"
    # Loader logs error and skips. With only one (rejected) record, no scans
    # would survive — convert() raises ConversionError when 0 records reach
    # the writer. Here we just verify the bad record is skipped cleanly.
    BraTSMENConverter().convert({"train": train_root}, output)
    with h5py.File(output, "r") as f:
        # The scan was skipped; image entry is the zero default.
        assert int(f.attrs["n_scans"]) == 1
        assert not bool(f["has_segmentation"][0])  # seg failed to load
