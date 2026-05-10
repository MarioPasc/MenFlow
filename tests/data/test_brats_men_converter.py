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

    # Six annotated training subjects (large enough for the e3 3-fold split)
    # plus two unannotated validation subjects.
    for i in range(1, 7):
        _write_synthetic_subject(train_root, f"BraTS-MEN-{i:05d}-000", with_seg=True, rng=rng)
    _write_synthetic_subject(val_root, "BraTS-MEN-90001-000", with_seg=False, rng=rng)
    _write_synthetic_subject(val_root, "BraTS-MEN-90002-000", with_seg=False, rng=rng)

    return {"train": train_root, "val": val_root}


def _build_converter() -> BraTSMENConverter:
    """Use a (1, 3) k-value layout matched to the 6-patient synthetic cohort.

    With test_pct=0.1 and 6 eligible patients, the test set holds out a single
    patient and the remaining 5 are split per k.
    """
    return BraTSMENConverter(
        kfold_k_values=(1, 3), kfold_test_pct=0.2, kfold_seed=0, kfold_n_strata=2
    )


@pytest.mark.integration
def test_converter_writes_schema_compliant_h5(
    synthetic_cohort: dict[str, Path], tmp_path: Path
) -> None:
    output = tmp_path / "brats_men.h5"
    _build_converter().convert(synthetic_cohort, output)

    assert output.exists()
    violations = validate_h5(output)
    assert violations == [], "\n".join(str(v) for v in violations)


@pytest.mark.integration
def test_converter_populates_splits_and_metadata(
    synthetic_cohort: dict[str, Path], tmp_path: Path
) -> None:
    output = tmp_path / "brats_men.h5"
    _build_converter().convert(synthetic_cohort, output)

    with h5py.File(output, "r") as f:
        assert f.attrs["dataset_name"] == "BraTS-MEN-2023"
        assert f.attrs["dataset_type"] == "cross-sectional"
        assert int(f.attrs["n_scans"]) == 8  # 6 annotated + 2 unannotated
        assert int(f.attrs["n_patients"]) == 8
        assert tuple(int(x) for x in f.attrs["spatial_shape"]) == _SHAPE
        assert list(f.attrs["modalities"]) == list(_MODALITIES)
        label_map = json.loads(f.attrs["label_map"])
        assert label_map["1"] == "necrotic_tumor_core"

        has_seg = f["has_segmentation"][:]
        scan_ids = [s.decode() if isinstance(s, bytes) else s for s in f["scan_ids"][:]]
        seg_by_id = dict(zip(scan_ids, has_seg))
        for i in range(1, 7):
            assert seg_by_id[f"BraTS-MEN-{i:05d}-000"]
        assert not seg_by_id["BraTS-MEN-90001-000"]
        assert not seg_by_id["BraTS-MEN-90002-000"]

        # K-fold splits: every k must cover all 6 annotated patients; the
        # test set is shared across k.
        union: set[int] = set()
        for k in (1, 3):
            test = list(f[f"splits/kfold/k{k}/test"][:])
            n_folds = int(f[f"splits/kfold/k{k}"].attrs["n_folds"])
            assert n_folds == k
            sub_union = set(test)
            for i in range(k):
                tr = list(f[f"splits/kfold/k{k}/fold_{i}/train"][:])
                va = list(f[f"splits/kfold/k{k}/fold_{i}/val"][:])
                assert not (set(tr) & set(va))
                sub_union |= set(tr) | set(va)
            assert len(sub_union) == 6
            union = sub_union
        # Test set shared across k.
        t1 = set(f["splits/kfold/k1/test"][:].tolist())
        t3 = set(f["splits/kfold/k3/test"][:].tolist())
        assert t1 == t3
        # Legacy splits not emitted.
        assert "train" not in f["splits"]
        assert "val" not in f["splits"]
        assert "e3_train" not in f["splits"]

        patient_list = [
            p.decode() if isinstance(p, bytes) else p for p in f["longitudinal/patient_list"][:]
        ]
        union_pids = {patient_list[i] for i in union}
        assert union_pids == {f"BraTS-MEN-{i:05d}" for i in range(1, 7)}

        assert bool(f.attrs["has_segmentation_any"]) is True

        subset = f["metadata/subset"][:]
        subset = [s.decode() if isinstance(s, bytes) else s for s in subset]
        subset_by_id = dict(zip(scan_ids, subset))
        assert subset_by_id["BraTS-MEN-90001-000"] == "val"
        assert subset_by_id["BraTS-MEN-00001-000"] == "train"


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
    # The loader rejects the bad seg and stores zeros. Build a converter with a
    # tiny fold layout so the single annotated patient still makes it through
    # build_splits without falling below the n_folds floor — since the bad seg
    # produced has_seg=False, the patient is treated as unannotated and excluded
    # from any e3_* split. Use force_keep_legacy fallback by emitting no e3_*
    # splits when no eligible patients exist.
    converter = BraTSMENConverter(kfold_k_values=(1, 3), kfold_n_strata=2)
    with pytest.raises(ValueError, match="eligible patients"):
        converter.convert({"train": train_root}, output)
