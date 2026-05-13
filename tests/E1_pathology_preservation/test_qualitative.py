"""Verify qualitative NPZ dump selects best/worst/median scans correctly."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from experiments.E1_pathology_preservation.analysis.qualitative import (
    _select_cases,
    save_qualitative_cases,
)


def test_select_cases_higher_is_better() -> None:
    s = pd.Series({"a": 30.0, "b": 25.0, "c": 20.0, "d": 27.5, "e": 22.5})
    cases = _select_cases(s, direction="higher")
    assert cases["best"] == "a"
    assert cases["worst"] == "c"
    # Median of [20, 22.5, 25, 27.5, 30] = 25.0 -> closest is "b"
    assert cases["median"] == "b"


def test_select_cases_lower_is_better() -> None:
    s = pd.Series({"a": 30.0, "b": 25.0, "c": 20.0})
    cases = _select_cases(s, direction="lower")
    assert cases["best"] == "c"
    assert cases["worst"] == "a"


def _make_dummy_h5(path: Path, *, n: int, shape: tuple[int, int, int]) -> None:
    H, W, D = shape
    M = 2
    with h5py.File(path, "w") as f:
        f.attrs["n_scans"] = np.int64(n)
        f.attrs["modalities"] = np.asarray(["t1c", "t2w"], dtype=object)
        f.attrs["spatial_shape"] = np.asarray([H, W, D], dtype=np.int64)
        f.attrs["spatial_op"] = "none"
        f.attrs["crop_offset"] = np.asarray([0, 0, 0], dtype=np.int64)
        f.create_dataset("images", data=np.random.rand(n, M, H, W, D).astype(np.float32))
        f.create_dataset("segmentations", data=np.zeros((n, H, W, D), dtype=np.int8))
        vlen = h5py.special_dtype(vlen=str)
        f.create_dataset(
            "scan_ids",
            data=np.asarray([f"scan_{i:03d}" for i in range(n)], dtype=object),
            dtype=vlen,
        )
        f.create_dataset(
            "patient_ids",
            data=np.asarray([f"pat_{i:03d}" for i in range(n)], dtype=object),
            dtype=vlen,
        )


def test_save_qualitative_cases_writes_npz(tmp_path: Path) -> None:
    n = 5
    shape = (8, 8, 8)
    src = tmp_path / "src.h5"
    rec = tmp_path / "rec.h5"
    _make_dummy_h5(src, n=n, shape=shape)
    _make_dummy_h5(rec, n=n, shape=shape)

    rows = []
    for i in range(n):
        for mod in ("t1c", "t2w"):
            rows.append(
                {
                    "scan_id": f"scan_{i:03d}",
                    "patient_id": f"pat_{i:03d}",
                    "modality": mod,
                    "psnr_global": 20.0 + i,
                    "ssim_global": 0.5 + 0.05 * i,
                    "mae_wt": 0.5 - 0.05 * i,
                }
            )
    df = pd.DataFrame(rows)

    written = save_qualitative_cases(
        df=df,
        source_h5=src,
        recon_h5=rec,
        output_dir=tmp_path,
        anchor_metrics=("psnr_global", "mae_wt"),
    )

    assert (tmp_path / "qualitative" / "psnr_global_best.npz").is_file()
    assert (tmp_path / "qualitative" / "psnr_global_worst.npz").is_file()
    assert (tmp_path / "qualitative" / "psnr_global_median.npz").is_file()
    assert (tmp_path / "qualitative" / "mae_wt_best.npz").is_file()

    npz = np.load(written[("psnr_global", "best")])
    assert str(npz["scan_id"]) == "scan_004"
    assert npz["source"].shape == (2, *shape)
    assert npz["recon"].shape == (2, *shape)
    assert npz["segmentation"].shape == shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
