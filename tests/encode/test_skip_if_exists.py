"""Tests for the encode short-circuit and parent-dir helper.

A re-launch of the E1 pipeline must not redo an encode pass that has already
produced a valid latent H5. ``EncodeEngine.run`` short-circuits via
:func:`routines.encode.engine.encode_engine._latent_already_encoded`.
"""

from __future__ import annotations

from pathlib import Path

import h5py

from routines.encode.engine.encode_engine import (
    _latent_already_encoded,
    _prepare_output_path,
)


def test_latent_already_encoded_false_for_missing_file(tmp_path: Path) -> None:
    assert _latent_already_encoded(tmp_path / "no_such_file.h5") is False


def test_latent_already_encoded_false_for_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.h5"
    p.write_bytes(b"")
    assert _latent_already_encoded(p) is False


def test_latent_already_encoded_false_for_corrupt_file(tmp_path: Path) -> None:
    p = tmp_path / "corrupt.h5"
    p.write_bytes(b"not an HDF5 file at all")
    assert _latent_already_encoded(p) is False


def test_latent_already_encoded_true_for_valid_h5(tmp_path: Path) -> None:
    p = tmp_path / "ok.h5"
    with h5py.File(p, "w") as f:
        f.attrs["dataset_name"] = "smoke"
        f.create_dataset("dummy", data=[1, 2, 3])
    assert _latent_already_encoded(p) is True


def test_prepare_output_path_creates_intermediate_dirs(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "out.h5"
    _prepare_output_path(target)
    assert target.parent.is_dir()
