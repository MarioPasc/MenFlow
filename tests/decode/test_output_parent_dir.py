"""Regression tests for the decode-output parent-dir bug.

The Picasso run failed with::

    FileNotFoundError: [Errno 2] Unable to synchronously create file
    (unable to open file: name = '/.../datasets/menflow/recon/brats_men_maisi_recon.h5')

because ``h5py.File(path, "w")`` does not create intermediate directories.
The fix is :func:`routines.decode.engine.decode_engine._prepare_output_path`,
which is now invoked at the top of :meth:`DecodeEngine.run`. These tests
replicate the failure locally and confirm the helper resolves it.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from routines.decode.engine.decode_engine import _prepare_output_path


def test_h5py_create_fails_when_parent_missing(tmp_path: Path) -> None:
    """Reproduces the original Picasso traceback locally."""
    target = tmp_path / "missing" / "subdir" / "out.h5"
    with pytest.raises(FileNotFoundError):
        h5py.File(target, "w").close()


def test_prepare_output_path_creates_intermediate_dirs(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "out.h5"
    _prepare_output_path(target)
    assert target.parent.is_dir()


def test_h5py_create_succeeds_after_prepare(tmp_path: Path) -> None:
    """End-to-end: applying the fix lets ``h5py.File(..., 'w')`` succeed."""
    target = tmp_path / "missing" / "subdir" / "out.h5"
    _prepare_output_path(target)
    with h5py.File(target, "w") as f:
        f.attrs["smoke"] = 1
    assert target.is_file()


def test_prepare_output_path_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "already_there" / "out.h5"
    target.parent.mkdir(parents=True)
    _prepare_output_path(target)  # must not raise
    assert target.parent.is_dir()
