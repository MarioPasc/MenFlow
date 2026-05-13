"""Unit tests for BratsDockerRunner — mocks subprocess.run."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.E2._lib.segmentation.docker_runner import BratsDockerRunner
from experiments.E2._lib.segmentation.models import get_model


def _fake_proc(returncode: int = 0, stdout: str = "", stderr: str = ""):
    class _P:
        pass

    p = _P()
    p.returncode = returncode
    p.stdout = stdout
    p.stderr = stderr
    return p


def test_image_already_present_skips_pull(tmp_path: Path):
    spec = get_model("BraTS25_1")
    runner = BratsDockerRunner(spec, gpu=False)

    with patch("experiments.E2._lib.segmentation.docker_runner.subprocess.run") as mock_run:
        mock_run.return_value = _fake_proc(0)
        runner.ensure_image()
        # Only one inspect call — no pull.
        assert mock_run.call_count == 1
        cmd = mock_run.call_args.args[0]
        assert cmd[:3] == ["docker", "image", "inspect"]


def test_image_missing_triggers_pull(tmp_path: Path):
    spec = get_model("BraTS25_1")
    runner = BratsDockerRunner(spec, gpu=False)

    calls = []

    def _side(cmd, **kwargs):  # noqa: ARG001
        calls.append(cmd)
        if cmd[:3] == ["docker", "image", "inspect"]:
            return _fake_proc(1)  # not present
        return _fake_proc(0)  # pull succeeds

    with patch("experiments.E2._lib.segmentation.docker_runner.subprocess.run", side_effect=_side):
        runner.ensure_image()

    assert any(c[:2] == ["docker", "pull"] for c in calls)


def test_run_docker_only_command_construction(tmp_path: Path):
    spec = get_model("BraTS25_1")
    runner = BratsDockerRunner(spec, gpu=False)

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    # Pre-create a fake output NIfTI so the result list is non-empty.
    output_dir.mkdir()
    (output_dir / "fake.nii.gz").write_bytes(b"")

    with patch("experiments.E2._lib.segmentation.docker_runner.subprocess.run") as mock_run:
        mock_run.return_value = _fake_proc(0)
        paths, elapsed = runner.run(input_dir, output_dir)

    cmd = mock_run.call_args.args[0]
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "--gpus" not in cmd  # gpu=False
    assert "-v" in cmd
    assert any(":/input:ro" in arg for arg in cmd)
    assert any(":/output:rw" in arg for arg in cmd)
    assert spec.docker_image in cmd
    assert paths == [output_dir / "fake.nii.gz"]
    assert elapsed >= 0.0


def test_run_mlcube_command_construction(tmp_path: Path):
    spec = get_model("BraTS23_2")
    runner = BratsDockerRunner(spec, gpu=False)

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    (output_dir / "pred.nii.gz").write_bytes(b"")

    with patch("experiments.E2._lib.segmentation.docker_runner.subprocess.run") as mock_run:
        mock_run.return_value = _fake_proc(0)
        runner.run(input_dir, output_dir)

    cmd = mock_run.call_args.args[0]
    assert "infer" in cmd
    assert any("--data_path=" in arg for arg in cmd)
    assert any("--output_path=" in arg for arg in cmd)
    assert "--user" in cmd  # requires_root
    assert "--shm-size=4g" in cmd


def test_run_failure_raises(tmp_path: Path):
    spec = get_model("BraTS25_1")
    runner = BratsDockerRunner(spec, gpu=False)
    (tmp_path / "in").mkdir()
    (tmp_path / "out").mkdir()
    with patch("experiments.E2._lib.segmentation.docker_runner.subprocess.run") as mock_run:
        mock_run.return_value = _fake_proc(1, stderr="boom")
        with pytest.raises(RuntimeError, match="failed"):
            runner.run(tmp_path / "in", tmp_path / "out")
