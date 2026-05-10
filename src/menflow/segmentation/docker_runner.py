"""Run BraTS challenge Docker containers on a per-subject NIfTI directory.

The two interface flavours used by BrainLesion containers are abstracted
behind a single :class:`BratsDockerRunner.run` call:

* ``docker_only`` (BraTS25): mount input at ``/input``, output at ``/output``,
  no extra arguments.
* ``mlcube`` (BraTS23): mount input at ``/mlcube_io0``, output at
  ``/mlcube_io2``, optional params at ``/mlcube_io1``; the entrypoint takes
  ``infer --data_path=/mlcube_io0 --output_path=/mlcube_io2``.

GPU passthrough is automatic — the runner probes for
``nvidia-container-toolkit`` once and reuses the result.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from menflow.segmentation.models import BratsModelSpec

logger = logging.getLogger(__name__)


class BratsDockerRunner:
    """Pull-then-run wrapper around a BraTS Docker image.

    Parameters
    ----------
    model
        The :class:`~menflow.segmentation.models.BratsModelSpec` to run.
    gpu
        If ``True`` (default), pass ``--gpus all`` to ``docker run`` provided
        the host has ``nvidia-container-toolkit`` working. Set to ``False`` to
        force CPU mode (only sensible for ``cpu_compatible`` models).
    docker_bin
        ``docker`` binary to invoke. Override only for testing.
    """

    def __init__(
        self,
        model: BratsModelSpec,
        *,
        gpu: bool = True,
        docker_bin: str = "docker",
    ) -> None:
        self.model = model
        self.docker_bin = docker_bin
        self._gpu_requested = gpu
        self._gpu_available: bool | None = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def ensure_image(self) -> None:
        """Pull ``model.docker_image`` if not already present locally."""
        if self._image_present():
            logger.info("Docker image already present: %s", self.model.docker_image)
            return
        logger.info("Pulling Docker image: %s", self.model.docker_image)
        cmd = [self.docker_bin, "pull", self.model.docker_image]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"docker pull failed for {self.model.docker_image}: "
                f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
            )

    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        *,
        timeout_s: float | None = None,
    ) -> tuple[list[Path], float]:
        """Run the container; return predicted NIfTI paths and wall-clock seconds.

        Parameters
        ----------
        input_dir
            Per-subject directory holding the BraTS-named modality NIfTIs.
        output_dir
            Directory the container will write predictions into. Created if
            absent.
        timeout_s
            Wall-clock cap; ``None`` for no timeout.

        Returns
        -------
        list[pathlib.Path]
            All ``*.nii.gz`` files written under ``output_dir`` (sorted).
        float
            Wall-clock seconds spent inside ``docker run``.
        """
        input_dir = Path(input_dir).resolve()
        output_dir = Path(output_dir).resolve()
        if not input_dir.is_dir():
            raise FileNotFoundError(input_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = self._build_run_cmd(input_dir, output_dir)
        logger.info("Running %s on %s -> %s", self.model.model_id, input_dir, output_dir)
        logger.debug("docker cmd: %s", " ".join(cmd))
        t0 = time.monotonic()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        elapsed = time.monotonic() - t0
        if proc.returncode != 0:
            tail = (proc.stdout or "") + "\n" + (proc.stderr or "")
            raise RuntimeError(
                f"{self.model.model_id} failed (exit {proc.returncode}); "
                f"output tail:\n{tail[-2000:]}"
            )
        nifti_paths = sorted(output_dir.rglob("*.nii.gz"))
        logger.info(
            "%s wrote %d nii.gz files in %.1f s",
            self.model.model_id,
            len(nifti_paths),
            elapsed,
        )
        return nifti_paths, elapsed

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _image_present(self) -> bool:
        cmd = [self.docker_bin, "image", "inspect", self.model.docker_image]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.returncode == 0

    def _gpu_passthrough_works(self) -> bool:
        if not self._gpu_requested:
            return False
        if self._gpu_available is None:
            cmd = [
                self.docker_bin,
                "run",
                "--rm",
                "--gpus",
                "all",
                "nvidia/cuda:12.0.0-base-ubuntu22.04",
                "true",
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                self._gpu_available = proc.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self._gpu_available = False
            if not self._gpu_available:
                logger.warning(
                    "GPU passthrough probe failed; falling back to CPU mode "
                    "(model.cpu_compatible=%s)",
                    self.model.cpu_compatible,
                )
        return bool(self._gpu_available)

    def _build_run_cmd(self, input_dir: Path, output_dir: Path) -> list[str]:
        gpu_args: list[str] = []
        if self._gpu_passthrough_works():
            gpu_args = ["--gpus", "all"]
        elif self._gpu_requested and not self.model.cpu_compatible:
            raise RuntimeError(
                f"{self.model.model_id} requires GPU but nvidia-container-toolkit "
                f"is unavailable (and cpu_compatible=False)"
            )
        shm_args: list[str] = []
        if self.model.shm_size:
            shm_args = [f"--shm-size={self.model.shm_size}"]
        user_args: list[str] = []
        if self.model.requires_root:
            user_args = ["--user", "0:0"]

        if self.model.interface == "docker_only":
            return [
                self.docker_bin,
                "run",
                "--rm",
                *gpu_args,
                *shm_args,
                *user_args,
                "-v",
                f"{input_dir}:/input:ro",
                "-v",
                f"{output_dir}:/output:rw",
                self.model.docker_image,
            ]
        if self.model.interface == "mlcube":
            params_dir = self._mlcube_params_dir()
            return [
                self.docker_bin,
                "run",
                "--rm",
                *gpu_args,
                *shm_args,
                *user_args,
                "-v",
                f"{input_dir}:/mlcube_io0:ro",
                "-v",
                f"{params_dir}:/mlcube_io1:ro",
                "-v",
                f"{output_dir}:/mlcube_io2:rw",
                self.model.docker_image,
                "infer",
                "--data_path=/mlcube_io0",
                "--output_path=/mlcube_io2",
                "--parameters_file=/mlcube_io1/params.yaml",
            ]
        raise ValueError(f"unknown interface {self.model.interface!r}")

    def _mlcube_params_dir(self) -> Path:
        """Create a tmp params dir with an empty ``params.yaml``.

        The BraTS23 mlcube containers refuse to start without
        ``--parameters_file`` even when the file is empty.
        """
        d = Path(tempfile.gettempdir()) / "menflow_mlcube_params"
        d.mkdir(parents=True, exist_ok=True)
        params = d / "params.yaml"
        if not params.is_file():
            params.write_text("{}\n")
        return d


def cleanup_image(model: BratsModelSpec, *, docker_bin: str = "docker") -> None:
    """Best-effort ``docker rmi`` for the given model image."""
    if shutil.which(docker_bin) is None:  # pragma: no cover
        return
    subprocess.run(
        [docker_bin, "rmi", model.docker_image],
        capture_output=True,
        text=True,
    )
