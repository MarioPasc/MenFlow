"""Concrete BraTS Docker segmenter wired to the Phase A `Segmenter` Protocol.

The Phase A Protocol expects ``predict(image: np.ndarray) -> np.ndarray``,
which is single-channel by design. BraTS containers are 4-channel multimodal,
so this class exposes a sibling :meth:`predict_from_subject_dir` that takes a
fully-staged per-subject directory (one file per modality with BraTS-style
naming). The ``predict`` method satisfies the Protocol with a not-implemented
guard — multimodal callers must use the dir-based path.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from menflow.segmentation.docker_runner import BratsDockerRunner
from menflow.segmentation.models import get_model
from menflow.segmentation.output import wt_mask_from_prediction

logger = logging.getLogger(__name__)


class BratsDockerSegmenter:
    """Multimodal BraTS segmenter backed by a Docker container.

    Parameters
    ----------
    model_id
        Key into :data:`menflow.segmentation.models.BRATS_MODELS`
        (e.g. ``"BraTS25_1"``).
    work_dir
        Scratch directory for per-subject docker outputs. Created if absent.
    gpu
        Try to enable GPU passthrough.
    timeout_s
        Per-subject docker-run timeout.
    """

    modality: str = "multi"
    expected_shape: tuple[int, int, int] = (240, 240, 155)

    def __init__(
        self,
        model_id: str,
        work_dir: Path,
        *,
        gpu: bool = True,
        timeout_s: float | None = 1800.0,
    ) -> None:
        self.spec = get_model(model_id)
        self.runner = BratsDockerRunner(self.spec, gpu=gpu)
        self.runner.ensure_image()
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s

    # ------------------------------------------------------------------
    # Protocol shim — single-image path is not applicable for multimodal.
    # ------------------------------------------------------------------

    def predict(self, image: np.ndarray) -> np.ndarray:  # noqa: ARG002
        raise NotImplementedError(
            "BratsDockerSegmenter is multimodal; use predict_from_subject_dir "
            "with a fully-staged per-subject directory instead."
        )

    # ------------------------------------------------------------------
    # Multimodal path
    # ------------------------------------------------------------------

    def predict_from_subject_dir(
        self,
        subject_dir: Path,
        *,
        keep_raw_dir: Path | None = None,
    ) -> tuple[np.ndarray, Path]:
        """Run the container on a 4-modality subject directory.

        BraTS containers expect the *parent* of the subject directory to be
        mounted at ``/input``, so the container sees
        ``/input/<scan_id>/<scan_id>-<modality>.nii.gz``. This wrapper takes
        the subject directory itself and mounts its parent transparently.

        Parameters
        ----------
        subject_dir
            Directory named after the scan id, holding
            ``<scan_id>-{t1c,t1n,t2f,t2w}.nii.gz``. The directory's parent is
            mounted; sibling subject directories under that parent are also
            visible to the container, so callers should keep one subject per
            staging parent.
        keep_raw_dir
            If given, the raw container output dir is copied here for
            inspection.

        Returns
        -------
        (mask, raw_seg_path)
            ``mask`` is a boolean ``np.ndarray`` of shape ``expected_shape``;
            ``raw_seg_path`` is the multi-class NIfTI emitted by the container.
        """
        subject_dir = Path(subject_dir).resolve()
        scan_id = subject_dir.name
        mount_dir = subject_dir.parent
        out_dir = Path(tempfile.mkdtemp(prefix=f"bdseg_{scan_id}_", dir=self.work_dir))
        nifti_paths, elapsed = self.runner.run(mount_dir, out_dir, timeout_s=self.timeout_s)
        if not nifti_paths:
            raise RuntimeError(f"{scan_id}: container exited 0 but wrote no NIfTI to {out_dir}")
        raw_seg_path = nifti_paths[0]
        seg = np.asarray(nib.load(str(raw_seg_path)).dataobj)
        mask = wt_mask_from_prediction(seg, self.spec.label_map_name)
        logger.info(
            "%s segmented %s in %.1f s (raw classes=%s, WT voxels=%d)",
            self.spec.model_id,
            scan_id,
            elapsed,
            sorted({int(v) for v in np.unique(seg)}),
            int(mask.sum()),
        )
        if keep_raw_dir is not None:
            keep_raw_dir = Path(keep_raw_dir)
            keep_raw_dir.mkdir(parents=True, exist_ok=True)
            new_raw_path = keep_raw_dir / raw_seg_path.name
            shutil.copy2(raw_seg_path, new_raw_path)
            raw_seg_path = new_raw_path
        return mask, raw_seg_path
