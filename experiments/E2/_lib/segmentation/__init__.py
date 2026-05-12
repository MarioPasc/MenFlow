"""Docker-backed BraTS-MEN segmentation primitives.

The library has three layers:

* :mod:`experiments.E2._lib.segmentation.models` — registry of BraTS challenge winner
  containers (BraTS25_1, BraTS25_2, BraTS23_1/2/3) with their interface
  metadata.
* :mod:`experiments.E2._lib.segmentation.docker_runner` — generic runner that mounts a
  per-subject NIfTI directory and collects predicted segmentations.
* :mod:`experiments.E2._lib.segmentation.output` — convert raw multi-class container
  outputs into a binary whole-tumor mask.
* :mod:`experiments.E2._lib.segmentation.companion_decode` — re-uses the MAISI decoder to
  produce companion-modality NIfTIs for a single anchor at Δ=0.

The library is dataset-agnostic; the BraTS-MEN file naming convention
(``<scan_id>-<modality>.nii.gz``) is the only assumption baked in.
"""

from experiments.E2._lib.segmentation.companion_decode import decode_companion_modalities
from experiments.E2._lib.segmentation.docker_runner import BratsDockerRunner
from experiments.E2._lib.segmentation.models import BRATS_MODELS, BratsModelSpec, get_model
from experiments.E2._lib.segmentation.output import LABEL_MAPS, wt_mask_from_prediction

__all__ = [
    "BRATS_MODELS",
    "BratsDockerRunner",
    "BratsModelSpec",
    "LABEL_MAPS",
    "decode_companion_modalities",
    "get_model",
    "wt_mask_from_prediction",
]
