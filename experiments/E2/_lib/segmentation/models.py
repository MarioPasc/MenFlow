"""Registry of BraTS-MEN challenge segmenter Docker images.

Mirrors the registry used in
``MenGrowth-Model/experiments/uncertainty_segmentation/benchmark/config.yaml``
so behaviour is identical when the same containers are invoked from MenFlow.

Each :class:`BratsModelSpec` carries every piece of information the runner
needs (image name, interface flavour, label map, root requirement) so callers
never need to special-case a model id.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BratsModelSpec:
    """Container metadata for one BraTS challenge segmenter.

    Attributes
    ----------
    model_id
        Stable identifier (e.g. ``"BraTS25_1"``).
    docker_image
        Fully-qualified image name; used by ``docker pull`` and ``docker run``.
    year
        Challenge year (``2023`` or ``2025``).
    interface
        ``"docker_only"`` (BraTS25 family) — bare ``docker run`` with
        ``-v {input}:/input -v {output}:/output``.
        ``"mlcube"`` (BraTS23 family) — ``docker run ... infer
        --data_path=/mlcube_io0 --output_path=/mlcube_io2``.
    label_map_name
        Key into :data:`experiments.E2._lib.segmentation.output.LABEL_MAPS`.
    requires_root
        If True, pass ``--user 0`` to ``docker run`` (BraTS23 mlcube containers
        write under directories owned by root).
    shm_size
        ``--shm-size`` argument; some containers OOM otherwise.
    cpu_compatible
        If True, the container is known to run without GPU.
    """

    model_id: str
    docker_image: str
    year: int
    interface: str
    label_map_name: str
    requires_root: bool = False
    shm_size: str | None = None
    cpu_compatible: bool = False


BRATS_MODELS: dict[str, BratsModelSpec] = {
    # NOTE on label_map_name for the BraTS25 family. The MenGrowth-Model
    # benchmark config documented BraTS25 outputs as {1: SNFH, 2: ET}
    # (2-class, NETC dropped). Empirically this container emits the same
    # 3-class layout as BraTS-MEN ground truth ({1: NETC, 2: SNFH, 3: ET}),
    # with the tumor bulk at label 3. We use ``"brats23"`` here so the
    # whole-tumor reduction picks up label 3 — verified on real
    # BraTS-MEN-00717-009 (4294 of 4314 WT voxels at label 3).
    "BraTS25_1": BratsModelSpec(
        model_id="BraTS25_1",
        docker_image="brainles/brats25_men_qing:latest",
        year=2025,
        interface="docker_only",
        label_map_name="brats23",
        shm_size="8g",
    ),
    "BraTS25_2": BratsModelSpec(
        model_id="BraTS25_2",
        docker_image="brainles/brats25_men_mmdp:latest",
        year=2025,
        interface="docker_only",
        label_map_name="brats23",
        shm_size="8g",
    ),
    "BraTS23_1": BratsModelSpec(
        model_id="BraTS23_1",
        docker_image="brainles/brats23_meningioma_nvauto:latest",
        year=2023,
        interface="mlcube",
        label_map_name="brats23",
        requires_root=True,
        shm_size="32g",
    ),
    "BraTS23_2": BratsModelSpec(
        model_id="BraTS23_2",
        docker_image="brainles/brats23_meningioma_blackbean:latest",
        year=2023,
        interface="mlcube",
        label_map_name="brats23",
        requires_root=True,
        shm_size="4g",
        cpu_compatible=True,
    ),
    "BraTS23_3": BratsModelSpec(
        model_id="BraTS23_3",
        docker_image="brainles/brats23_meningioma_cnmc_pmi2023:latest",
        year=2023,
        interface="mlcube",
        label_map_name="brats23",
        requires_root=True,
        shm_size="2g",
    ),
}


def get_model(model_id: str) -> BratsModelSpec:
    """Return the :class:`BratsModelSpec` for ``model_id`` or raise."""
    if model_id not in BRATS_MODELS:
        raise KeyError(f"unknown BraTS model id {model_id!r}; choose from {sorted(BRATS_MODELS)}")
    return BRATS_MODELS[model_id]
