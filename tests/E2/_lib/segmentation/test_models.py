"""Unit tests for the BRATS_MODELS registry."""

from __future__ import annotations

import pytest

from experiments.E2._lib.segmentation.models import BRATS_MODELS, BratsModelSpec, get_model


def test_registry_has_expected_models():
    expected = {"BraTS25_1", "BraTS25_2", "BraTS23_1", "BraTS23_2", "BraTS23_3"}
    assert expected.issubset(BRATS_MODELS.keys())


def test_specs_have_consistent_label_maps():
    for model_id, spec in BRATS_MODELS.items():
        assert spec.label_map_name in {"brats25", "brats23"}, (
            f"{model_id} has unknown label map {spec.label_map_name}"
        )
        assert spec.interface in {"docker_only", "mlcube"}, (
            f"{model_id} has unknown interface {spec.interface}"
        )


def test_brats25_models_are_docker_only():
    # NOTE on label_map_name: although the BrainLesion docs flag BraTS25
    # outputs as 2-class {1,2}, the BraTS25_1 container empirically emits
    # 3-class {1,2,3} (verified on BraTS-MEN-00717-009: 4 294 of 4 314 WT
    # voxels are at label 3). The registry uses ``"brats23"`` so the WT
    # reduction picks up label 3.
    for mid in ("BraTS25_1", "BraTS25_2"):
        spec = BRATS_MODELS[mid]
        assert spec.interface == "docker_only"
        assert spec.label_map_name == "brats23"


def test_brats23_models_are_mlcube_and_brats23_label():
    for mid in ("BraTS23_1", "BraTS23_2", "BraTS23_3"):
        spec = BRATS_MODELS[mid]
        assert spec.interface == "mlcube"
        assert spec.label_map_name == "brats23"
        assert spec.requires_root is True


def test_get_model_unknown_raises():
    with pytest.raises(KeyError, match="unknown BraTS model id"):
        get_model("NOT_A_MODEL")


def test_specs_are_frozen():
    spec = BRATS_MODELS["BraTS25_1"]
    with pytest.raises((AttributeError, TypeError)):
        spec.year = 1999  # type: ignore[misc]
    assert isinstance(spec, BratsModelSpec)
