"""Reusable building blocks for parameter-efficient FM finetuning.

Originally lived under ``routines/finetune_fm_volume/engine``; promoted to
``src/menflow/finetuning`` because the LoRA wrapper, volume conditioner,
U-Net injection wrapper, and calibration metrics are useful to any future
finetuning routine (e.g. multi-conditional, longitudinal, or alternative
backbones), not just the volume-conditional MAISI-v2 routine that introduced
them.
"""

from menflow.finetuning.calibration import (
    CalibrationMetrics,
    fit_calibration,
    per_logv_bin_metrics,
)
from menflow.finetuning.lora import (
    LoRAConfig,
    apply_lora,
    collect_lora_state,
    load_lora_state,
    lora_param_iter,
)
from menflow.finetuning.sampling import (
    SamplerConfig,
    sample_latents,
)
from menflow.finetuning.unet_wrapper import (
    MAISI_MODALITY_INDEX,
    MAISI_MR_FM_ARCH,
    VolumeConditionalUNet,
    load_maisi_fm_unet,
    serialize_arch,
)
from menflow.finetuning.volume_conditioner import (
    FourierFeatures,
    VolumeConditioner,
    VolumeConditionerConfig,
)

__all__ = [
    "CalibrationMetrics",
    "FourierFeatures",
    "LoRAConfig",
    "SamplerConfig",
    "MAISI_MODALITY_INDEX",
    "MAISI_MR_FM_ARCH",
    "VolumeConditionalUNet",
    "VolumeConditioner",
    "VolumeConditionerConfig",
    "apply_lora",
    "collect_lora_state",
    "fit_calibration",
    "load_lora_state",
    "load_maisi_fm_unet",
    "lora_param_iter",
    "per_logv_bin_metrics",
    "sample_latents",
    "serialize_arch",
]
