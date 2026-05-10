"""Volume-conditional wrapper around MONAI's `DiffusionModelUNetMaisi`.

Injection strategy
------------------
MONAI's `DiffusionModelUNetMaisi` builds the per-resblock conditioning embedding
in two steps inside `forward`:

1. ``_get_time_and_class_embedding(x, timesteps, class_labels)`` returns the
   time-embedding plus (optionally) the class-conditioning addend, shape
   ``[B, time_embed_dim]`` (= 256 for the MR config).
2. ``_get_input_embeddings`` then concatenates the spacing branch onto this,
   producing ``[B, new_time_embed_dim]`` (= 512), which is fed to every resblock.

The volume embedding is added to the output of step 1, *before* spacing concat.
This keeps `new_time_embed_dim` unchanged so no resblock-projection surgery is
required. The hook is installed by replacing
``unet._get_time_and_class_embedding`` with a closure that captures the wrapper
and reads a per-forward ``self._volume_emb`` slot.

This is single-process / single-thread by design; if you ever fan out to threads
sharing the same module, you must serialize forwards.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from menflow.finetuning.volume_conditioner import (
    VolumeConditioner,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# MAISI MR rectified-flow architecture (mirrors config_network_rflow.json's
# `diffusion_unet_def`). Embedded here so we don't hard-depend on the neuromf
# repo path. Validated: the resulting model loads `unet_state_dict` strictly.
# ----------------------------------------------------------------------------

MAISI_MR_FM_ARCH: dict[str, Any] = {
    "spatial_dims": 3,
    "in_channels": 4,
    "out_channels": 4,
    "num_channels": [64, 128, 256, 512],
    "attention_levels": [False, False, True, True],
    "num_head_channels": [0, 0, 32, 32],
    "num_res_blocks": 2,
    "use_flash_attention": True,
    "include_top_region_index_input": False,
    "include_bottom_region_index_input": False,
    "include_spacing_input": True,
    "num_class_embeds": 128,
    "resblock_updown": True,
    "include_fc": True,
}

# MAISI modality registry (subset relevant to BraTS-MEN). Values are the
# `class_labels` integers consumed by `nn.Embedding(128, 256)` in the U-Net.
MAISI_MODALITY_INDEX: dict[str, int] = {
    "t1c": 17,  # mri_t1c
    "t1n": 9,  # mri_t1 (T1 native / pre-contrast)
    "t2f": 11,  # mri_flair
    "t2w": 10,  # mri_t2
}


def load_maisi_fm_unet(
    checkpoint_path: str | Path,
    *,
    arch: dict[str, Any] | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[nn.Module, float, dict[str, Any]]:
    """Instantiate `DiffusionModelUNetMaisi` and load the FM checkpoint.

    Returns
    -------
    model : nn.Module
        The U-Net with weights loaded (still on `map_location`).
    scale_factor : float
        The latent rescaling factor stored in the checkpoint (= 1/std(z)).
    extra : dict
        `{"epoch": int, "loss": float, "num_train_timesteps": int}` from the
        checkpoint metadata (best-effort).
    """
    from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
        DiffusionModelUNetMaisi,
    )

    arch = dict(arch or MAISI_MR_FM_ARCH)
    model = DiffusionModelUNetMaisi(**arch)
    state = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if "unet_state_dict" not in state:
        raise KeyError(
            f"checkpoint at {checkpoint_path} has keys {list(state.keys())[:6]}…; "
            "expected 'unet_state_dict'"
        )
    missing, unexpected = model.load_state_dict(state["unet_state_dict"], strict=strict)
    if missing:
        logger.warning("missing keys: %d (e.g. %s)", len(missing), missing[:3])
    if unexpected:
        logger.warning("unexpected keys: %d (e.g. %s)", len(unexpected), unexpected[:3])
    scale_factor = float(state.get("scale_factor", torch.tensor(1.0)).cpu().item())
    extra = {
        "epoch": int(state.get("epoch", -1)),
        "loss": float(state.get("loss", float("nan"))),
        "num_train_timesteps": int(state.get("num_train_timesteps", 1000)),
    }
    return model, scale_factor, extra


class VolumeConditionalUNet(nn.Module):
    """Wraps a `DiffusionModelUNetMaisi` with a `VolumeConditioner` injection.

    Forward signature
    -----------------
    ``forward(z_t, timesteps, class_labels, spacing_tensor, log_v, use_uncond=None)``

    The wrapper installs a closure on `unet._get_time_and_class_embedding` that
    additively injects the volume embedding into the time+class addend.

    Parameters
    ----------
    unet : nn.Module
        A `DiffusionModelUNetMaisi` instance with weights already loaded.
    conditioner : VolumeConditioner
        The trainable embedding for log-V.
    """

    def __init__(self, unet: nn.Module, conditioner: VolumeConditioner) -> None:
        super().__init__()
        self.unet = unet
        self.conditioner = conditioner
        self._volume_emb: torch.Tensor | None = None
        self._patch_unet()

    def _patch_unet(self) -> None:
        original = self.unet._get_time_and_class_embedding  # bound method
        wrapper = self

        def patched(
            x: torch.Tensor,
            timesteps: torch.Tensor,
            class_labels: torch.Tensor | None,
        ) -> torch.Tensor:
            emb = original(x, timesteps, class_labels)
            if wrapper._volume_emb is not None:
                emb = emb + wrapper._volume_emb.to(dtype=emb.dtype)
            return emb

        # Replace the bound attribute on the instance only.
        self.unet._get_time_and_class_embedding = patched  # type: ignore[assignment]

    def forward(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: torch.Tensor,
        spacing_tensor: torch.Tensor,
        log_v: torch.Tensor,
        use_uncond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.conditioner(log_v, use_uncond=use_uncond)  # [B, 256]
        self._volume_emb = emb
        try:
            return self.unet(
                x=z_t,
                timesteps=timesteps,
                class_labels=class_labels,
                spacing_tensor=spacing_tensor,
            )
        finally:
            self._volume_emb = None

    def forward_unconditional(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: torch.Tensor,
        spacing_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Pass through with the volume injection disabled (pretrained-equivalent)."""
        self._volume_emb = None
        return self.unet(
            x=z_t,
            timesteps=timesteps,
            class_labels=class_labels,
            spacing_tensor=spacing_tensor,
        )

    def trainable_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Group trainable parameters by source for separate optimizer logging."""
        lora_params = [
            p for n, p in self.unet.named_parameters() if p.requires_grad and "lora_" in n
        ]
        cond_params = [p for p in self.conditioner.parameters() if p.requires_grad]
        return {"lora": lora_params, "conditioner": cond_params}

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only the LoRA (A/B) + VolumeConditioner weights.

        peft's injected adapter parameters carry `.lora_A.<adapter>.weight` /
        `.lora_B.<adapter>.weight` in their names; that substring match is
        enough to filter them. The volume conditioner's params are namespaced
        under `conditioner.` so its `c_null` parameter is included via the
        ordinary `named_parameters()` walk.
        """
        out: dict[str, torch.Tensor] = {}
        for n, p in self.unet.state_dict().items():
            if "lora_" in n:
                out[f"unet.{n}"] = p.detach().cpu()
        for n, p in self.conditioner.state_dict().items():
            out[f"conditioner.{n}"] = p.detach().cpu()
        return out

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        unet_state = {k[len("unet.") :]: v for k, v in state.items() if k.startswith("unet.")}
        cond_state = {
            k[len("conditioner.") :]: v for k, v in state.items() if k.startswith("conditioner.")
        }
        if unet_state:
            self.unet.load_state_dict(unet_state, strict=False)
        if cond_state:
            self.conditioner.load_state_dict(cond_state, strict=False)


def serialize_arch(arch: dict[str, Any]) -> str:
    """Stable JSON serialization for run provenance."""
    return json.dumps(arch, sort_keys=True, indent=2)
