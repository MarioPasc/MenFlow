"""Tests for the volume-conditional U-Net wrapper.

The MAISI U-Net is heavy (180 M params); these tests use a tiny
`DiffusionModelUNetMaisi` configuration so they remain CPU-friendly. They verify:

1. The wrapper monkey-patches `_get_time_and_class_embedding` to additively
   inject the volume embedding.
2. With the conditioner zero-initialised, the wrapper output equals the
   pretrained-equivalent pass.
3. After perturbing the conditioner, the wrapper output differs.
"""

from __future__ import annotations

import pytest
import torch

monai = pytest.importorskip("monai")

from menflow.finetuning.unet_wrapper import (
    VolumeConditionalUNet,
)
from menflow.finetuning.volume_conditioner import (
    VolumeConditioner,
    VolumeConditionerConfig,
)


def _build_tiny_unet() -> torch.nn.Module:
    from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
        DiffusionModelUNetMaisi,
    )

    # Minimal config: 2 levels, no attention, small channels — fits on CPU.
    # Note: tiny configs (single-resblock or no-attention paths) do NOT route
    # the time/class embedding into the residual blocks, so a smaller config
    # would pass the wrapper trivially while hiding regressions in the
    # injection logic. We use 3 levels with attention on the deepest two and
    # 2 res blocks per level — the smallest config that still consumes
    # `_get_time_and_class_embedding`'s output in the down/middle/up path.
    return DiffusionModelUNetMaisi(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        num_channels=[32, 64, 128],
        attention_levels=[False, True, True],
        num_head_channels=[0, 8, 16],
        num_res_blocks=2,
        norm_num_groups=8,
        use_flash_attention=False,
        include_top_region_index_input=False,
        include_bottom_region_index_input=False,
        include_spacing_input=True,
        num_class_embeds=128,
        resblock_updown=False,
        include_fc=False,
    )


@pytest.fixture
def tiny_setup() -> dict:
    torch.manual_seed(0)
    unet = _build_tiny_unet()
    # MAISI U-Nets ship several zero-initialised parameters (final `out` conv,
    # ResBlock skip projections, etc. — standard practice for diffusion models
    # so the network starts as the identity prediction). On a freshly built
    # model that means many internal pathways emit 0 and mask any conditioning
    # bug. Replace every zero-norm parameter with random noise to make the
    # forward pass actually depend on its inputs.
    with torch.no_grad():
        for p in unet.parameters():
            if p.numel() > 0 and p.norm().item() == 0.0:
                torch.nn.init.normal_(p, std=0.02)
    cond_cfg = VolumeConditionerConfig(
        fourier_features=8,
        fourier_sigma=1.0,
        embed_dim=32 * 4,  # block_out_channels[0] * 4 = 128
        hidden_dim=64,
    )
    conditioner = VolumeConditioner(cond_cfg)
    wrapper = VolumeConditionalUNet(unet, conditioner)

    z = torch.randn(2, 4, 4, 4, 4)
    timesteps = torch.tensor([100.0, 700.0])
    class_labels = torch.tensor([17, 11])  # t1c, t2f
    spacing = torch.tensor([[100.0, 100.0, 100.0], [100.0, 100.0, 100.0]])
    log_v = torch.tensor([-1.0, 1.5])

    return {
        "wrapper": wrapper,
        "z": z,
        "timesteps": timesteps,
        "class_labels": class_labels,
        "spacing": spacing,
        "log_v": log_v,
    }


def test_wrapper_identity_at_init(tiny_setup: dict) -> None:
    """At init: wrapper(log_v) ≡ pretrained-equivalent pass, since conditioner is zero."""
    w = tiny_setup["wrapper"]
    out_cond = w(
        z_t=tiny_setup["z"],
        timesteps=tiny_setup["timesteps"],
        class_labels=tiny_setup["class_labels"],
        spacing_tensor=tiny_setup["spacing"],
        log_v=tiny_setup["log_v"],
    )
    out_uncond = w.forward_unconditional(
        z_t=tiny_setup["z"],
        timesteps=tiny_setup["timesteps"],
        class_labels=tiny_setup["class_labels"],
        spacing_tensor=tiny_setup["spacing"],
    )
    assert torch.allclose(out_cond, out_uncond, atol=1e-6), (
        "wrapper with zero-init conditioner must match unconditional pass"
    )


def test_wrapper_changes_after_conditioner_perturb(tiny_setup: dict) -> None:
    """Once the conditioner has nonzero output, the conditional pass must differ."""
    w = tiny_setup["wrapper"]
    with torch.no_grad():
        torch.nn.init.normal_(w.conditioner.mlp[-1].weight, std=0.5)
        torch.nn.init.normal_(w.conditioner.mlp[-1].bias, std=0.5)
    out_cond = w(
        z_t=tiny_setup["z"],
        timesteps=tiny_setup["timesteps"],
        class_labels=tiny_setup["class_labels"],
        spacing_tensor=tiny_setup["spacing"],
        log_v=tiny_setup["log_v"],
    )
    out_uncond = w.forward_unconditional(
        z_t=tiny_setup["z"],
        timesteps=tiny_setup["timesteps"],
        class_labels=tiny_setup["class_labels"],
        spacing_tensor=tiny_setup["spacing"],
    )
    diff = (out_cond - out_uncond).abs().max().item()
    assert diff > 1e-4, (
        f"after perturbing the conditioner, the conditioned pass must differ (max |Δ| = {diff:.3e})"
    )


def test_wrapper_log_v_modulates_output(tiny_setup: dict) -> None:
    """Different log_v values must yield different outputs once conditioner is non-zero."""
    w = tiny_setup["wrapper"]
    with torch.no_grad():
        torch.nn.init.normal_(w.conditioner.mlp[-1].weight, std=0.5)
    log_v_a = torch.tensor([-3.0, -3.0])
    log_v_b = torch.tensor([3.0, 3.0])
    out_a = w(
        z_t=tiny_setup["z"],
        timesteps=tiny_setup["timesteps"],
        class_labels=tiny_setup["class_labels"],
        spacing_tensor=tiny_setup["spacing"],
        log_v=log_v_a,
    )
    out_b = w(
        z_t=tiny_setup["z"],
        timesteps=tiny_setup["timesteps"],
        class_labels=tiny_setup["class_labels"],
        spacing_tensor=tiny_setup["spacing"],
        log_v=log_v_b,
    )
    assert (out_a - out_b).abs().max().item() > 1e-4, "outputs must change when log_v changes"


def test_wrapper_trainable_state_dict(tiny_setup: dict) -> None:
    w = tiny_setup["wrapper"]
    state = w.trainable_state_dict()
    # Must include conditioner params and at least the c_null vector.
    assert any("conditioner" in k for k in state)
    assert "conditioner.c_null" in state
