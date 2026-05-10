"""Tests for VolumeConditioner: identity-at-init and CFG null-token mixing."""

from __future__ import annotations

import torch

from menflow.finetuning.volume_conditioner import (
    FourierFeatures,
    VolumeConditioner,
    VolumeConditionerConfig,
)


def test_fourier_features_shape() -> None:
    ff = FourierFeatures(num_features=32, sigma=1.0)
    out = ff(torch.tensor([0.0, 1.0, 2.0]))
    assert out.shape == (3, 64)


def test_fourier_features_frozen() -> None:
    ff = FourierFeatures(num_features=8, sigma=1.0)
    f1 = ff.freqs.clone()
    out = ff(torch.zeros(4))
    assert torch.equal(ff.freqs, f1), "frequencies must be frozen across forwards"
    assert out.dtype == torch.float32


def test_volume_conditioner_zero_at_init() -> None:
    cfg = VolumeConditionerConfig(
        fourier_features=32, fourier_sigma=1.0, embed_dim=256, hidden_dim=256
    )
    cond = VolumeConditioner(cfg)
    log_v = torch.tensor([-2.0, 0.0, 2.0, 5.0])
    out = cond(log_v)
    assert out.shape == (4, 256)
    assert out.abs().max().item() == 0.0, "conditioner must output exactly zero at init"


def test_volume_conditioner_uncond_uses_null() -> None:
    cfg = VolumeConditionerConfig()
    cond = VolumeConditioner(cfg)
    # Manually set the conditioner output to be non-zero by perturbing the last linear.
    with torch.no_grad():
        cond.mlp[-1].weight.fill_(0.1)
        cond.mlp[-1].bias.fill_(0.05)
        cond.c_null.fill_(7.0)
    log_v = torch.tensor([1.0, 1.0, 1.0])
    use_uncond = torch.tensor([False, True, False])
    out = cond(log_v, use_uncond=use_uncond)
    assert torch.allclose(out[1], cond.c_null), "uncond entry must equal null token"
    assert not torch.allclose(out[0], cond.c_null), "cond entry must not equal null token"


def test_volume_conditioner_grad_flows_to_last_linear() -> None:
    cfg = VolumeConditionerConfig()
    cond = VolumeConditioner(cfg)
    # Perturb away from zero init to get nonzero gradient signal.
    with torch.no_grad():
        cond.mlp[-1].weight.add_(torch.randn_like(cond.mlp[-1].weight) * 0.01)
    log_v = torch.tensor([0.5, -1.0])
    target = torch.zeros(2, cfg.embed_dim)
    out = cond(log_v)
    loss = (out - target).pow(2).mean()
    loss.backward()
    assert cond.mlp[-1].weight.grad is not None
    assert cond.mlp[-1].weight.grad.abs().sum().item() > 0.0
