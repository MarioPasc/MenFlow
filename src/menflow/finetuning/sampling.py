"""Volume-conditional ODE sampling for the MAISI-v2 rectified-flow U-Net.

Mirrors the reference inference loop at
``NV-Generate-CTMR/scripts/diff_model_infer.py:153-212`` but talks through the
:class:`menflow.finetuning.VolumeConditionalUNet` so the volume embedding
and CFG come for free. The function is intentionally backbone-agnostic —
any wrapper that exposes the same forward signature can be sampled with it.

Used by:

* the engine's periodic ``sample_interval`` preview (decoded T1c at a small
  anchor × log_v grid, persisted next to the run);
* the final-test calibration pass (latent-only sampling for R²/Spearman with
  no decode);
* downstream analysis scripts on Picasso (rank sweep, full Stage-1 gate).

References
----------
Liu et al., "Flow Straight and Fast", arXiv:2209.03003.
Ho & Salimans, "Classifier-Free Diffusion Guidance", arXiv:2207.12598.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SamplerConfig:
    """Knobs for :func:`sample_latents`."""

    n_ode_steps: int = 30
    cfg_scale: float = 4.0
    autocast_dtype: torch.dtype = torch.float32
    deterministic: bool = True  # if True, reseeds noise per call


@torch.no_grad()
def sample_latents(
    wrapper: torch.nn.Module,
    *,
    log_v: torch.Tensor,
    class_labels: torch.Tensor,
    spacing_tensor: torch.Tensor,
    latent_shape: tuple[int, int, int, int],  # (C, H', W', D')
    scheduler,
    cfg: SamplerConfig = SamplerConfig(),
    device: torch.device | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Run the rectified-flow ODE with classifier-free guidance.

    Returns a tensor of shape ``(B, *latent_shape)`` where ``B = log_v.numel()``.
    Initial noise is drawn from ``N(0, I)``; the ODE integrates from
    ``t=num_train_timesteps`` (pure noise) to ``t≈1`` (pure signal).
    """
    device = device or next(wrapper.parameters()).device
    B = int(log_v.numel())
    if class_labels.numel() == 1:
        class_labels = class_labels.expand(B)
    if spacing_tensor.dim() == 1:
        spacing_tensor = spacing_tensor.unsqueeze(0).expand(B, -1)

    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
        z = torch.randn(B, *latent_shape, device=device, generator=gen)
    else:
        z = torch.randn(B, *latent_shape, device=device)

    n_train = int(scheduler.num_train_timesteps)
    steps = int(cfg.n_ode_steps)
    # Linearly spaced timesteps from near-noise (n_train) down to near-signal (1).
    ts = torch.linspace(n_train, 1, steps + 1, device=device)
    s_minus = ts[:-1]
    s_next = ts[1:]

    use_uncond_true = torch.ones(B, dtype=torch.bool, device=device)
    use_uncond_false = torch.zeros(B, dtype=torch.bool, device=device)

    autocast = torch.amp.autocast(
        device_type=device.type,
        dtype=cfg.autocast_dtype,
        enabled=cfg.autocast_dtype is not torch.float32,
    )
    for t_now, t_next in zip(s_minus, s_next):
        t_batch = t_now.expand(B)
        with autocast:
            pred_cond = wrapper(
                z_t=z,
                timesteps=t_batch,
                class_labels=class_labels,
                spacing_tensor=spacing_tensor,
                log_v=log_v,
                use_uncond=use_uncond_false,
            ).float()
            if cfg.cfg_scale != 1.0:
                pred_uncond = wrapper(
                    z_t=z,
                    timesteps=t_batch,
                    class_labels=class_labels,
                    spacing_tensor=spacing_tensor,
                    log_v=log_v,
                    use_uncond=use_uncond_true,
                ).float()
                pred = pred_uncond + cfg.cfg_scale * (pred_cond - pred_uncond)
            else:
                pred = pred_cond
        # RFlowScheduler.step returns the next-state latent.
        z = scheduler.step(pred, t_now, z, t_next).prev_sample
    return z
