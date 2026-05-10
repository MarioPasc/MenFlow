"""Volume conditioner: scalar log-V → 256-d embedding additive on the U-Net's
time+class embedding.

The conditioner is a Fourier feature embedding (Karras et al., EDM, arXiv:2206.00364
§C.2) followed by a 2-layer MLP. Two design constraints:

1. **Identity at init.** The MLP's last linear weights are zero-initialised so the
   conditioner output is exactly zero before any optimizer step. Combined with the
   LoRA `B=0` init, the patched U-Net is bit-identical to the pretrained model at
   step 0. This is asserted by a unit test.
2. **Learned null token (CFG).** With probability `p_uncond`, the volume embedding
   is replaced by a learned vector `c_null` (a `nn.Parameter` initialised to zero).
   At inference, classifier-free guidance interpolates `v_θ(c_∅)` and `v_θ(c_v)` at
   guidance scale `s`.

References
----------
Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models",
arXiv:2206.00364.

Ho & Salimans, "Classifier-Free Diffusion Guidance", arXiv:2207.12598.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class VolumeConditionerConfig:
    """Hyperparameters for :class:`VolumeConditioner`."""

    fourier_features: int = 32
    fourier_sigma: float = 1.0
    embed_dim: int = 256  # must equal block_out_channels[0] * 4 of MAISI U-Net
    hidden_dim: int = 256


class FourierFeatures(nn.Module):
    """Random Fourier features for a scalar input.

    Frequencies sampled at construction time and frozen. Output is
    ``[sin(2π f_k · x), cos(2π f_k · x)]`` of total length ``2 * num_features``.
    """

    def __init__(self, num_features: int, sigma: float, seed: int = 1234) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        # Frozen frequencies: registered as buffer so `.to(device)` moves them.
        freqs = torch.randn(num_features, generator=gen) * sigma
        self.register_buffer("freqs", freqs, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B] or [B, 1]
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # [B, 1]
        # 2π f_k · x → [B, K]
        proj = 2 * torch.pi * x * self.freqs.to(dtype=x.dtype)
        return torch.cat([proj.sin(), proj.cos()], dim=-1)  # [B, 2K]


class VolumeConditioner(nn.Module):
    """Map scalar log-V → 256-d embedding to be added to the U-Net's time+class emb.

    Forward expects a `(B,)` log-V tensor and an optional boolean mask
    `use_uncond[b] == True` selecting the learned null token for entry b.
    """

    def __init__(self, cfg: VolumeConditionerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.fourier = FourierFeatures(cfg.fourier_features, cfg.fourier_sigma)
        in_dim = 2 * cfg.fourier_features  # sin + cos
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.embed_dim),
        )
        # Identity-at-init: zero the last linear so conditioner output is zero.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        # Learned null token; zero-init so c_∅ contribution is also zero at start.
        self.c_null = nn.Parameter(torch.zeros(cfg.embed_dim))

    def forward(
        self,
        log_v: torch.Tensor,
        use_uncond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return `[B, embed_dim]` embedding.

        Parameters
        ----------
        log_v : torch.Tensor
            Shape `(B,)` float. Natural log of tumor volume.
        use_uncond : torch.Tensor | None
            Optional boolean tensor `(B,)`. Where True, the entry is replaced by
            the learned null token. None ≡ all False (fully conditioned).
        """
        if log_v.dim() != 1:
            raise ValueError(f"log_v must be 1-D, got {tuple(log_v.shape)}")
        feats = self.fourier(log_v)  # [B, 2K]
        emb = self.mlp(feats)  # [B, embed_dim]
        if use_uncond is not None:
            mask = use_uncond.view(-1, 1).to(dtype=emb.dtype)
            emb = mask * self.c_null.unsqueeze(0).to(dtype=emb.dtype) + (1.0 - mask) * emb
        return emb

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]
