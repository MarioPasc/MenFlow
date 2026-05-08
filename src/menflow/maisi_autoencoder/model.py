"""Wrapper around :class:`monai.apps.generation.maisi.networks.AutoencoderKlMaisi`.

Provides a small, opinionated API:

* :meth:`MaisiAutoencoder.from_checkpoint` — instantiate the architecture from a
  :class:`MaisiV2Config` and load NV-Generate-CTMR's ``autoencoder_v2.pt`` weights.
* :meth:`MaisiAutoencoder.encode` — return the latent mean (``z_mu``) by default
  for fully deterministic reconstruction, or a sample from the posterior when
  ``deterministic=False``.
* :meth:`MaisiAutoencoder.decode` — single-step decode of a latent tensor.
* :meth:`MaisiAutoencoder.reconstruct` — convenience encode→decode pass returning
  ``(latent, reconstruction)``.

All inputs and outputs are raw torch tensors with shape ``(B, 1, H, W, D)``.
Numpy ↔ tensor conversion and intensity rescaling live in
:mod:`menflow.maisi_autoencoder.transforms`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from menflow.maisi_autoencoder.config import MaisiV2Config

logger = logging.getLogger(__name__)


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class MaisiAutoencoder(nn.Module):
    """Thin wrapper exposing encode/decode primitives over a frozen MAISI VAE-GAN."""

    def __init__(self, autoencoder: nn.Module, config: MaisiV2Config) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.config = config

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: MaisiV2Config | None = None,
        *,
        device: str | torch.device = "cuda",
        dtype: str | torch.dtype = "float32",
        strict: bool = True,
    ) -> "MaisiAutoencoder":
        """Construct the architecture and load weights from *checkpoint_path*.

        Parameters
        ----------
        checkpoint_path
            Path to ``autoencoder_v2.pt`` (or any state-dict-bearing checkpoint).
        config
            Optional override; defaults to :data:`MAISI_V2_MR_DEFAULTS`.
        device, dtype
            Target device and parameter dtype. ``dtype`` may be a string name
            (``"float32"`` / ``"float16"`` / ``"bfloat16"``) or a torch dtype.
        strict
            Forwarded to :func:`torch.nn.Module.load_state_dict`.
        """
        from monai.apps.generation.maisi.networks.autoencoderkl_maisi import (
            AutoencoderKlMaisi,
        )

        config = config or MaisiV2Config()
        torch_device = torch.device(device)
        torch_dtype = _resolve_dtype(dtype)

        ae = AutoencoderKlMaisi(**config.as_kwargs())
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"MAISI-v2 checkpoint not found: {ckpt_path}")

        # NV-Generate-CTMR convention: the autoencoder weights live under "unet_state_dict".
        raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and "unet_state_dict" in raw:
            state_dict = raw["unet_state_dict"]
        elif isinstance(raw, dict) and "state_dict" in raw:
            state_dict = raw["state_dict"]
        else:
            state_dict = raw  # already a state dict
        missing, unexpected = ae.load_state_dict(state_dict, strict=strict)
        if missing:
            logger.warning("Missing keys when loading MAISI-v2 weights: %s", missing[:5])
        if unexpected:
            logger.warning(
                "Unexpected keys when loading MAISI-v2 weights: %s", unexpected[:5]
            )

        ae.eval().requires_grad_(False)
        ae.to(device=torch_device, dtype=torch_dtype)
        wrapper = cls(ae, config)
        wrapper.to(device=torch_device, dtype=torch_dtype)
        logger.info(
            "Loaded MAISI-v2 autoencoder from %s on %s/%s", ckpt_path, torch_device, torch_dtype
        )
        return wrapper

    # ------------------------------------------------------------------
    # Forward primitives
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def encode(self, x: Tensor, *, deterministic: bool = True) -> Tensor:
        """Encode an input volume to a latent tensor.

        Parameters
        ----------
        x
            Input batch with shape ``(B, 1, H, W, D)``.
        deterministic
            If True (default), return the posterior mean ``z_mu``. If False, draw
            a sample via reparameterisation (``encode_stage_2_inputs``).

        Returns
        -------
        torch.Tensor
            Latent tensor with shape ``(B, latent_channels, H/4, W/4, D/4)``.
        """
        _validate_input(x)
        if deterministic:
            z_mu, _z_sigma = self.autoencoder.encode(x)
            return z_mu
        return self.autoencoder.encode_stage_2_inputs(x)

    @torch.inference_mode()
    def decode(self, z: Tensor) -> Tensor:
        """Decode a latent tensor into image space."""
        if z.ndim != 5:
            raise ValueError(f"latent must be 5D (B,C,H,W,D); got {tuple(z.shape)}")
        return self.autoencoder.decode(z)

    @torch.inference_mode()
    def reconstruct(self, x: Tensor, *, deterministic: bool = True) -> tuple[Tensor, Tensor]:
        """Encode then decode in one call."""
        z = self.encode(x, deterministic=deterministic)
        x_hat = self.decode(z)
        return z, x_hat

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def expected_latent_shape(self, spatial_shape: tuple[int, int, int]) -> tuple[int, ...]:
        """Return ``(latent_channels, H/d, W/d, D/d)`` for a given input spatial shape."""
        d = self.config.downsampling_factor
        h, w, dim = spatial_shape
        return (self.config.latent_channels, h // d, w // d, dim // d)


def _validate_input(x: Tensor) -> None:
    if x.ndim != 5:
        raise ValueError(f"input must be 5D (B,1,H,W,D); got {tuple(x.shape)}")
    if x.shape[1] != 1:
        raise ValueError(f"input must be single-channel; got C={x.shape[1]}")


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype not in _DTYPE_MAP:
        raise ValueError(f"unsupported dtype {dtype!r}; choose from {sorted(_DTYPE_MAP)}")
    return _DTYPE_MAP[dtype]
