"""MAISI-v2 VAE-GAN integration: model wrapper, transforms, latent I/O."""

from __future__ import annotations

from menflow.maisi_autoencoder.config import (
    InferenceConfig,
    MaisiV2Config,
    MAISI_V2_MR_DEFAULTS,
)
from menflow.maisi_autoencoder.latents_h5 import (
    LATENT_SCHEMA_VERSION,
    LatentH5Writer,
    read_latent_h5,
)
from menflow.maisi_autoencoder.model import MaisiAutoencoder
from menflow.maisi_autoencoder.transforms import (
    PercentileNormalizer,
    build_postprocess,
    build_preprocess,
)

__all__ = [
    "InferenceConfig",
    "MaisiV2Config",
    "MAISI_V2_MR_DEFAULTS",
    "MaisiAutoencoder",
    "LatentH5Writer",
    "LATENT_SCHEMA_VERSION",
    "PercentileNormalizer",
    "build_postprocess",
    "build_preprocess",
    "read_latent_h5",
]
