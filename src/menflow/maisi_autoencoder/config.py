"""Configuration objects for the MAISI-v2 VAE-GAN.

The architecture parameters mirror NV-Generate-CTMR's ``config_network_rflow.json``
for the MR (v2) checkpoint. The defaults must match the saved weights bit-for-bit;
otherwise :func:`MaisiAutoencoder.from_checkpoint` will refuse to load.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class MaisiV2Config:
    """Architecture hyperparameters of ``AutoencoderKlMaisi`` for the v2 (MR) checkpoint."""

    spatial_dims: int = 3
    in_channels: int = 1
    out_channels: int = 1
    num_res_blocks: tuple[int, ...] = (2, 2, 2)
    num_channels: tuple[int, ...] = (64, 128, 256)
    attention_levels: tuple[bool, ...] = (False, False, False)
    latent_channels: int = 4
    norm_num_groups: int = 32
    norm_eps: float = 1e-6
    with_encoder_nonlocal_attn: bool = False
    with_decoder_nonlocal_attn: bool = False
    include_fc: bool = False
    use_combined_linear: bool = False
    use_flash_attention: bool = False
    use_checkpointing: bool = False
    use_convtranspose: bool = False
    num_splits: int = 4
    dim_split: int = 1
    norm_float16: bool = True  # MR config
    print_info: bool = False
    save_mem: bool = True

    def as_kwargs(self) -> dict:
        """Return the config as keyword arguments for :class:`AutoencoderKlMaisi`."""
        kwargs = asdict(self)
        # AutoencoderKlMaisi expects sequences (list/tuple); dataclass already provides tuples.
        return kwargs

    @property
    def downsampling_factor(self) -> int:
        """Spatial downsampling factor between input and latent (per axis).

        For the v2 MR config (3 stages, 2 internal stride-2 ops), this is 4.
        """
        return 2 ** (len(self.num_channels) - 1)


# Convenience alias for the canonical v2 MR defaults.
MAISI_V2_MR_DEFAULTS: MaisiV2Config = MaisiV2Config()


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Inference-time controls (independent of model weights).

    Spatial preparation is governed by ``spatial_op``:

    * ``"none"`` — feed source-shape volumes directly (Picasso default).
    * ``"resize"`` — trilinear resize to ``target_spatial_shape``. Fast but
      changes voxel volume, so any tumor-volume metric (E1 §5) becomes
      inconsistent across scans of different source shapes.
    * ``"crop"`` — center-crop to ``target_spatial_shape``. Preserves voxel
      volume; only the brain region inside the crop is encoded. Recommended
      for memory-constrained smoke tests on 12 GB GPUs.
    """

    device: str = "cuda"
    dtype: str = "float32"  # weights+activation dtype; "float16" possible on Ampere+
    deterministic: bool = True  # encode-mean (z_mu) instead of sampling
    pad_multiple: int = 16  # pad spatial dims to a multiple of this before encode
    pad_mode: str = "reflect"
    spatial_op: str = "none"  # "none" | "resize" | "crop"
    target_spatial_shape: tuple[int, int, int] | None = None
    intensity_lower_percentile: float = 0.0
    intensity_upper_percentile: float = 99.5
    intensity_b_min: float = 0.0
    intensity_b_max: float = 1.0
    intensity_clip: bool = False

    def __post_init__(self) -> None:
        if self.spatial_op not in {"none", "resize", "crop"}:
            raise ValueError(
                f"spatial_op must be one of {{none,resize,crop}}; got {self.spatial_op!r}"
            )
        if self.spatial_op != "none" and self.target_spatial_shape is None:
            raise ValueError(
                f"spatial_op={self.spatial_op!r} requires target_spatial_shape to be set"
            )

    def as_dict(self) -> dict:
        return asdict(self)
