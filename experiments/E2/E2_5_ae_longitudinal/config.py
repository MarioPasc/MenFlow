"""Frozen configuration for E2.5 — AE longitudinal traversal diagnostic."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass(frozen=True, slots=True)
class AELongitudinalConfig:
    """Pydantic-style frozen config for the E2.5 diagnostic.

    Paths are resolved from the YAML at load time. Default values mirror
    the spec (`E2_5_ae_longitudinal_diagnostic.md`) §7.2 where applicable;
    deviations are documented in the plan file.
    """

    # ---- Inputs ----
    unified_h5: Path
    latents_h5: Path
    e2_1_direction_npz: dict[str, Path]
    maisi_checkpoint: Path

    # ---- Cohort ----
    min_timepoints: int = 3
    min_log_volume_spread: float = 0.3
    min_log_volume_spread_relaxed: tuple[float, ...] = (0.2, 0.1)
    min_effective_cohort: int = 20

    # ---- Diagnostic ----
    modalities: tuple[str, ...] = ("t1c", "t1n", "t2f", "t2w")
    mask_label_set: tuple[int, ...] = (1, 2, 3)
    beta_grid_size: int = 21
    anatomy_beta_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    use_segmenter_dice: bool = False

    # ---- Mask handling ----
    mask_dilation_voxels: int = 2  # native-resolution dilation of union endpoint mask
    mask_pool_strategy: Literal["union_endpoints"] = "union_endpoints"

    # ---- Null baseline ----
    n_null_triples: int = 500
    null_random_seed: int = 42

    # ---- Bootstrap ----
    n_bootstrap_resamples: int = 1000
    bootstrap_ci_alpha: float = 0.05
    bootstrap_seed: int = 0

    # ---- Decision thresholds (spec §9) ----
    rho_lin_pass: float = 0.30
    rho_lin_intermediate: float = 0.50
    dice_pass: float = 0.60
    dice_intermediate: float = 0.40
    ssim_nontumour_pass: float = 0.85
    ssim_nontumour_intermediate: float = 0.80
    round_trip_ssim_threshold: float = 0.90
    round_trip_pass_fraction: float = 0.80

    # ---- Hardware ----
    device: str = "cuda"
    decode_dtype: str = "float16"
    skip_decode: bool = False  # smoke-only short-circuit
    # MAISI tiled-convolution split count. The default (4) fits Picasso A100;
    # the 4060 (8 GB) typically needs 8 or 16 to avoid OOM on the 240³ decode.
    num_splits: int = 4

    # ---- IO / logging ----
    output_dir: Path = Path(
        "/media/mpascual/Sandisk2TB/research/menflow/experiments/E2/E2.5_ae_longitudinal"
    )
    max_patients: int | None = None  # limit cohort size for smoke
    n_qualitative_panel_patients: int = 3
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: str | Path) -> AELongitudinalConfig:
        """Load a YAML config.

        ``e2_1_direction_npz`` is read as a mapping ``{modality: path}``; all
        other path-valued fields are converted to :class:`pathlib.Path`.
        """
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        raw.pop("slurm", None)

        path_keys = ("unified_h5", "latents_h5", "maisi_checkpoint", "output_dir")
        for k in path_keys:
            if k in raw and raw[k] is not None:
                raw[k] = Path(raw[k]).expanduser()

        if "e2_1_direction_npz" in raw and raw["e2_1_direction_npz"] is not None:
            raw["e2_1_direction_npz"] = {
                str(k): Path(v).expanduser() for k, v in raw["e2_1_direction_npz"].items()
            }
        for k in ("modalities", "anatomy_beta_grid", "mask_label_set", "min_log_volume_spread_relaxed"):
            if k in raw and raw[k] is not None:
                raw[k] = tuple(raw[k])
        return cls(**raw)
