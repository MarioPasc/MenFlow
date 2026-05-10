"""Steered-decode routine for E2.4 — Phase A.

Per-anchor workflow
-------------------

For one held-out anchor scan ``i`` and one ``Δ log V`` step:

1. Read the latent ``z0 = latents[i, modality_index]`` of shape
   ``(C, H', W', D')`` from the latents H5; cast to the routine ``dtype`` and
   send to GPU.
2. Read the source segmentation, isolate the tumor labels, replicate the
   encoder's spatial preparation (currently ``spatial_op == "none"`` plus
   trailing zero-pad to ``padded_spatial_shape``), then project to latent
   resolution via :func:`menflow.latent_features.mask.project_mask_to_latent`
   (stride 4 max-pool). Optionally dilate by ``mask_dilation_radius``.
3. Compute the steered latent via
   :func:`menflow.steering.operator.local_steer` (or ``global_steer``):
   ``z' = z0 + alpha * u_v ⊗ mask_lat`` with
   ``alpha = Δ log V / direction_norm``.
4. Decode ``z'`` -> padded reconstruction; crop the trailing pad and
   inverse-rescale to source intensities using the per-scan percentiles stored
   in the latents H5.
5. Re-encode the cropped+rescaled image to compute off-manifold drift.
6. Save the decoded volume as a gzipped NIfTI under
   ``decoded/{scan_id}/{scan_id}__delta_{:+.2f}.nii.gz``; record drift +
   intensity proxy + the NIfTI path in the per-anchor accumulator.

A single ``sweep_results.h5`` is written at the end with all metrics flat-laid
across ``(n_anchors, n_deltas)``. The ``decoded_log_v`` and
``tumor_voxels_decoded`` fields are pre-filled with sentinel values (NaN and
``-1``) — a Phase-B segmenter consumes the NIfTIs and overwrites them via
:func:`experiments.E2_4_causal_steering.analysis.segmenter_interface.fill_sweep_results`.

Schema is provisional (per ``.claude/rules/h5-format.md`` §"Provisional
schemas") so :func:`menflow.data.h5_schema.assert_h5_valid` is intentionally
not invoked.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import nibabel as nib
import numpy as np
import torch
import yaml
from tqdm import tqdm

from menflow.latent_features.mask import project_mask_to_latent
from menflow.latent_features.volume import voxel_count_to_log_volume_cm3
from menflow.maisi_autoencoder.config import MaisiV2Config
from menflow.maisi_autoencoder.model import MaisiAutoencoder
from menflow.maisi_autoencoder.transforms import PercentileNormalizer
from menflow.steering.anchors import StratifiedAnchor, stratified_anchor_indices
from menflow.steering.calibration import delta_to_alpha
from menflow.steering.drift import off_manifold_drift
from menflow.steering.mask_dilation import dilate_mask_lat
from menflow.steering.operator import STEER_OPERATORS

logger = logging.getLogger(__name__)


SWEEP_SCHEMA_VERSION = "0.1-provisional"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SteerDecodeRoutineConfig:
    """Everything the routine needs to produce a sweep_results.h5 + NIfTIs.

    Attributes
    ----------
    latents_h5
        Self-describing latents H5 (``brats_men_maisi_latents.h5``).
    source_h5
        Unified-schema source H5 (``brats_men.h5``); used for masks + spacing.
    direction_npz
        E2.1 / E2.2 direction artifact with ``direction (C,)`` and
        ``direction_norm`` scalar.
    checkpoint
        MAISI-v2 checkpoint path (``autoencoder_v2.pt``).
    output_dir
        Root for ``decoded/``, ``sweep_results.h5``, ``anchors.json``,
        ``config_used.yaml``.
    modality, modality_index
        Modality name + its column in ``latents`` axis 1. ``modality_index`` is
        cross-checked against the latents H5 ``modalities`` attr.
    operator
        ``"local"`` (default per E2.3 carry-forward), ``"strongly_local"``
        (alias), or ``"global"`` (control comparator).
    deltas
        Sweep grid in ``Δ log V`` (nats).
    mask_dilation_radius
        Latent-voxel dilation radius applied to ``mask_lat`` before steering;
        0 by default (matches E2.4 spec literally).
    n_anchors_per_bin
        Anchors drawn per ``log V`` bin.
    volume_bin_edges_log_cm3
        Length-(B+1) edges in ``log V`` (nats, cm³). Default reflects E1's
        B1-B5 layout.
    splits_to_use
        Which splits to sample anchors from (``"val"`` by default; ``"train"``
        only for the smoke variant).
    dtype
        Decode dtype (``"float16"`` recommended on the 3060).
    device
        Torch device string.
    seed
        RNG seed (anchor sampling, latent shuffles).
    save_decoded_nifti
        If False, drift + intensity are still computed but no NIfTI is written.
    save_steered_latents
        Debug toggle; persists the steered latents next to NIfTIs (large).
    intensity_rescale
        If True, undo the per-scan percentile rescale stored in the latents H5
        before saving the NIfTI. Always recommended (default).
    log_level
        Python logging level string.
    model
        MAISI architecture override (rarely needed; defaults match
        ``MaisiV2Config()``).
    """

    latents_h5: Path
    source_h5: Path
    direction_npz: Path
    checkpoint: Path
    output_dir: Path
    modality: str = "t1c"
    modality_index: int | None = None
    operator: str = "local"
    deltas: tuple[float, ...] = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
    mask_dilation_radius: int = 0
    n_anchors_per_bin: int = 1
    volume_bin_edges_log_cm3: tuple[float, ...] = (-2.3, -0.5, 0.7, 1.5, 2.3, 4.5)
    splits_to_use: tuple[str, ...] = ("val",)
    mask_label_set: tuple[int, ...] = (1, 2, 3)
    dtype: str = "float16"
    device: str = "cuda"
    seed: int = 0
    save_decoded_nifti: bool = True
    save_steered_latents: bool = False
    intensity_rescale: bool = True
    log_level: str = "INFO"
    model: MaisiV2Config = dataclasses.field(default_factory=MaisiV2Config)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SteerDecodeRoutineConfig:
        """Load config from YAML; coerce path-like fields and tuples."""
        with open(path) as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        # Drop top-level keys that belong to the calling experiment, not the
        # routine, so the experiment YAML can be a superset of the routine YAML.
        for k in ("slurm", "mode", "write_phase_a_report"):
            raw.pop(k, None)
        model = MaisiV2Config(**(raw.pop("model", {}) or {}))
        for path_key in (
            "latents_h5",
            "source_h5",
            "direction_npz",
            "checkpoint",
            "output_dir",
        ):
            if path_key in raw and raw[path_key] is not None:
                raw[path_key] = Path(str(raw[path_key])).expanduser()
        for tup_key in (
            "deltas",
            "volume_bin_edges_log_cm3",
            "splits_to_use",
            "mask_label_set",
        ):
            if tup_key in raw and raw[tup_key] is not None:
                raw[tup_key] = tuple(raw[tup_key])
        return cls(model=model, **raw)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SteerDecodeEngine:
    """Drive a steered-decode sweep over a stratified set of anchors."""

    def __init__(self, config: SteerDecodeRoutineConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Path:
        cfg = self.config
        for p in (cfg.latents_h5, cfg.source_h5, cfg.direction_npz, cfg.checkpoint):
            if not Path(p).is_file():
                raise FileNotFoundError(p)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        decoded_root = cfg.output_dir / "decoded"
        decoded_root.mkdir(parents=True, exist_ok=True)

        if cfg.operator not in STEER_OPERATORS:
            raise ValueError(
                f"unknown operator {cfg.operator!r}; choose from {sorted(STEER_OPERATORS)}"
            )
        steer_fn = STEER_OPERATORS[cfg.operator]

        # 1) Load direction.
        u_v_np, direction_norm = _load_direction(cfg.direction_npz)
        logger.info(
            "Direction loaded: shape=%s, norm=%.4f from %s",
            u_v_np.shape,
            direction_norm,
            cfg.direction_npz,
        )

        with h5py.File(cfg.latents_h5, "r") as lat, h5py.File(cfg.source_h5, "r") as src:
            geom = _Geometry.from_attrs(lat, src)
            if u_v_np.shape != (geom.latent_channels,):
                raise ValueError(
                    f"direction shape {u_v_np.shape} != latent_channels ({geom.latent_channels},)"
                )
            modality_idx = self._resolve_modality_index(lat, geom.modalities)
            spacing_mm = geom.spacing_mm

            # 2) Pick anchor candidates. Compute log V only for the requested
            # split scans (not all n_scans) — a full segmentation read would
            # take ~40 s for BraTS-MEN.
            split_scan_indices = self._scan_indices_in_splits(lat)
            log_v, mask_present = _compute_per_scan_log_v(
                src,
                label_set=cfg.mask_label_set,
                geom=geom,
                n_scans=geom.n_scans,
                only_indices=split_scan_indices,
            )
            cand_indices = split_scan_indices[mask_present[split_scan_indices]]
            logger.info(
                "Anchor pool: %d scans (splits=%s, mask_lat_present=%d/%d in pool)",
                cand_indices.size,
                cfg.splits_to_use,
                int(mask_present[split_scan_indices].sum()),
                split_scan_indices.size,
            )
            if cand_indices.size == 0:
                raise RuntimeError(
                    "Anchor pool is empty after mask-presence filter. The "
                    "official BraTS-MEN-2023 val split ships without "
                    "segmentations; use splits_to_use=['train'] (with the "
                    "documented caveat that probe CV folds touched every "
                    "patient) until a held-out cohort with masks is wired in."
                )

            # 3) Stratified sampling.
            anchors = stratified_anchor_indices(
                log_v=log_v,
                candidate_indices=cand_indices,
                bin_edges=np.asarray(cfg.volume_bin_edges_log_cm3, dtype=np.float64),
                n_per_bin=cfg.n_anchors_per_bin,
                seed=cfg.seed,
            )
            scan_ids = [_decode(s) for s in lat["scan_ids"][:].tolist()]
            anchor_scan_ids = [scan_ids[a.index] for a in anchors]
            _save_anchors_json(cfg.output_dir / "anchors.json", anchors, anchor_scan_ids)
            logger.info(
                "Sampled %d anchors across bins %s",
                len(anchors),
                [a.bin_id for a in anchors],
            )

            # 4) Load model.
            model = MaisiAutoencoder.from_checkpoint(
                cfg.checkpoint,
                config=cfg.model,
                device=cfg.device,
                dtype=cfg.dtype,
            )
            torch_device = torch.device(cfg.device)
            torch_dtype = next(model.parameters()).dtype

            u_v_torch = torch.from_numpy(u_v_np).to(device=torch_device, dtype=torch_dtype)

            # 5) Sweep.
            results = self._sweep(
                lat=lat,
                src=src,
                anchors=anchors,
                anchor_scan_ids=anchor_scan_ids,
                modality_idx=modality_idx,
                geom=geom,
                u_v=u_v_torch,
                direction_norm=direction_norm,
                steer_fn=steer_fn,
                model=model,
                torch_device=torch_device,
                torch_dtype=torch_dtype,
                decoded_root=decoded_root,
                spacing_mm=spacing_mm,
            )

            # 6) Persist.
            sweep_h5 = cfg.output_dir / "sweep_results.h5"
            _write_sweep_h5(
                sweep_h5,
                cfg=cfg,
                results=results,
                anchors=anchors,
                anchor_scan_ids=anchor_scan_ids,
                direction_norm=direction_norm,
                geom=geom,
                modality_idx=modality_idx,
            )
            self._save_used_config(cfg.output_dir / "config_used.yaml")

        logger.info("Wrote sweep_results.h5 -> %s", sweep_h5)
        return sweep_h5

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_modality_index(self, lat: h5py.File, modalities: tuple[str, ...]) -> int:
        cfg = self.config
        if cfg.modality_index is not None:
            if not 0 <= cfg.modality_index < len(modalities):
                raise ValueError(
                    f"modality_index {cfg.modality_index} out of range for "
                    f"{len(modalities)} modalities {modalities}"
                )
            if modalities[cfg.modality_index] != cfg.modality:
                raise ValueError(
                    f"modality_index {cfg.modality_index} maps to "
                    f"{modalities[cfg.modality_index]!r}, not {cfg.modality!r}"
                )
            return cfg.modality_index
        if cfg.modality not in modalities:
            raise ValueError(f"modality {cfg.modality!r} not in latents H5 ({modalities})")
        return modalities.index(cfg.modality)

    def _scan_indices_in_splits(self, lat: h5py.File) -> np.ndarray:
        """Return the unique sorted scan indices belonging to ``splits_to_use``."""
        cfg = self.config
        if "splits" not in lat:
            raise RuntimeError("latents H5 has no /splits group; cannot select anchors")
        if "longitudinal/patient_offsets" not in lat:
            raise RuntimeError("latents H5 has no /longitudinal/patient_offsets")
        offsets = lat["longitudinal/patient_offsets"][:].astype(np.int64)
        scan_indices: list[np.ndarray] = []
        for split in cfg.splits_to_use:
            if split not in lat["splits"]:
                raise RuntimeError(f"requested split {split!r} not in latents H5")
            patient_idx = lat["splits"][split][:].astype(np.int64)
            for p in patient_idx:
                lo, hi = int(offsets[p]), int(offsets[p + 1])
                scan_indices.append(np.arange(lo, hi, dtype=np.int64))
        if not scan_indices:
            return np.array([], dtype=np.int64)
        return np.unique(np.concatenate(scan_indices))

    def _sweep(
        self,
        *,
        lat: h5py.File,
        src: h5py.File,
        anchors: list[StratifiedAnchor],
        anchor_scan_ids: list[str],
        modality_idx: int,
        geom: _Geometry,
        u_v: torch.Tensor,
        direction_norm: float,
        steer_fn: Any,
        model: MaisiAutoencoder,
        torch_device: torch.device,
        torch_dtype: torch.dtype,
        decoded_root: Path,
        spacing_mm: tuple[float, float, float],
    ) -> _SweepArrays:
        cfg = self.config
        deltas = np.asarray(cfg.deltas, dtype=np.float32)
        alphas = np.array(
            [delta_to_alpha(float(d), direction_norm) for d in deltas], dtype=np.float32
        )

        a, n_d = len(anchors), len(deltas)
        drift = np.full((a, n_d), np.nan, dtype=np.float32)
        intensity_proxy = np.full((a, n_d), np.nan, dtype=np.float32)
        nifti_paths = np.empty((a, n_d), dtype=object)
        nifti_paths.fill("")

        affine = _affine_from_spacing(spacing_mm)
        spatial_affine_dtype = np.float64

        for ai, anchor in enumerate(tqdm(anchors, desc="anchors")):
            scan_id = anchor_scan_ids[ai]
            anchor_dir = decoded_root / scan_id
            if cfg.save_decoded_nifti:
                anchor_dir.mkdir(parents=True, exist_ok=True)

            # Latent + mask are constant across deltas.
            z0_np = lat["latents"][anchor.index, modality_idx].astype(np.float32)
            z0 = torch.from_numpy(z0_np).to(device=torch_device, dtype=torch_dtype)

            mask_src_full = np.isin(
                src["segmentations"][anchor.index].astype(np.int32),
                np.asarray(cfg.mask_label_set, dtype=np.int32),
            )
            mask_padded = _pad_mask_to_padded_shape(mask_src_full, geom)
            mask_lat = project_mask_to_latent(
                torch.from_numpy(mask_padded).to(torch_device),
                stride=geom.stride,
            )[0, 0]  # (H', W', D'), bool
            mask_lat = dilate_mask_lat(mask_lat, radius=cfg.mask_dilation_radius)

            lower = float(lat["intensity_lower"][anchor.index, modality_idx])
            upper = float(lat["intensity_upper"][anchor.index, modality_idx])
            normalizer = PercentileNormalizer(
                lower_value=lower, upper_value=upper, b_min=0.0, b_max=1.0
            )

            for di, delta in enumerate(deltas):
                alpha = float(alphas[di])
                z_prime = steer_fn(z0, mask_lat, u_v, alpha)  # (C, H', W', D')
                z_in = z_prime[None]  # (1, C, H', W', D')

                x_pad = model.decode(z_in)[0, 0].detach().to("cpu", dtype=torch.float32).numpy()
                x_cropped = _crop_to_source(x_pad, geom)
                if cfg.intensity_rescale:
                    x_image = normalizer.inverse(x_cropped).astype(np.float32, copy=False)
                else:
                    x_image = x_cropped.astype(np.float32, copy=False)

                if cfg.save_decoded_nifti:
                    rel = anchor_dir / f"{scan_id}__delta_{delta:+.2f}.nii.gz"
                    img = nib.Nifti1Image(x_image, affine.astype(spatial_affine_dtype))
                    nib.save(img, str(rel))
                    nifti_paths[ai, di] = str(rel.relative_to(cfg.output_dir))

                # Re-encode for drift. Encoder expects [b_min, b_max] input.
                x_for_encoder = normalizer.forward(x_image) if cfg.intensity_rescale else x_image
                x_padded_for_encoder = _pad_to_padded_shape(x_for_encoder, geom)
                x_t = torch.from_numpy(x_padded_for_encoder).to(
                    device=torch_device, dtype=torch_dtype
                )[None, None]
                z_re = model.encode(x_t, deterministic=True)[0]  # (C,H',W',D')
                drift[ai, di] = off_manifold_drift(z_prime, z_re)

                # Intensity proxy: mean intensity inside the (un-dilated) source mask.
                intensity_proxy[ai, di] = float(_masked_mean(x_image, mask_src_full))

                if cfg.save_steered_latents:
                    np.save(
                        anchor_dir / f"{scan_id}__delta_{delta:+.2f}__zprime.npy",
                        z_prime.detach().to("cpu", dtype=torch.float32).numpy(),
                    )

                del z_prime, z_in, x_pad, x_cropped, x_image, x_t, z_re
                if torch_device.type == "cuda":
                    torch.cuda.empty_cache()

            del z0
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()

        log_v0 = np.array([a.log_v for a in anchors], dtype=np.float32)
        predicted = log_v0[:, None] + deltas[None, :]

        return _SweepArrays(
            scan_ids=np.asarray(anchor_scan_ids, dtype=object),
            log_v0=log_v0,
            volume_bin=np.array([a.bin_id for a in anchors], dtype=np.int8),
            delta_log_v_grid=deltas,
            alpha_grid=alphas,
            predicted_log_v=predicted.astype(np.float32),
            drift=drift,
            intensity_proxy=intensity_proxy,
            nifti_paths=nifti_paths,
        )

    def _save_used_config(self, path: Path) -> None:
        cfg_dict = _jsonable(dataclasses.asdict(self.config))
        with open(path, "w") as fh:
            yaml.safe_dump(cfg_dict, fh, sort_keys=False)


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Geometry:
    """Resolved spatial geometry, derived from latents+source attrs."""

    n_scans: int
    modalities: tuple[str, ...]
    source_shape: tuple[int, int, int]
    working_shape: tuple[int, int, int]
    padded_shape: tuple[int, int, int]
    latent_spatial: tuple[int, int, int]
    latent_channels: int
    spatial_op: str
    crop_offset: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    stride: int

    @classmethod
    def from_attrs(cls, lat: h5py.File, src: h5py.File) -> _Geometry:
        modalities = tuple(_decode(m) for m in lat.attrs["modalities"])
        source_shape = tuple(int(x) for x in lat.attrs["source_spatial_shape"])
        working_shape = tuple(int(x) for x in lat.attrs["working_spatial_shape"])
        padded_shape = tuple(int(x) for x in lat.attrs["padded_spatial_shape"])
        latent_spatial = tuple(int(x) for x in lat.attrs["latent_spatial_shape"])
        latent_channels = int(lat.attrs["latent_channels"])
        spatial_op = _decode(lat.attrs["spatial_op"])
        crop_offset = tuple(int(x) for x in lat.attrs.get("crop_offset", (0, 0, 0)))
        spacing_mm = tuple(float(x) for x in src.attrs["spacing_mm"])
        ratios = [p // l for p, l in zip(padded_shape, latent_spatial)]
        if len(set(ratios)) != 1:
            raise ValueError(
                f"non-uniform latent stride: padded={padded_shape}, latent={latent_spatial}"
            )
        return cls(
            n_scans=int(lat.attrs["n_scans"]),
            modalities=modalities,
            source_shape=source_shape,  # type: ignore[arg-type]
            working_shape=working_shape,  # type: ignore[arg-type]
            padded_shape=padded_shape,  # type: ignore[arg-type]
            latent_spatial=latent_spatial,  # type: ignore[arg-type]
            latent_channels=latent_channels,
            spatial_op=spatial_op,
            crop_offset=crop_offset,  # type: ignore[arg-type]
            spacing_mm=spacing_mm,  # type: ignore[arg-type]
            stride=int(ratios[0]),
        )


# ---------------------------------------------------------------------------
# Output bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SweepArrays:
    scan_ids: np.ndarray  # (A,) vlen str
    log_v0: np.ndarray  # (A,)
    volume_bin: np.ndarray  # (A,)
    delta_log_v_grid: np.ndarray  # (D,)
    alpha_grid: np.ndarray  # (D,)
    predicted_log_v: np.ndarray  # (A, D)
    drift: np.ndarray  # (A, D)
    intensity_proxy: np.ndarray  # (A, D)
    nifti_paths: np.ndarray  # (A, D) vlen str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind in {"S", "U", "O"}:
        try:
            return _decode(value.item())
        except Exception:  # pragma: no cover
            return str(value)
    return str(value)


def _pad_mask_to_padded_shape(mask: np.ndarray, geom: _Geometry) -> np.ndarray:
    """Replicate the encoder's ``spatial_op`` on a binary mask.

    Currently supports ``"none"`` (the production setting) — the mask is
    zero-padded on the trailing edge of each axis to match
    ``padded_spatial_shape``.
    """
    if geom.spatial_op != "none":
        raise NotImplementedError(f"spatial_op={geom.spatial_op!r} not supported by E2.4 Phase A")
    pad_widths = tuple((0, p - s) for p, s in zip(geom.padded_shape, mask.shape))
    if any(p[1] for p in pad_widths):
        return np.pad(mask.astype(bool), pad_widths, mode="constant", constant_values=False)
    return mask.astype(bool)


def _pad_to_padded_shape(volume: np.ndarray, geom: _Geometry) -> np.ndarray:
    """Same trailing-edge zero-pad as :func:`_pad_mask_to_padded_shape` for floats."""
    pad_widths = tuple((0, p - s) for p, s in zip(geom.padded_shape, volume.shape))
    if any(p[1] for p in pad_widths):
        return np.pad(volume, pad_widths, mode="constant", constant_values=0.0)
    return volume


def _crop_to_source(padded: np.ndarray, geom: _Geometry) -> np.ndarray:
    """Crop the trailing zero-pad introduced before encoding."""
    sl = tuple(slice(0, s) for s in geom.source_shape)
    return padded[sl]


def _masked_mean(volume: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    return float(volume[mask].mean())


def _affine_from_spacing(spacing_mm: tuple[float, float, float]) -> np.ndarray:
    """Identity-orientation affine with the cohort spacing on the diagonal."""
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = float(spacing_mm[0])
    affine[1, 1] = float(spacing_mm[1])
    affine[2, 2] = float(spacing_mm[2])
    return affine


def _compute_per_scan_log_v(
    src: h5py.File,
    *,
    label_set: tuple[int, ...],
    geom: _Geometry,
    n_scans: int,
    only_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``log_v[N]`` and ``mask_present[N]`` from the source segmentations.

    A scan is treated as ``mask_lat_present`` iff its source mask has at least
    ``stride**3`` voxels — the necessary condition for a non-empty latent mask
    after stride-``stride`` max-pool. (A more conservative upstream definition
    uses the actual projected count; reusing it here would force a full
    GPU pass. The voxel-count proxy is a strict subset.)

    When ``only_indices`` is provided, segmentations for the rest of the cohort
    are not read from disk — the corresponding ``log_v`` / ``mask_present``
    entries stay at their zero / False defaults. Saves ~40 s on BraTS-MEN.
    """
    log_v = np.zeros(n_scans, dtype=np.float32)
    n_vox = np.zeros(n_scans, dtype=np.int32)
    label_arr = np.asarray(label_set, dtype=np.int32)
    has_seg = src["has_segmentation"][:n_scans].astype(bool)
    iter_indices = (
        np.asarray(only_indices, dtype=np.int64)
        if only_indices is not None
        else np.arange(n_scans, dtype=np.int64)
    )
    for i in iter_indices:
        if not has_seg[i]:
            continue
        seg_i = src["segmentations"][int(i)].astype(np.int32)
        n_vox[i] = int(np.isin(seg_i, label_arr).sum())
        log_v[i] = float(voxel_count_to_log_volume_cm3(n_vox[i], geom.spacing_mm))
    mask_present = (n_vox >= geom.stride**3) & has_seg
    return log_v, mask_present


def _load_direction(npz_path: Path) -> tuple[np.ndarray, float]:
    """Load ``direction (C,)`` and ``direction_norm`` scalar from an E2.1 NPZ."""
    with np.load(npz_path) as data:
        if "direction" not in data.files:
            raise KeyError(f"{npz_path} missing 'direction' field")
        direction = np.asarray(data["direction"], dtype=np.float32)
        if "direction_norm" in data.files:
            norm = float(np.asarray(data["direction_norm"]).item())
        else:
            # Back-compat: derive from coef_raw if direction_norm absent.
            if "coef_raw" not in data.files:
                raise KeyError(f"{npz_path} has neither 'direction_norm' nor 'coef_raw'")
            norm = float(np.linalg.norm(np.asarray(data["coef_raw"])))
    if direction.ndim != 1:
        raise ValueError(f"direction must be 1-D; got shape {direction.shape}")
    n = float(np.linalg.norm(direction))
    if not 0.99 <= n <= 1.01:
        raise ValueError(f"direction is not unit-norm (||u||={n:.4f})")
    return direction, norm


def _save_anchors_json(path: Path, anchors: list[StratifiedAnchor], scan_ids: list[str]) -> None:
    payload = [
        {
            "row_index": int(a.index),
            "scan_id": scan_ids[i],
            "log_v": float(a.log_v),
            "volume_bin": int(a.bin_id),
        }
        for i, a in enumerate(anchors)
    ]
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def _write_sweep_h5(
    path: Path,
    *,
    cfg: SteerDecodeRoutineConfig,
    results: _SweepArrays,
    anchors: list[StratifiedAnchor],
    anchor_scan_ids: list[str],
    direction_norm: float,
    geom: _Geometry,
    modality_idx: int,
) -> None:
    """Persist the provisional sweep H5 with NaN/-1 sentinels for Phase B fields."""
    a, n_d = results.predicted_log_v.shape
    decoded_log_v = np.full((a, n_d), np.nan, dtype=np.float32)
    tumor_voxels_decoded = np.full((a, n_d), -1, dtype=np.int32)
    vlen = h5py.special_dtype(vlen=str)
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = SWEEP_SCHEMA_VERSION
        f.attrs["phase"] = "A_decoded_only"
        f.attrs["segmenter_completed"] = False
        f.attrs["created_at"] = _dt.datetime.now(_dt.UTC).isoformat()
        f.attrs["latents_h5"] = str(cfg.latents_h5)
        f.attrs["source_h5"] = str(cfg.source_h5)
        f.attrs["direction_npz"] = str(cfg.direction_npz)
        f.attrs["direction_norm"] = float(direction_norm)
        f.attrs["modality"] = cfg.modality
        f.attrs["modality_index"] = np.int64(modality_idx)
        f.attrs["operator"] = cfg.operator
        f.attrs["mask_dilation_radius"] = np.int64(cfg.mask_dilation_radius)
        f.attrs["dtype"] = cfg.dtype
        f.attrs["seed"] = np.int64(cfg.seed)
        f.attrs["spacing_mm"] = np.asarray(geom.spacing_mm, dtype=np.float64)
        f.attrs["latent_spatial_shape"] = np.asarray(geom.latent_spatial, dtype=np.int64)
        f.attrs["latent_channels"] = np.int64(geom.latent_channels)
        f.attrs["source_spatial_shape"] = np.asarray(geom.source_shape, dtype=np.int64)
        f.attrs["mask_label_set"] = np.asarray(cfg.mask_label_set, dtype=np.int32)
        f.attrs["volume_bin_edges_log_cm3"] = np.asarray(
            cfg.volume_bin_edges_log_cm3, dtype=np.float64
        )
        f.attrs["routine_config"] = json.dumps(_jsonable(dataclasses.asdict(cfg)))
        f.attrs["anchor_row_indices"] = np.asarray([a.index for a in anchors], dtype=np.int64)

        f.create_dataset("scan_id", data=results.scan_ids, dtype=vlen)
        f.create_dataset("log_v0", data=results.log_v0)
        f.create_dataset("volume_bin", data=results.volume_bin)
        f.create_dataset("delta_log_v_grid", data=results.delta_log_v_grid)
        f.create_dataset("alpha_grid", data=results.alpha_grid)
        f.create_dataset("predicted_log_v", data=results.predicted_log_v)
        f.create_dataset("drift", data=results.drift)
        f.create_dataset("intensity_proxy", data=results.intensity_proxy)
        f.create_dataset("decoded_log_v", data=decoded_log_v)
        f.create_dataset("tumor_voxels_decoded", data=tumor_voxels_decoded)
        f.create_dataset("decoded_nifti_path", data=results.nifti_paths, dtype=vlen)


def _jsonable(obj: object) -> object:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
