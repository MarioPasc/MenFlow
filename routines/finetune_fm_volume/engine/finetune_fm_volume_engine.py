"""Volume-conditional MAISI-v2 FM finetuning engine.

Pipeline
--------
1. Load pretrained FM U-Net (`DiffusionModelUNetMaisi`, 180.5 M params) and its
   `scale_factor` from the checkpoint.
2. Wrap it in :class:`VolumeConditionalUNet` with a fresh
   :class:`VolumeConditioner` (zero-initialised → identity at step 0).
3. Apply LoRA adapters via :func:`peft.inject_adapter_in_model` on attention
   `to_q/to_k/to_v/out_proj` linears.
4. Open :class:`JointLatentVolumeDataset` for the configured train / val / test
   splits (default: ``e3_train``, ``e3_val``, ``e3_test``).
5. Train with predict-x₀ + L1 loss against `RFlowScheduler` (matches MAISI's
   pretraining objective). With probability `p_uncond`, the volume embedding is
   replaced by the learned null token.
6. EMA of trainable params via :class:`torch.optim.swa_utils.AveragedModel`;
   warmup + cosine LR schedule via :class:`torch.optim.lr_scheduler.SequentialLR`.
7. Streaming val metrics with :mod:`torchmetrics`.
8. Persist a self-describing run directory: config snapshot, code/env state,
   per-step CSV + final parquet, splits.json, data_stats.json, LoRA
   ``.safetensors`` artifact, and ``final_test_metrics.json`` with the
   calibration regression + 1000-resample patient-level bootstrap CIs.

References
----------
- Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data
  with Rectified Flow", arXiv:2209.03003.
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
  arXiv:2106.09685.
- MAISI training reference: NV-Generate-CTMR/scripts/diff_model_train.py:295-342.
- E3.1 §8.2 Stage-1 gate.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from safetensors.torch import save_file as safetensors_save_file
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from torchmetrics.regression import R2Score, SpearmanCorrCoef

from menflow.finetuning.calibration import fit_calibration
from menflow.finetuning.lora import (
    LoRAConfig,
    apply_lora,
)
from menflow.finetuning.unet_wrapper import (
    MAISI_MR_FM_ARCH,
    VolumeConditionalUNet,
    load_maisi_fm_unet,
    serialize_arch,
)
from menflow.finetuning.volume_conditioner import (
    VolumeConditioner,
    VolumeConditionerConfig,
)
from routines.finetune_fm_volume.engine.data import (
    DatasetConfig,
    JointLatentVolumeDataset,
    collate_samples,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Optimisation hyperparameters."""

    batch_size: int = 1
    grad_accum: int = 4
    max_steps: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 50
    p_uncond: float = 0.1
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    seed: int = 42
    num_workers: int = 2
    val_interval: int = 50
    sample_interval: int = 100
    save_interval: int = 100
    autocast_dtype: str = "float16"  # "float16" | "bfloat16" | "float32"
    activation_checkpointing: bool = False
    val_max_batches: int = 32  # cap val passes for fast checkpoints


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Per-step in-train sampling (cheap preview, not full Stage-1 eval)."""

    n_anchors: int = 2
    log_v_grid: tuple[float, ...] = (-2.0, 0.0, 2.0)
    n_ode_steps: int = 10
    cfg_scale: float = 4.0


@dataclass(frozen=True, slots=True)
class FinetuneFMVolumeRoutineConfig:
    """Top-level config for `routines/finetune_fm_volume`."""

    fm_checkpoint: Path
    latent_h5: Path
    features_h5_dir: Path
    output_dir: Path
    # Optional MAISI VAE checkpoint. When set, the engine decodes periodic
    # latent samples to NIfTI for visual inspection of the conditioning.
    ae_checkpoint: Path | None = None
    # Optional unified source H5 for the cohort. Currently unused by training
    # but recorded in the run manifest so downstream analysis scripts can
    # locate the original NIfTI volumes.
    source_h5: Path | None = None
    run_name: str = "smoke"
    modalities: tuple[str, ...] = ("t1c",)
    log_level: str = "INFO"
    # K-fold selection. The default kfold=1, fold=0 reproduces the previous
    # single-holdout behaviour (train ≈ 80 %, val ≈ 10 %, test ≈ 10 %).
    # `train_split`, `val_split`, `test_split` may be set explicitly to
    # override the (kfold, fold) → path derivation; leave None to derive.
    kfold: int = 1
    fold: int = 0
    train_split: str | None = None
    val_split: str | None = None
    test_split: str | None = None
    train: TrainConfig = field(default_factory=TrainConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    conditioner: VolumeConditionerConfig = field(default_factory=VolumeConditionerConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> FinetuneFMVolumeRoutineConfig:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        raw.pop("slurm", None)
        train = TrainConfig(**(raw.pop("train", {}) or {}))
        lora_raw = raw.pop("lora", {}) or {}
        if "target_modules" in lora_raw and lora_raw["target_modules"] is not None:
            lora_raw["target_modules"] = tuple(lora_raw["target_modules"])
        lora = LoRAConfig(**lora_raw)
        cond = VolumeConditionerConfig(**(raw.pop("conditioner", {}) or {}))
        sampling_raw = raw.pop("sampling", {}) or {}
        if "log_v_grid" in sampling_raw and sampling_raw["log_v_grid"] is not None:
            sampling_raw["log_v_grid"] = tuple(sampling_raw["log_v_grid"])
        sampling = SamplingConfig(**sampling_raw)
        for path_key in (
            "fm_checkpoint",
            "latent_h5",
            "features_h5_dir",
            "output_dir",
            "ae_checkpoint",
            "source_h5",
        ):
            if path_key in raw and raw[path_key] is not None:
                raw[path_key] = Path(raw[path_key]).expanduser()
        if "modalities" in raw and raw["modalities"] is not None:
            raw["modalities"] = tuple(raw["modalities"])
        return cls(train=train, lora=lora, conditioner=cond, sampling=sampling, **raw)


# ============================================================================
# Engine
# ============================================================================


class FinetuneFMVolumeEngine:
    """Drive a full volume-conditional FM finetune pass."""

    def __init__(self, config: FinetuneFMVolumeRoutineConfig) -> None:
        self.config = config

    def run(self) -> Path:
        cfg = self.config
        run_dir = self._init_run_dir()
        logger.info("run dir: %s", run_dir)

        torch.manual_seed(cfg.train.seed)
        np.random.seed(cfg.train.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        autocast_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[cfg.train.autocast_dtype]

        # ---- 1. Load pretrained FM and build the wrapper ----
        unet, scale_factor, ckpt_meta = load_maisi_fm_unet(
            cfg.fm_checkpoint, map_location="cpu", strict=True
        )
        logger.info(
            "loaded FM: epoch=%d, loss=%.4f, scale_factor=%.6f",
            ckpt_meta["epoch"],
            ckpt_meta["loss"],
            scale_factor,
        )
        ref_state = {k: v.detach().clone() for k, v in unet.state_dict().items()}

        conditioner = VolumeConditioner(cfg.conditioner)
        wrapper = VolumeConditionalUNet(unet, conditioner)
        adapters = apply_lora(wrapper.unet, cfg=cfg.lora)
        logger.info("inserted %d LoRA adapters at attention linears", len(adapters))

        if cfg.train.activation_checkpointing:
            self._enable_activation_checkpointing(wrapper.unet)

        wrapper.to(device)
        wrapper.train()
        for p in wrapper.unet.parameters():
            if not p.requires_grad:
                p.data = p.data.to(device)

        # ---- 2. Sanity invariant: identity at init ----
        self._assert_identity_at_init(wrapper, ref_state, device)

        # ---- 3. Data ----
        train_ds, val_ds, test_ds = self._build_datasets()
        train_path, val_path, test_path = self._resolve_split_paths()
        logger.info(
            "samples: train=%d | val=%d | test=%d",
            len(train_ds),
            len(val_ds),
            len(test_ds) if test_ds is not None else 0,
        )
        self._write_splits_artifact(run_dir, train_ds, val_ds, test_ds)
        self._write_data_stats_artifact(run_dir, train_ds, val_ds, test_ds)

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.train.num_workers,
            collate_fn=collate_samples,
            drop_last=True,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_samples,
        )

        # ---- 4. Scheduler / optimizer / EMA ----
        from monai.networks.schedulers.rectified_flow import RFlowScheduler

        scheduler = RFlowScheduler(
            num_train_timesteps=ckpt_meta["num_train_timesteps"],
            use_discrete_timesteps=False,
            use_timestep_transform=True,
            sample_method="uniform",
            scale=1.4,
        )

        groups = wrapper.trainable_parameter_groups()
        trainable_params: list[nn.Parameter] = groups["lora"] + groups["conditioner"]
        n_trainable = sum(p.numel() for p in trainable_params)
        logger.info(
            "trainable params: %d (lora=%d, conditioner=%d)",
            n_trainable,
            sum(p.numel() for p in groups["lora"]),
            sum(p.numel() for p in groups["conditioner"]),
        )
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=cfg.train.lr,
            betas=(0.9, 0.999),
            weight_decay=cfg.train.weight_decay,
        )

        # SequentialLR composes a linear warmup followed by cosine decay; the
        # math reproduces the previous manual scheduler 1:1.
        warmup_steps = max(int(cfg.train.warmup_steps), 1)
        cosine_steps = max(int(cfg.train.max_steps - warmup_steps), 1)
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps),
                CosineAnnealingLR(optimizer, T_max=cosine_steps),
            ],
            milestones=[warmup_steps],
        )

        # AveragedModel with EMA averaging fn — CPU-resident shadow.
        ema = AveragedModel(
            wrapper,
            multi_avg_fn=get_ema_multi_avg_fn(decay=cfg.train.ema_decay),
            device=torch.device("cpu"),
        )

        grad_scaler = (
            torch.amp.GradScaler("cuda")
            if device.type == "cuda" and autocast_dtype is torch.float16
            else None
        )

        # ---- 4b. Optional MAISI VAE for periodic decoded samples ----
        ae_engine = None
        latent_shape: tuple[int, int, int, int] | None = None
        try:
            sample_batch = next(iter(val_loader))
            latent_shape = tuple(int(x) for x in sample_batch["z"].shape[1:])
        except Exception:  # pragma: no cover
            latent_shape = None
        if cfg.ae_checkpoint is not None and latent_shape is not None:
            try:
                from menflow.maisi_autoencoder.config import MaisiV2Config
                from menflow.maisi_autoencoder.model import MaisiAutoencoder

                ae_engine = MaisiAutoencoder.from_checkpoint(
                    cfg.ae_checkpoint,
                    config=MaisiV2Config(),
                    device=str(device),
                    dtype="float32",
                ).eval()
                logger.info("loaded MAISI VAE for periodic sample decoding")
            except Exception as e:  # pragma: no cover
                logger.warning("could not load AE for sampling (%s); skipping decode dumps", e)
                ae_engine = None

        # ---- 5. Train loop ----
        log_paths = self._open_log_files(run_dir)
        train_csv = log_paths["train_csv"]
        val_csv = log_paths["val_csv"]
        loss_fn = nn.L1Loss(reduction="mean")

        step = 0
        start_t = time.time()
        accum_loss = 0.0
        accum_count = 0
        train_iter = iter(train_loader)
        optimizer.zero_grad(set_to_none=True)
        first_grad_recorded = False
        best_val_loss: float = float("inf")
        best_val_step: int = -1

        while step < cfg.train.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            z = batch["z"].to(device, non_blocking=True) * scale_factor
            log_v = batch["log_v"].to(device, non_blocking=True)
            class_labels = batch["modality_idx"].to(device, non_blocking=True)
            spacing = batch["spacing"].to(device=device, dtype=autocast_dtype)
            B = z.shape[0]
            noise = torch.randn_like(z)
            timesteps = scheduler.sample_timesteps(z)
            noisy = scheduler.add_noise(z, noise, timesteps)
            use_uncond = torch.rand(B, device=device) < cfg.train.p_uncond

            with torch.amp.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not torch.float32,
            ):
                pred = wrapper(
                    z_t=noisy,
                    timesteps=timesteps,
                    class_labels=class_labels,
                    spacing_tensor=spacing,
                    log_v=log_v,
                    use_uncond=use_uncond,
                )
                loss = loss_fn(pred.float(), z.float()) / cfg.train.grad_accum

            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += loss.item() * cfg.train.grad_accum
            accum_count += 1

            if accum_count % cfg.train.grad_accum == 0:
                if grad_scaler is not None:
                    grad_scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, cfg.train.grad_clip)
                if not first_grad_recorded:
                    last_lin = wrapper.conditioner.mlp[-1].weight
                    g = last_lin.grad
                    has_grad = (g is not None) and torch.any(g != 0).item()
                    logger.info(
                        "diagnostic: conditioner last-linear has nonzero grad after step 1: %s",
                        bool(has_grad),
                    )
                    first_grad_recorded = True

                if grad_scaler is not None:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update_parameters(wrapper)
                step += 1

                avg_loss = accum_loss / cfg.train.grad_accum
                vram_gb = (
                    torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
                )
                sec_per_step = (time.time() - start_t) / max(step, 1)
                self._append_csv(
                    train_csv,
                    {
                        "step": step,
                        "loss": f"{avg_loss:.6f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.6e}",
                        "grad_norm": f"{float(grad_norm):.4f}",
                        "vram_gb": f"{vram_gb:.3f}",
                        "sec_per_step": f"{sec_per_step:.3f}",
                    },
                )
                if step % 10 == 0 or step == 1:
                    logger.info(
                        "step %4d | loss %.4f | grad %.3f | vram %.2f GB | %.2f s/step",
                        step,
                        avg_loss,
                        float(grad_norm),
                        vram_gb,
                        sec_per_step,
                    )
                accum_loss = 0.0
                accum_count = 0

                if step % cfg.train.val_interval == 0 and len(val_ds) > 0:
                    val_loss_now = self._validate(
                        wrapper,
                        val_loader,
                        scheduler,
                        scale_factor,
                        autocast_dtype,
                        device,
                        val_csv,
                        step,
                        loss_fn,
                        max_batches=cfg.train.val_max_batches,
                    )
                    if val_loss_now is not None and val_loss_now < best_val_loss:
                        best_val_loss = float(val_loss_now)
                        best_val_step = step
                        self._save_best_checkpoint(run_dir, wrapper, ema, step, best_val_loss)
                if step % cfg.train.save_interval == 0:
                    self._save_checkpoint(run_dir, wrapper, ema, step)
                if step % cfg.train.sample_interval == 0 and ae_engine is not None:
                    self._dump_samples(
                        wrapper=wrapper,
                        ae=ae_engine,
                        scheduler=scheduler,
                        scale_factor=scale_factor,
                        latent_shape=latent_shape,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        run_dir=run_dir,
                        step=step,
                        label="train",
                    )

        # ---- 6. Final artifacts ----
        self._save_checkpoint(run_dir, wrapper, ema, step, final=True)
        self._save_safetensors(run_dir, wrapper, cfg.lora.rank)
        self._write_metrics_parquet(run_dir, train_csv, val_csv)

        # Dump samples from the best-val checkpoint if one was found and the
        # AE is available. The best checkpoint was saved on disk; we restore
        # its EMA weights into the EMA module before sampling so the dump
        # reflects the actual best snapshot, not the final-step weights.
        if ae_engine is not None and latent_shape is not None and best_val_step >= 0:
            try:
                best_path = run_dir / "checkpoints" / "best.pt"
                if best_path.exists():
                    state = torch.load(best_path, map_location=device, weights_only=False)
                    ema_state = state.get("ema_state_dict", {})
                    if ema_state:
                        # Apply the best-EMA trainable weights onto the EMA module
                        # before sampling.
                        sd = ema.module.state_dict()
                        for k, v in ema_state.items():
                            if k in sd:
                                sd[k].copy_(v.to(sd[k].device))
                        ema.module.load_state_dict(sd)
                self._dump_samples(
                    wrapper=ema.module,
                    ae=ae_engine,
                    scheduler=scheduler,
                    scale_factor=scale_factor,
                    latent_shape=latent_shape,
                    device=device,
                    autocast_dtype=autocast_dtype,
                    run_dir=run_dir,
                    step=best_val_step,
                    label="best",
                )
            except Exception as e:  # pragma: no cover
                logger.warning("best-epoch sample dump skipped: %s", e)

        final_test_metrics: dict[str, Any] = {}
        if test_ds is not None and len(test_ds) > 0:
            final_test_metrics = self._compute_final_test_metrics(
                ema,
                test_ds,
                scheduler,
                scale_factor,
                autocast_dtype,
                device,
                run_dir,
            )

        wall = time.time() - start_t
        manifest = {
            "run_name": cfg.run_name,
            "status": "success",
            "wall_time_sec": wall,
            "final_step": step,
            "fm_checkpoint": str(cfg.fm_checkpoint),
            "scale_factor": scale_factor,
            "ckpt_meta": ckpt_meta,
            "n_train_samples": len(train_ds),
            "n_val_samples": len(val_ds),
            "n_test_samples": len(test_ds) if test_ds is not None else 0,
            "modalities": list(cfg.modalities),
            "splits": {
                "kfold": int(cfg.kfold),
                "fold": int(cfg.fold),
                "train": train_path,
                "val": val_path,
                "test": test_path,
            },
            "trainable_params": int(n_trainable),
            "n_lora_adapters": len(adapters),
            "best_val_loss": float(best_val_loss) if best_val_loss < float("inf") else None,
            "best_val_step": int(best_val_step) if best_val_step >= 0 else None,
            "final_test_metrics": final_test_metrics,
            "finished_at": datetime.now(UTC).isoformat(),
        }
        with open(run_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("done (%.1fs). manifest at %s", wall, run_dir / "manifest.json")
        return run_dir

    # ------------------------------------------------------------------
    # Sub-routines
    # ------------------------------------------------------------------

    def _resolve_split_paths(self) -> tuple[str, str, str]:
        """Derive (train, val, test) split paths from kfold + fold knobs.

        Explicit ``train_split`` / ``val_split`` / ``test_split`` overrides on
        the config win; otherwise paths default to
        ``("kfold/k{k}/fold_{f}/train", "kfold/k{k}/fold_{f}/val",
        "kfold/k{k}/test")``.
        """
        cfg = self.config
        base = f"kfold/k{cfg.kfold}/fold_{cfg.fold}"
        train = cfg.train_split or f"{base}/train"
        val = cfg.val_split or f"{base}/val"
        test = cfg.test_split if cfg.test_split is not None else f"kfold/k{cfg.kfold}/test"
        return train, val, test

    def _build_datasets(
        self,
    ) -> tuple[
        JointLatentVolumeDataset,
        JointLatentVolumeDataset,
        JointLatentVolumeDataset | None,
    ]:
        cfg = self.config
        train_path, val_path, test_path = self._resolve_split_paths()
        logger.info(
            "split paths: train=%r val=%r test=%r",
            train_path,
            val_path,
            test_path,
        )
        common = dict(
            latent_h5=cfg.latent_h5,
            features_h5_dir=cfg.features_h5_dir,
            modalities=cfg.modalities,
        )
        train_ds = JointLatentVolumeDataset(DatasetConfig(**common, split=train_path))
        val_ds = JointLatentVolumeDataset(DatasetConfig(**common, split=val_path))
        test_ds: JointLatentVolumeDataset | None = None
        if test_path:
            try:
                test_ds = JointLatentVolumeDataset(DatasetConfig(**common, split=test_path))
            except KeyError as e:
                logger.warning("test split %r not found: %s", test_path, e)
                test_ds = None
        return train_ds, val_ds, test_ds

    def _init_run_dir(self) -> Path:
        cfg = self.config
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = cfg.output_dir / f"{cfg.run_name}_{ts}"
        for sub in ("checkpoints", "logs", "samples", "models"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)

        with open(run_dir / "config.snapshot.yaml", "w") as f:
            yaml.safe_dump(_config_to_jsonable(cfg), f, sort_keys=False)
        with open(run_dir / "arch.json", "w") as f:
            f.write(serialize_arch(MAISI_MR_FM_ARCH))
        with open(run_dir / "code_state.json", "w") as f:
            json.dump(_collect_code_state(), f, indent=2)
        try:
            env_lines = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout
        except Exception:
            env_lines = "<pip freeze failed>"
        with open(run_dir / "env.txt", "w") as f:
            f.write(env_lines)
        return run_dir

    def _open_log_files(self, run_dir: Path) -> dict[str, Path]:
        train_csv = run_dir / "logs" / "train_metrics.csv"
        val_csv = run_dir / "logs" / "val_metrics.csv"
        if not train_csv.exists():
            self._write_csv_header(
                train_csv, ["step", "loss", "lr", "grad_norm", "vram_gb", "sec_per_step"]
            )
        if not val_csv.exists():
            self._write_csv_header(
                val_csv,
                ["step", "val_loss", "latent_R2_proxy", "latent_spearman_proxy", "n_batches"],
            )
        return {"train_csv": train_csv, "val_csv": val_csv}

    def _assert_identity_at_init(
        self,
        wrapper: VolumeConditionalUNet,
        ref_state: dict[str, torch.Tensor],
        device: torch.device,
    ) -> None:
        seen = 0
        for n, p in wrapper.unet.named_parameters():
            if "lora_" in n:
                continue
            ref_name = n.replace(".base_layer.", ".")
            if ref_name not in ref_state:
                continue
            if not torch.allclose(p.detach().cpu(), ref_state[ref_name], atol=0):
                raise RuntimeError(f"weight {n!r} drifted from pretrained reference at init")
            seen += 1
        with torch.no_grad():
            v_out = wrapper.conditioner(torch.zeros(2, device=device))
        if v_out.abs().max().item() > 1e-7:
            raise RuntimeError(
                f"VolumeConditioner is not zero at init "
                f"(max |out| = {v_out.abs().max().item():.3e})"
            )
        logger.info(
            "identity-at-init OK (verified %d base params; conditioner output ≡ 0)",
            seen,
        )

    def _validate(
        self,
        wrapper: VolumeConditionalUNet,
        val_loader: DataLoader,
        scheduler: Any,
        scale_factor: float,
        autocast_dtype: torch.dtype,
        device: torch.device,
        val_csv: Path,
        step: int,
        loss_fn: nn.Module,
        *,
        max_batches: int,
    ) -> float | None:
        wrapper.eval()
        losses: list[float] = []
        z_means: list[float] = []
        log_vs: list[float] = []

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= max_batches:
                    break
                z = batch["z"].to(device) * scale_factor
                log_v = batch["log_v"].to(device)
                class_labels = batch["modality_idx"].to(device)
                spacing = batch["spacing"].to(device=device, dtype=autocast_dtype)
                noise = torch.randn_like(z)
                timesteps = scheduler.sample_timesteps(z)
                noisy = scheduler.add_noise(z, noise, timesteps)
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=autocast_dtype is not torch.float32,
                ):
                    pred = wrapper(
                        z_t=noisy,
                        timesteps=timesteps,
                        class_labels=class_labels,
                        spacing_tensor=spacing,
                        log_v=log_v,
                        use_uncond=None,
                    )
                losses.append(loss_fn(pred.float(), z.float()).item())
                # Latent-mean proxy: collect all (z_mean, log_v) pairs across
                # the val pass and compute R²/Spearman once at the end. With
                # batch=1, per-batch torchmetrics updates can hit numerical
                # edge cases that yield NaN even when the cumulative arrays
                # are well-defined; this avoids that.
                z_means.extend(z.float().mean(dim=(1, 2, 3, 4)).cpu().numpy().tolist())
                log_vs.extend(log_v.float().cpu().numpy().tolist())

        n = len(losses)
        val_loss = float(np.mean(losses)) if losses else float("nan")
        if len(z_means) >= 2:
            preds_t = torch.tensor(z_means, dtype=torch.float64)
            targets_t = torch.tensor(log_vs, dtype=torch.float64)
            try:
                r2 = float(R2Score()(preds_t, targets_t).item())
            except Exception:
                r2 = float("nan")
            try:
                sp = float(SpearmanCorrCoef()(preds_t, targets_t).item())
            except Exception:
                sp = float("nan")
        else:
            r2 = float("nan")
            sp = float("nan")

        self._append_csv(
            val_csv,
            {
                "step": step,
                "val_loss": f"{val_loss:.6f}",
                "latent_R2_proxy": f"{r2:.4f}",
                "latent_spearman_proxy": f"{sp:.4f}",
                "n_batches": n,
            },
        )
        logger.info(
            "[val] step %d | val_loss %.4f | R2 %.3f | Spearman %.3f | n=%d",
            step,
            val_loss,
            r2,
            sp,
            n,
        )
        wrapper.train()
        return val_loss if not (val_loss != val_loss) else None  # NaN guard

    def _save_checkpoint(
        self,
        run_dir: Path,
        wrapper: VolumeConditionalUNet,
        ema: AveragedModel,
        step: int,
        final: bool = False,
    ) -> None:
        ckpt_dir = run_dir / "checkpoints"
        # EMA holds the averaged copy on CPU; persist its trainable params only.
        ema_trainable = {
            k: v.detach().cpu()
            for k, v in ema.module.state_dict().items()
            if "lora_" in k or k.startswith("conditioner.")
        }
        payload = {
            "step": step,
            "trainable_state_dict": wrapper.trainable_state_dict(),
            "ema_state_dict": ema_trainable,
            "ema_n_averaged": int(ema.n_averaged.item()),
        }
        torch.save(payload, ckpt_dir / f"lora_step_{step}.pt")
        torch.save(payload, ckpt_dir / "last.pt")
        if final:
            logger.info("final checkpoint saved at step %d", step)

    def _save_best_checkpoint(
        self,
        run_dir: Path,
        wrapper: VolumeConditionalUNet,
        ema: AveragedModel,
        step: int,
        val_loss: float,
    ) -> None:
        """Persist the running best (lowest val_loss) checkpoint."""
        ckpt_dir = run_dir / "checkpoints"
        ema_trainable = {
            k: v.detach().cpu()
            for k, v in ema.module.state_dict().items()
            if "lora_" in k or k.startswith("conditioner.")
        }
        payload = {
            "step": step,
            "best_val_loss": float(val_loss),
            "trainable_state_dict": wrapper.trainable_state_dict(),
            "ema_state_dict": ema_trainable,
            "ema_n_averaged": int(ema.n_averaged.item()),
        }
        torch.save(payload, ckpt_dir / "best.pt")
        with open(run_dir / "best.json", "w") as f:
            json.dump({"step": int(step), "val_loss": float(val_loss)}, f, indent=2)
        logger.info("[best] new best val_loss=%.4f at step %d (saved best.pt)", val_loss, step)

    def _dump_samples(
        self,
        *,
        wrapper: VolumeConditionalUNet,
        ae: Any,
        scheduler: Any,
        scale_factor: float,
        latent_shape: tuple[int, int, int, int],
        device: torch.device,
        autocast_dtype: torch.dtype,
        run_dir: Path,
        step: int,
        label: str = "train",
    ) -> None:
        """Run a small ODE pass at fixed log_v anchors and decode to NIfTI.

        Drops one HDF5 + one NIfTI per (anchor, log_v) into
        ``samples/{label}/step_{step}/``. Used to monitor how the conditioning
        signal evolves during training. Failures are logged but never abort
        the run — the visual sanity check is non-critical.
        """
        try:
            cfg = self.config
            log_v_grid = list(cfg.sampling.log_v_grid)
            n_anchors = max(1, int(cfg.sampling.n_anchors))
            sampler_cfg = SamplerConfig(
                n_ode_steps=int(cfg.sampling.n_ode_steps),
                cfg_scale=float(cfg.sampling.cfg_scale),
                autocast_dtype=autocast_dtype,
            )
            modality_idx = self._first_modality_idx(device)
            spacing = (
                torch.tensor(
                    self._spacing_for_dump(),
                    dtype=torch.float32,
                    device=device,
                )
                * 100.0
            )

            wrapper.eval()
            samples_dir = run_dir / "samples" / label / f"step_{step}"
            samples_dir.mkdir(parents=True, exist_ok=True)
            for anchor in range(n_anchors):
                seed = int(cfg.train.seed) + 7919 * step + anchor
                log_v = torch.tensor(log_v_grid, dtype=torch.float32, device=device)
                cls = modality_idx.expand(log_v.numel())
                sp = spacing.unsqueeze(0).expand(log_v.numel(), -1)
                z = sample_latents(
                    wrapper,
                    log_v=log_v,
                    class_labels=cls,
                    spacing_tensor=sp,
                    latent_shape=latent_shape,
                    scheduler=scheduler,
                    cfg=sampler_cfg,
                    device=device,
                    seed=seed,
                )
                # Inverse the train-time scale_factor so the decoder sees the
                # same magnitude it was trained on.
                z_for_decode = z / max(scale_factor, 1e-9)
                # Persist latents.
                with h5py.File(samples_dir / f"anchor_{anchor:02d}.h5", "w") as f:
                    f.create_dataset(
                        "latents",
                        data=z.detach().cpu().numpy(),
                        compression="gzip",
                        compression_opts=4,
                    )
                    f.create_dataset("log_v", data=log_v.detach().cpu().numpy())
                    f.attrs["step"] = int(step)
                    f.attrs["anchor"] = int(anchor)
                    f.attrs["seed"] = int(seed)
                    f.attrs["cfg_scale"] = float(sampler_cfg.cfg_scale)
                    f.attrs["n_ode_steps"] = int(sampler_cfg.n_ode_steps)
                # Decode + save NIfTI per log_v slot.
                if ae is not None:
                    try:
                        import nibabel as nib

                        with torch.no_grad():
                            decoded = (
                                ae.decode(z_for_decode).float().cpu().numpy()
                            )  # (B, 1, H, W, D)
                        affine = np.eye(4)
                        for j, lv in enumerate(log_v_grid):
                            vol = decoded[j, 0]
                            nib.save(
                                nib.Nifti1Image(vol.astype(np.float32), affine),
                                str(samples_dir / f"anchor_{anchor:02d}_logv_{lv:+.2f}.nii.gz"),
                            )
                    except Exception as e:  # pragma: no cover
                        logger.warning("decode-to-nifti failed at step %d: %s", step, e)
            wrapper.train()
            logger.info(
                "[samples] step %d | %d anchors × %d log_v values dumped to %s",
                step,
                n_anchors,
                len(log_v_grid),
                samples_dir,
            )
        except Exception as e:  # pragma: no cover
            logger.warning("sample dump skipped at step %d: %s", step, e)

    def _first_modality_idx(self, device: torch.device) -> torch.Tensor:
        """MAISI class index for the first active modality (e.g. 17 for T1c)."""
        from menflow.finetuning.unet_wrapper import MAISI_MODALITY_INDEX

        modality = self.config.modalities[0]
        return torch.tensor(int(MAISI_MODALITY_INDEX[modality]), dtype=torch.long, device=device)

    def _spacing_for_dump(self) -> tuple[float, float, float]:
        """Read the cohort spacing from the latent H5 attrs."""
        try:
            with h5py.File(self.config.latent_h5, "r") as f:
                sp = f.attrs.get("spacing_mm", [1.0, 1.0, 1.0])
                return tuple(float(x) for x in sp)
        except Exception:  # pragma: no cover
            return (1.0, 1.0, 1.0)

    def _save_safetensors(
        self,
        run_dir: Path,
        wrapper: VolumeConditionalUNet,
        rank: int,
    ) -> None:
        out = run_dir / "models" / f"e3_1_lora_r{rank}.safetensors"
        # safetensors does not allow non-tensor metadata; keep tensors only.
        td = wrapper.trainable_state_dict()
        # Some keys collide with safetensors' restrictions on `.` in names if
        # they end with metadata-like suffixes; prefix uniformly.
        flat = {k.replace("/", "."): v.contiguous() for k, v in td.items()}
        safetensors_save_file(flat, str(out), metadata={"e3_lora_rank": str(rank)})
        logger.info("wrote LoRA safetensors: %s (%d tensors)", out, len(flat))

    def _write_metrics_parquet(
        self,
        run_dir: Path,
        train_csv: Path,
        val_csv: Path,
    ) -> None:
        try:
            t = pd.read_csv(train_csv)
            v = pd.read_csv(val_csv) if val_csv.exists() else pd.DataFrame()
            (run_dir / "logs" / "metrics.parquet").parent.mkdir(parents=True, exist_ok=True)
            t.to_parquet(run_dir / "logs" / "train_metrics.parquet", compression="snappy")
            if not v.empty:
                v.to_parquet(run_dir / "logs" / "val_metrics.parquet", compression="snappy")
            # Joined snapshot: left-merge val rows onto train rows by step.
            if not v.empty:
                joined = t.merge(v, on="step", how="left")
                joined.to_parquet(run_dir / "logs" / "metrics.parquet", compression="snappy")
            else:
                t.to_parquet(run_dir / "logs" / "metrics.parquet", compression="snappy")
            logger.info("wrote metrics.parquet")
        except Exception as e:  # pragma: no cover
            logger.warning("could not write metrics parquet: %s", e)

    def _write_splits_artifact(
        self,
        run_dir: Path,
        train_ds: JointLatentVolumeDataset,
        val_ds: JointLatentVolumeDataset,
        test_ds: JointLatentVolumeDataset | None,
    ) -> None:
        train_path, val_path, test_path = self._resolve_split_paths()
        payload: dict[str, Any] = {
            "kfold": int(self.config.kfold),
            "fold": int(self.config.fold),
            "splits": {
                train_path: _list_scan_ids(train_ds),
                val_path: _list_scan_ids(val_ds),
            },
        }
        if test_ds is not None:
            payload["splits"][test_path] = _list_scan_ids(test_ds)
        with open(run_dir / "splits.json", "w") as f:
            json.dump(payload, f, indent=2)

    def _write_data_stats_artifact(
        self,
        run_dir: Path,
        train_ds: JointLatentVolumeDataset,
        val_ds: JointLatentVolumeDataset,
        test_ds: JointLatentVolumeDataset | None,
    ) -> None:
        train_path, val_path, test_path = self._resolve_split_paths()
        out: dict[str, Any] = {"by_split": {}}
        for name, ds in [
            (train_path, train_ds),
            (val_path, val_ds),
            (test_path, test_ds),
        ]:
            if not name or ds is None:
                continue
            log_vs = _list_log_v(ds)
            if not log_vs:
                continue
            arr = np.asarray(log_vs)
            out["by_split"][name] = {
                "n": int(arr.size),
                "log_v_min": float(arr.min()),
                "log_v_p05": float(np.quantile(arr, 0.05)),
                "log_v_p50": float(np.quantile(arr, 0.50)),
                "log_v_p95": float(np.quantile(arr, 0.95)),
                "log_v_max": float(arr.max()),
            }
        with open(run_dir / "data_stats.json", "w") as f:
            json.dump(out, f, indent=2)

    def _compute_final_test_metrics(
        self,
        ema: AveragedModel,
        test_ds: JointLatentVolumeDataset,
        scheduler: Any,
        scale_factor: float,
        autocast_dtype: torch.dtype,
        device: torch.device,
        run_dir: Path,
    ) -> dict[str, Any]:
        """Latent-R² proxy on e3_test using the EMA wrapper.

        Computes the regression of ``log_v_pred`` (predicted from the noiseless
        latent's flat mean) against ``log_v_true`` over the entire test set
        with patient-level bootstrap CIs. The soft-volume R² (image-space) is
        an extension point left to the Picasso run; the function returns its
        slot as ``None`` here so the JSON layout is stable.
        """
        wrapper = ema.module
        wrapper.eval()
        wrapper.to(device)
        loader = DataLoader(
            test_ds,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_samples,
        )
        log_v_pred: list[float] = []
        log_v_true: list[float] = []
        patient_ids: list[str] = []
        with torch.no_grad():
            for batch in loader:
                z = batch["z"].to(device) * scale_factor
                log_v = batch["log_v"].to(device)
                # Noiseless latent flat mean as a cheap stand-in until the E2.1
                # regressor is wired here. Predicted log_v ≡ z_mean (rescaled).
                z_mean = z.float().mean(dim=(1, 2, 3, 4))
                log_v_pred.extend(z_mean.cpu().numpy().tolist())
                log_v_true.extend(log_v.cpu().numpy().tolist())
                patient_ids.extend(batch["scan_id"])

        per_bin = per_logv_bin_metrics(
            log_v_pred=np.asarray(log_v_pred),
            log_v_true=np.asarray(log_v_true),
            n_bins=5,
        )
        cal = fit_calibration(
            log_v_pred=np.asarray(log_v_pred),
            log_v_true=np.asarray(log_v_true),
            patient_ids=np.asarray(patient_ids),
            n_bootstrap=1000,
            seed=self.config.train.seed,
        )
        out = {
            "split": self.config.test_split,
            "n_test_samples": len(test_ds),
            "metric_kind": "latent_proxy_via_z_mean",
            "calibration": cal.to_jsonable(),
            "per_logv_bin": per_bin,
            "soft_volume_r2": None,
        }
        with open(run_dir / "final_test_metrics.json", "w") as f:
            json.dump(out, f, indent=2)
        logger.info(
            "[test] n=%d | slope %.3f | R² %.3f | Spearman_med %.3f | %%≥0.9 %.2f",
            cal.n,
            cal.slope,
            cal.r2,
            cal.spearman_median,
            cal.pct_spearman_ge_0p9,
        )
        return out

    @staticmethod
    def _enable_activation_checkpointing(unet: nn.Module) -> None:
        try:
            from torch.utils.checkpoint import checkpoint_sequential  # noqa: F401

            for m in unet.modules():
                if hasattr(m, "use_checkpoint"):
                    m.use_checkpoint = True
            logger.info("activation checkpointing flag set on supported submodules")
        except Exception as e:  # pragma: no cover
            logger.warning("could not enable activation checkpointing: %s", e)

    @staticmethod
    def _write_csv_header(path: Path, columns: list[str]) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

    @staticmethod
    def _append_csv(path: Path, row: dict) -> None:
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)


# ============================================================================
# Helpers
# ============================================================================


def _list_scan_ids(ds: JointLatentVolumeDataset) -> list[str]:
    if ds is None:
        return []
    rows = ds._scan_rows  # type: ignore[attr-defined]
    return [str(ds._scan_ids[r]) for r in rows]  # type: ignore[attr-defined]


def _list_log_v(ds: JointLatentVolumeDataset) -> list[float]:
    if ds is None:
        return []
    rows = ds._scan_rows  # type: ignore[attr-defined]
    return [float(ds._log_v[r]) for r in rows]  # type: ignore[attr-defined]


def _config_to_jsonable(cfg: FinetuneFMVolumeRoutineConfig) -> dict:
    d = asdict(cfg)
    return _to_jsonable(d)


def _to_jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    return o


def _collect_code_state() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    try:
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
    except Exception:
        dirty = False
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "cwd": os.getcwd(),
    }
