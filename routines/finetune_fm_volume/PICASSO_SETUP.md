# Picasso submission — `routines/finetune_fm_volume`

End-to-end setup for running the volume-conditional MAISI-v2 FM finetune on
Picasso A100 nodes. The launcher fans out **one independent SLURM job per
fold**, so picking `kfold: 5` yields five concurrent jobs that share a held-out
test set but train on disjoint (train, val) folds.

## 1. Files to ship to Picasso

Total ≈ 4 GB. Copy with rsync (one-shot or incremental):

```bash
# from your local workstation
rsync -avhP --partial \
  /media/mpascual/MeningD2/MAISI_VAEGAN_LATENTS/BRATS_MEN/brats_men_maisi_latents.h5 \
  /media/mpascual/MeningD2/MAISI_VAEGAN_LATENTS/BRATS_MEN/per_modality_features/ \
  /media/mpascual/MeningD2/MENINGIOMAS/BRATS_MEN/h5/brats_men.h5 \
  /media/mpascual/Sandisk2TB/research/menflow/checkpoints/NV-Generate-MR/models/diff_unet_3d_rflow-mr.pt \
  /media/mpascual/Sandisk2TB/research/menflow/checkpoints/NV-Generate-MR/models/autoencoder_v2.pt \
  picasso:/mnt/home/users/tic_163_uma/mpascual/fscratch/staging/
```

Then re-arrange to the canonical layout the YAML expects:

```
/mnt/home/users/tic_163_uma/mpascual/fscratch/
├── checkpoints/NV-Generate-MR/models/
│   ├── diff_unet_3d_rflow-mr.pt        # FM (180.5 M params, ~700 MB)
│   └── autoencoder_v2.pt                # MAISI VAE (~80 MB, used for sample decoding)
├── datasets/MAISI_VAEGAN_LATENTS/BRATS_MEN/
│   ├── brats_men_maisi_latents.h5       # latents [N, 4 mod, 4 ch, 60, 60, 40] (~3 GB) — REQUIRED
│   └── per_modality_features/
│       ├── brats_men_features_t1c.h5    # log_volume per scan — REQUIRED
│       ├── brats_men_features_t1n.h5
│       ├── brats_men_features_t2f.h5
│       └── brats_men_features_t2w.h5
├── datasets/MENINGIOMAS/BRATS_MEN/h5/
│   └── brats_men.h5                     # source unified H5 — OPTIONAL (used only for provenance attrs)
└── runs/finetune_fm_volume/             # outputs land here
```

**Required for training** (the engine refuses to start without these):

| File | Why |
|---|---|
| `brats_men_maisi_latents.h5` | The cached latents the FM operates on. Already migrated → `splits/kfold/k{1,3,5,10}/...`. |
| `per_modality_features/brats_men_features_t1c.h5` | log_volume per scan, used as the conditioning target. |
| `diff_unet_3d_rflow-mr.pt` | MAISI rectified-flow U-Net. |

**Optional** (engine degrades gracefully if missing):

| File | Why |
|---|---|
| `autoencoder_v2.pt` | If present, the engine decodes sample latents to NIfTI at every `sample_interval` and at the best-val checkpoint, so you can eyeball the conditioning evolution. |
| `brats_men.h5` | Recorded in `manifest.json` for provenance and used by future analysis scripts that need the original NIfTI volumes. |

## 2. Singularity image

Recipe lives in `~/sif/menflow_ngc.def` (NGC PyTorch base, conda env layered on
top). Build once on the login node:

```bash
cd ~/sif
apptainer build menflow_ngc.sif menflow_ngc.def
```

The worker bind-mounts your local repo into the container at `/opt/MenFlow`
and runs `pip install --no-deps -e /opt/MenFlow` per job — no rebuild needed
when you push code changes; just re-submit.

## 3. Configure the run

Edit `routines/finetune_fm_volume/configs/picasso_a100_kfold.yaml`:

```yaml
fm_checkpoint: /mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/NV-Generate-MR/models/diff_unet_3d_rflow-mr.pt
ae_checkpoint: /mnt/home/users/.../models/autoencoder_v2.pt
latent_h5:     /mnt/home/users/.../brats_men_maisi_latents.h5
features_h5_dir: /mnt/home/users/.../per_modality_features
output_dir:    /mnt/home/users/.../runs/finetune_fm_volume

run_name: a100_k5
kfold: 5     # ← number of folds; equals number of jobs the launcher fires
fold:  0     # placeholder; the launcher overrides per job

train:
  batch_size: 4
  grad_accum: 4
  max_steps: 50000
  autocast_dtype: bfloat16
  val_interval: 500
  sample_interval: 2500
  save_interval: 5000

slurm:
  partition: dgx2q
  qos: dgx2q
  constraint: dgx
  gres: "gpu:1"
  cpus_per_task: 8
  mem: "64G"
  time: "48:00:00"
```

## 4. Launch the sweep

From the repo root on Picasso:

```bash
# Fire one sbatch per fold (kfold=5 → five jobs):
bash routines/finetune_fm_volume/slurm/launcher_finetune_fm_volume.sh \
    routines/finetune_fm_volume/configs/picasso_a100_kfold.yaml

# Dry-run (prints sbatch invocations without submitting):
bash routines/finetune_fm_volume/slurm/launcher_finetune_fm_volume.sh \
    routines/finetune_fm_volume/configs/picasso_a100_kfold.yaml --dry-run
```

The launcher prints one line per submitted job with the assigned jobid.
Each fold produces a distinct run directory under `output_dir/`:

```
runs/finetune_fm_volume/
├── a100_k5_fold_0_<TS>/
├── a100_k5_fold_1_<TS>/
├── a100_k5_fold_2_<TS>/
├── a100_k5_fold_3_<TS>/
└── a100_k5_fold_4_<TS>/
```

Re-running with `kfold: 1` falls back to the single 80/10/10 holdout (one
job, behaviour-identical to the local smoke).

## 5. What gets tracked per run

Each run directory ships:

| Path | Content |
|---|---|
| `config.snapshot.yaml` | Frozen, fully-resolved config (every default included). |
| `arch.json` | MAISI U-Net architecture JSON; lets a fresh checkout reproduce the model exactly. |
| `code_state.json` | Git commit + dirty flag + python/torch/cuda versions + cwd. |
| `env.txt` | `pip freeze` snapshot. |
| `splits.json` | scan_id list per active split (train/val/test). |
| `data_stats.json` | log_v quantile summary per split. |
| `logs/train_metrics.csv` | Per-step `loss, lr, grad_norm, vram_gb, sec_per_step` (streamed). |
| `logs/val_metrics.csv` | Per validation step `val_loss, R², Spearman, n_batches`. |
| `logs/metrics.parquet` | Final joined snapshot (pandas-friendly). |
| `checkpoints/lora_step_{N}.pt` | LoRA + conditioner + EMA state per `save_interval`. |
| `checkpoints/best.pt` + `best.json` | Lowest-val_loss snapshot, with the achieving step. |
| `models/e3_1_lora_r{R}.safetensors` | Spec-mandated final LoRA artifact. |
| `samples/train/step_{N}/` | Decoded NIfTI of `n_anchors × log_v_grid` ODE samples per `sample_interval`. Lets you watch the conditioning evolve. |
| `samples/best/step_{best}/` | Same, computed once at the end from `best.pt`. |
| `final_test_metrics.json` | Calibration regression on `e3_test`: slope, R², per-patient Spearman, 1000-resample bootstrap CIs. **Plus per-log_v-bin slope/R²/Spearman** so you can correlate calibration quality with bin density and check for volume-range overfitting. |
| `manifest.json` | Final summary: `best_val_loss`, `best_val_step`, wall time, split paths, all of the above by reference. |

## 6. Post-run aggregation (k-fold case)

After all K jobs finish:

```bash
# concatenate per-fold final_test_metrics.json into one CSV:
python - <<'PY'
import json, glob, pandas as pd
rows = []
for p in glob.glob("/mnt/home/users/.../runs/finetune_fm_volume/a100_k5_fold_*/final_test_metrics.json"):
    with open(p) as f: m = json.load(f)
    cal = m["calibration"]
    rows.append({
        "run": p.split("/")[-2],
        "n": cal["n"], "slope": cal["slope"], "r2": cal["r2"],
        "spearman_med": cal["spearman_median"],
        "best_val_loss": json.load(open(p.replace("final_test_metrics.json", "manifest.json")))["best_val_loss"],
    })
pd.DataFrame(rows).to_csv("k5_summary.csv", index=False)
print(pd.DataFrame(rows))
PY
```

Per-bin overfitting check is in `final_test_metrics.json["per_logv_bin"]["bins"]`
under each fold; merge across folds to spot volume-range bias.

## 7. Troubleshooting

- **OOM at step 1.** Drop `train.batch_size` to 2 and bump `grad_accum` to 8 to keep the effective batch at 16.
- **`splits/kfold/k5` missing.** Re-run the migration on Picasso once: `python -m menflow.data.migrate_e3_splits --source-h5 ... --latent-h5 ... --features-h5 ... --k-values 1 3 5 10`. Idempotent.
- **No internet from compute nodes.** The worker sets `WANDB_MODE=offline`; sync after the run with `wandb sync ~/execs/menflow/finetune_fm_volume/wandb/` if you wired W&B in.
- **AE not loading.** Engine logs `could not load AE for sampling (...); skipping decode dumps` and continues training — but you lose the periodic NIfTI montages. Latent-only sample dumps still go into `samples/.../anchor_*.h5`.
