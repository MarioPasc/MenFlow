# E2.5 — MAISI AE Longitudinal Traversal Diagnostic

Pre-commitment diagnostic that decides between regimes **R1 / R2 / R3** for the
roadmap's Gate G1 (anchor-propagation Mechanism 1). Spec:
`docs/E2/E2_5_ae_longitudinal_diagnostic.md`.

## Run

```
menflow-e2-5 experiments/E2/E2_5_ae_longitudinal/configs/local_3060_smoke.yaml
```

or equivalently

```
python -m experiments.E2.E2_5_ae_longitudinal.cli <yaml>
```

## Configs

- `default.yaml` — Picasso baseline, all 4 modalities, full cohort, 500 null triples.
- `local_3060_smoke.yaml` — 3 patients, t1c only, no null baseline. Verifies the pipeline + round-trip-SSIM gate.
- `local_3060_full.yaml` — full local run on RTX 4060.

## Outputs (`output_dir`)

| File | Description |
|---|---|
| `per_patient.csv` | One row per retained patient triple (ρ_lin, β_vol, β_residual, SSIMs, ...). `dice_at_beta_vol` is NaN (deferred). |
| `aggregate.json` | Bootstrap CIs for every metric, cohort-construction log, decision JSON inlined. |
| `decision.json` | `regime`, `downstream_action`, `rationale`, `metrics`, `caveats`. |
| `figures/rho_lin_distribution.png` | Same-patient ρ_lin histogram overlaid with the null. |
| `figures/interpolation_quality_scatter.png` | β_residual vs SSIMs. |
| `figures/anatomy_ssim_curves.png` | Per-patient + mean ± SE curves. |
| `figures/qualitative_panel.png` | Decoded β grid for representative patients. |
| `E2_5_local_report.md` | Short text report with regime + caveats. |

## Caveats

- The local build defers the Dice@β_vol gate (no Python-callable nnU-Net wired in). Re-run on Picasso with the Docker BraTS segmenter to close the gate.
- MenGrowth's effective cohort after C1–C4 is typically below the spec's `min_effective_cohort=20`; the engine auto-relaxes C4 (0.3 → 0.2 → 0.1) and the CSV/JSON record the relaxation.
- The full-resolution decode (240×240×160 fp16) must fit on the 4060's 8 GB. If OOM, lower `decode_dtype` to `bfloat16` (4060 supports it), or restrict `modalities` to a single channel.
