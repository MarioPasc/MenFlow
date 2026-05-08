---
name: explore
description: Deep codebase exploration in isolated context
---

# MenGrowth Codebase Exploration

Thoroughly explore the MenGrowth-Model codebase to answer the query.

## Framework: 3-Stage Complexity Ladder

- **Stage 1 (Primary):** Volume-only growth prediction (ScalarGP, LME, HGP)
- **Stage 2 (Secondary):** Latent severity model (NLME, quantile transform)
- **Stage 3 (Tertiary):** Representation learning (LoRA → SDP → PCA → GP+ARD)

## Key Locations
- `src/growth/shared/` — **Cross-stage infrastructure**: GrowthModel ABC, LOPOEvaluator, metrics, bootstrap, covariates, trajectory I/O
- `src/growth/stages/stage1_volumetric/` — Stage 1: Gompertz mean function
- `src/growth/stages/stage2_severity/` — Stage 2: SeverityModel, QuantileTransform, growth functions
- `src/growth/stages/stage3_latent/` — Stage 3: facade for encoder, SDP, projection
- `src/growth/models/growth/` — Stage-agnostic growth models (scalar_gp, lme_model, hgp_model)
- `src/growth/models/encoder/` — SwinUNETR loader, LoRA adapter, feature extractor (Stage 3)
- `src/growth/models/projection/` — SDP network, partition, semantic heads (Stage 3)
- `src/growth/data/` — BraTS-MEN/GLI loaders, transforms, semantic features
- `src/growth/losses/` — Segmentation, SDP composite, VICReg, dCor
- `src/growth/evaluation/` — GP probes, latent quality, variance_decomposition
- `experiments/stage1_volumetric/` — Stage 1 experiment pipeline
- `experiments/stage2_severity/` — Stage 2 experiment pipeline
- `experiments/stage3_latent/{lora,sdp,domain_gap}/` — Stage 3 experiments
- `experiments/variance_decomposition/` — Cross-stage ΔR² analysis
- `docs/stages/` — Self-contained stage specifications
- `docs/RESEACH_PLAN.md` — Literature synthesis
- `docs/PLAN_OF_ACTION_v1.md` — Implementation plan

## Critical Conventions
- Channel order: `["t2f", "t1c", "t1n", "t2w"]` (FLAIR, T1ce, T1, T2)
- ROI: 128³ (training), 192³ (feature extraction)
- Segmentation: TC/WT/ET (3-ch sigmoid, hierarchical overlapping)
- N=31-58 patients, max 2-3 predictor params at N=31

$ARGUMENTS
