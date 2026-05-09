"""E1 — MAISI-v2 reconstruction sufficiency for meningioma MRI.

Implements the protocol in ``docs/E1_MAISI_Sufficiency_Protocol.md``: encode a
cohort with the frozen MAISI-v2 VAE-GAN, decode back to image space, and
quantify how reconstruction error correlates with tumor volume on global,
tumor-region, and mask-preservation axes.

The §6.3 mask-preservation block (Dice / HD95 between expert and resegmented
mask) requires a separate BraTS-MEN segmentor and is currently a TODO; this
package implements §6.1 (global metrics), §6.2 (tumor-region metrics), §5
volume stratification, and §7 statistical analysis (log–log Huber regression
+ piecewise breakpoint).
"""
