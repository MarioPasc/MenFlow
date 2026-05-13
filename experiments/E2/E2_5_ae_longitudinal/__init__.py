"""E2.5 — MAISI Autoencoder Longitudinal Traversal Diagnostic.

Pre-commitment diagnostic that decides between R1/R2/R3 regimes for the
roadmap's Gate G1 (anchor-propagation Mechanism 1). Tests whether the MAISI-v2
latent space supports approximately linear longitudinal traversal on MenGrowth.
"""

from experiments.E2.E2_5_ae_longitudinal.config import AELongitudinalConfig

__all__ = ["AELongitudinalConfig"]
