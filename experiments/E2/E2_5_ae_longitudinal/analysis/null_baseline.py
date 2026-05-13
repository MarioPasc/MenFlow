"""Null baseline — ρ_lin distribution for matched-volume cross-patient triples.

Per spec §4.4: draw ``z_A^{(1)}, z_B^{(2)}, z_C^{(3)}`` from three different
patients at random timepoints, matched on volume order ``V_A < V_B < V_C``.
Compute ρ_lin on each triple. A meaningful Δρ = mean(ρ_lin_null) − mean(ρ_lin)
indicates that same-patient longitudinal triples are more linear than random.

Implementation: pool all (scan, modality) pairs with finite ``log_volume_cm3``
and ``has_segmentation==True``; sample three distinct patients; sample one
timepoint per patient; reject if the volume order is not strictly increasing.
Same-patient triples are excluded by construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import h5py
import numpy as np

from experiments.E2.E2_5_ae_longitudinal.analysis.metrics import (
    DegenerateTripletError,
    linearity_residual,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NullBaselineResult:
    rho_lin: np.ndarray  # shape (n_drawn,)
    n_drawn: int
    n_rejected_volume_order: int


def compute_null_baseline(
    *,
    latents_h5_path: str,
    unified_h5_path: str,
    n_triples: int,
    seed: int,
    use_joint: bool = True,
    max_attempts: int | None = None,
) -> NullBaselineResult:
    """Sample cross-patient triples and compute ρ_lin for each.

    Notes
    -----
    For tractability with full-resolution latents (~1 MB per scan), the
    function loads all latents into memory once. For MenGrowth (N=179)
    this is < 200 MB.
    """
    rng = np.random.default_rng(seed)

    with h5py.File(unified_h5_path, "r") as f:
        log_vol = f["features/log_volume_cm3"][:].astype(np.float64)
        has_seg = f["has_segmentation"][:].astype(bool)
        patient_offsets = f["longitudinal/patient_offsets"][:].astype(int)
        patient_list = [
            s.decode() if isinstance(s, bytes) else s
            for s in f["longitudinal/patient_list"][:]
        ]

    valid = has_seg & np.isfinite(log_vol)
    scan_to_patient = np.zeros(len(log_vol), dtype=int)
    for pi in range(len(patient_list)):
        scan_to_patient[patient_offsets[pi] : patient_offsets[pi + 1]] = pi

    # Per-patient pool of valid scan indices.
    per_patient_valid: list[np.ndarray] = []
    for pi in range(len(patient_list)):
        rows = np.arange(int(patient_offsets[pi]), int(patient_offsets[pi + 1]))
        rows = rows[valid[rows]]
        per_patient_valid.append(rows)

    eligible_patients = [pi for pi, rs in enumerate(per_patient_valid) if rs.size > 0]
    if len(eligible_patients) < 3:
        raise RuntimeError(
            f"Not enough eligible patients for null baseline: {len(eligible_patients)}"
        )

    # Load all latents up front; cheap for MenGrowth.
    with h5py.File(latents_h5_path, "r") as lat:
        latents = lat["latents"][:].astype(np.float32)  # (N, M, C, H', W', D')

    rho_lin = np.empty(n_triples, dtype=np.float64)
    n_drawn = 0
    n_rejected = 0
    if max_attempts is None:
        max_attempts = 20 * n_triples
    attempts = 0

    while n_drawn < n_triples and attempts < max_attempts:
        attempts += 1
        ps = rng.choice(eligible_patients, size=3, replace=False)
        rA = int(rng.choice(per_patient_valid[ps[0]]))
        rB = int(rng.choice(per_patient_valid[ps[1]]))
        rC = int(rng.choice(per_patient_valid[ps[2]]))
        vols = np.array([log_vol[rA], log_vol[rB], log_vol[rC]])
        # Sort by volume to enforce V_A < V_B < V_C.
        order = np.argsort(vols)
        if vols[order[0]] >= vols[order[1]] or vols[order[1]] >= vols[order[2]]:
            n_rejected += 1
            continue
        rA, rB, rC = (int(np.array([rA, rB, rC])[i]) for i in order)
        za = latents[rA]
        zb = latents[rB]
        zc = latents[rC]
        try:
            if use_joint:
                res = linearity_residual(za, zb, zc)
            else:
                # Per-modality average ρ_lin.
                per_m = []
                for mi in range(za.shape[0]):
                    per_m.append(linearity_residual(za[mi], zb[mi], zc[mi]).rho_lin)
                from types import SimpleNamespace

                res = SimpleNamespace(rho_lin=float(np.mean(per_m)))  # type: ignore[assignment]
        except DegenerateTripletError:
            n_rejected += 1
            continue
        rho_lin[n_drawn] = float(res.rho_lin)
        n_drawn += 1

    if n_drawn < n_triples:
        logger.warning(
            "Null baseline drew only %d / %d triples (attempts=%d, rejected=%d)",
            n_drawn,
            n_triples,
            attempts,
            n_rejected,
        )
        rho_lin = rho_lin[:n_drawn]

    return NullBaselineResult(
        rho_lin=rho_lin,
        n_drawn=n_drawn,
        n_rejected_volume_order=n_rejected,
    )
