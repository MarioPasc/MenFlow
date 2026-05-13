"""Population longitudinal-direction test (spec property P3).

Companion to :mod:`metrics` for the stronger diagnostic asked for in the
"find the longitudinal direction" follow-up: instead of testing whether each
patient's `z²` lies on the line `z¹ → z³`, fit a 1-D principal component over
the per-patient trajectory vectors `{z³ − z¹}_i` and ask:

* What fraction of the variance in `{z³ − z¹}` is captured by the leading
  component? A high value (≥ 0.8) means there *is* a shared longitudinal
  axis even if individual triples are curved.
* For each patient, what is the cosine alignment between `(z² − z¹)` and the
  population axis (sign-aware)? A high mean cosine means the within-patient
  trajectories use the same direction as the population axis, even when the
  global line `z¹ → z³` does not pass through `z²` itself.

The test runs in the mask-pooled `(M, C) = 16`-D feature space (cheap, exactly
the features the probe uses) so the leading PC is interpretable as the
"feature-space volume direction". The roadmap's Mechanism 1 (anchor
propagation) depends on this *population* direction more than on per-patient
linearity, so this diagnostic is closer to the load-bearing claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SubspaceResult:
    """Output of :func:`pca_longitudinal_direction`.

    Attributes
    ----------
    variance_explained
        Fraction of the variance in `{z³ − z¹}` captured by the leading
        PC. Range ``[1 / n_dim, 1]``.
    direction
        Unit vector of shape ``(D,)`` — the leading PC.
    singular_values
        All singular values of the centred ``(z³ − z¹)`` matrix (descending).
    cos_z2_minus_z1
        Per-patient sign-corrected cosine between ``(z² − z¹)`` and the
        leading PC. Shape ``(n_patients,)``.
    cos_z3_minus_z1
        Per-patient cosine between ``(z³ − z¹)`` and the leading PC. The
        sign of the PC is fixed by majority-vote on this array.
    """

    variance_explained: float
    direction: np.ndarray
    singular_values: np.ndarray
    cos_z2_minus_z1: np.ndarray
    cos_z3_minus_z1: np.ndarray


def pca_longitudinal_direction(
    pooled_triples: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> SubspaceResult:
    """Fit a 1-D PCA over `{z³ − z¹}` and report alignment of `(z² − z¹)`.

    Parameters
    ----------
    pooled_triples
        One entry per patient: ``(p¹, p², p³)`` with each element a flat
        feature vector of shape ``(D,)`` (typically ``D = M * C = 16``).
    """
    if not pooled_triples:
        raise ValueError("pooled_triples is empty")

    p1 = np.stack([np.asarray(t[0]).ravel() for t in pooled_triples], axis=0).astype(np.float64)
    p2 = np.stack([np.asarray(t[1]).ravel() for t in pooled_triples], axis=0).astype(np.float64)
    p3 = np.stack([np.asarray(t[2]).ravel() for t in pooled_triples], axis=0).astype(np.float64)
    diffs_13 = p3 - p1  # (n_patients, D)
    diffs_12 = p2 - p1

    # Centred PCA on diffs_13. With n < D, the leading SV is a 1-D projection.
    mean_d = diffs_13.mean(axis=0, keepdims=True)
    centred = diffs_13 - mean_d
    # economy SVD: U S V^T, with V^T (n_components, D)
    _, s, vh = np.linalg.svd(centred, full_matrices=False)
    direction = vh[0]
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    total_var = float(np.sum(s**2))
    var_explained = float(s[0] ** 2) / max(total_var, 1e-12)

    # Sign-fix the PC so most patients have cos((z³−z¹), direction) > 0 — this
    # convention makes the axis point "in the growth direction" on average.
    raw_cos_13 = diffs_13 @ direction / np.maximum(np.linalg.norm(diffs_13, axis=1), 1e-12)
    if np.mean(raw_cos_13) < 0:
        direction = -direction
        raw_cos_13 = -raw_cos_13

    cos_12 = diffs_12 @ direction / np.maximum(np.linalg.norm(diffs_12, axis=1), 1e-12)

    return SubspaceResult(
        variance_explained=var_explained,
        direction=direction.astype(np.float64),
        singular_values=s.astype(np.float64),
        cos_z2_minus_z1=cos_12.astype(np.float64),
        cos_z3_minus_z1=raw_cos_13.astype(np.float64),
    )


def null_pca_alignment(
    triples_null: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    direction: np.ndarray,
) -> np.ndarray:
    """Compute the same cos_z2_minus_z1 alignment for cross-patient null triples.

    Pass the population-fitted direction; this avoids re-fitting per null
    sample. Returns shape ``(n_null,)``.
    """
    out = np.empty(len(triples_null), dtype=np.float64)
    for i, (a, b, _c) in enumerate(triples_null):
        d = np.asarray(b).ravel() - np.asarray(a).ravel()
        n = float(np.linalg.norm(d))
        out[i] = float(d @ direction / max(n, 1e-12))
    return out
