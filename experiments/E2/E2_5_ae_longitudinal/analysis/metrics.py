"""Core E2.5 metrics: ρ_lin, volume-matched β, image SSIM, non-tumour SSIM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.metrics import structural_similarity as _ssim


class DegenerateTripletError(ValueError):
    """Raised when ``z1`` and ``z3`` are numerically coincident."""


@dataclass(frozen=True, slots=True)
class LinearityResult:
    beta_star: float
    rho_lin: float
    line_norm: float


def linearity_residual(
    z1: np.ndarray,
    z2: np.ndarray,
    z3: np.ndarray,
    *,
    mask_latent: np.ndarray | None = None,
    eps: float = 1e-10,
) -> LinearityResult:
    """Compute β* and ρ_lin for the orthogonal projection of z2 onto z1→z3.

    Operates on any tensor shape; flattens internally so the formula matches
    the spec literally (§4.1). ``rho_lin`` is normalised by the trajectory
    length ``||z3 − z1||`` so it is scale-invariant.

    Parameters
    ----------
    mask_latent
        Optional boolean mask aligned with the *spatial* axes of ``z1`` (i.e.
        shape ``(H', W', D')`` when the latents are ``(M, C, H', W', D')``).
        If supplied, both β* and ρ_lin are computed *inside the mask only*:
        only voxels where ``mask_latent`` is True contribute. This isolates
        the trajectory signal from background latent noise.
    """
    a = np.asarray(z1, dtype=np.float64)
    b = np.asarray(z2, dtype=np.float64)
    c = np.asarray(z3, dtype=np.float64)
    if mask_latent is not None:
        if a.shape[-mask_latent.ndim :] != mask_latent.shape:
            raise ValueError(
                f"trailing axes of z {a.shape} must match mask shape {mask_latent.shape}"
            )
        sel = np.broadcast_to(mask_latent, a.shape)
        af = a[sel]
        bf = b[sel]
        cf = c[sel]
    else:
        af = a.ravel()
        bf = b.ravel()
        cf = c.ravel()
    d = cf - af
    d_sq = float(np.dot(d, d))
    if d_sq < eps:
        raise DegenerateTripletError("z1 and z3 are numerically identical (within the mask).")
    beta = float(np.dot(bf - af, d) / d_sq)
    beta_clipped = float(np.clip(beta, 0.0, 1.0))
    proj = (1.0 - beta_clipped) * af + beta_clipped * cf
    line_norm = float(np.sqrt(d_sq))
    residual = float(np.linalg.norm(bf - proj))
    return LinearityResult(
        beta_star=beta_clipped,
        rho_lin=residual / line_norm,
        line_norm=line_norm,
    )


def linearity_residual_pooled(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, eps: float = 1e-10
) -> LinearityResult:
    """ρ_lin in mask-pooled feature space.

    ``p1, p2, p3`` are the ``(M, C)`` per-modality mask-pooled feature
    vectors emitted by :func:`analysis.latents.mask_pool_per_modality`. The
    triple is flattened to a single 16-D vector before applying the standard
    projection formula. This directly measures the linearity of the features
    the E2.1 probe consumes.
    """
    return linearity_residual(
        p1.ravel(), p2.ravel(), p3.ravel(), mask_latent=None, eps=eps
    )


def volume_matched_beta(
    z1: np.ndarray,
    z3: np.ndarray,
    *,
    mask_latent: np.ndarray,
    log_v_target: float,
    probe_predict_per_modality: list,
    n_grid: int = 21,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Sweep β grid, mask-pool z(β), predict log V per modality, average.

    Parameters
    ----------
    z1, z3
        Latents of shape ``(M, C, H', W', D')``.
    mask_latent
        Bool array of shape ``(H', W', D')`` — the union-of-endpoints dilated
        tumour mask projected to latent space. Used identically across β
        (we do not re-segment per β; the union mask is the defensible choice
        per spec §6.3).
    log_v_target
        Target ``log V₂`` from the GT segmentation, in ``log(cm^3)``.
    probe_predict_per_modality
        List of length M; each entry is a callable
        ``predict(pooled: (..., C)) -> log_v``. Use
        :class:`RidgeProbe.predict` bound to the per-modality probe.

    Returns
    -------
    beta_vol
        Argmin grid point in [0, 1].
    betas
        The β grid.
    log_v_hat
        Probe-predicted ``log V`` for each grid point (modality-averaged).
    """
    if not mask_latent.any():
        # Degenerate: empty tumour mask. Fall back to β = 0.5 with NaN log_v.
        betas = np.linspace(0.0, 1.0, n_grid)
        return 0.5, betas, np.full(n_grid, np.nan)

    betas = np.linspace(0.0, 1.0, n_grid)
    m_flat = mask_latent.reshape(-1)
    n_mask = int(m_flat.sum())

    log_v_hat = np.zeros(n_grid, dtype=np.float64)
    n_mod = z1.shape[0]
    n_ch = z1.shape[1]

    # Pre-flatten and select masked voxels once per modality, per channel.
    # z shape (M, C, ...) -> per-modality (C, n_voxels) — easier vectorisation.
    z1f = z1.reshape(n_mod, n_ch, -1)[:, :, m_flat]  # (M, C, n_mask)
    z3f = z3.reshape(n_mod, n_ch, -1)[:, :, m_flat]

    for bi, beta in enumerate(betas):
        z_beta = (1.0 - beta) * z1f + beta * z3f  # (M, C, n_mask)
        pooled = z_beta.mean(axis=2)  # (M, C)
        # Per-modality log V then average across modalities.
        preds = np.array(
            [
                float(probe_predict_per_modality[mi](pooled[mi]))
                for mi in range(n_mod)
            ]
        )
        log_v_hat[bi] = float(np.nanmean(preds))
    idx = int(np.nanargmin(np.abs(log_v_hat - log_v_target)))
    return float(betas[idx]), betas, log_v_hat


def ssim_3d(x: np.ndarray, y: np.ndarray, *, data_range: float | None = None) -> float:
    """3D SSIM via scikit-image. Returns NaN if both inputs are degenerate."""
    if x.shape != y.shape:
        raise ValueError(f"SSIM shape mismatch: {x.shape} vs {y.shape}")
    if data_range is None:
        data_range = float(max(x.max() - x.min(), y.max() - y.min(), 1e-8))
    try:
        return float(_ssim(x.astype(np.float32), y.astype(np.float32), data_range=data_range))
    except ValueError:
        # Volumes too small for the default win_size; reduce.
        return float(
            _ssim(
                x.astype(np.float32),
                y.astype(np.float32),
                data_range=data_range,
                win_size=3,
            )
        )


def ssim_masked(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray, *, data_range: float | None = None
) -> float:
    """SSIM computed over masked voxels.

    skimage's SSIM does not directly support masking; we proxy by zeroing
    out non-mask voxels in both volumes (so the unmasked region contributes
    identical values to both and the structural-similarity inside the mask
    dominates).
    """
    if x.shape != y.shape or x.shape != mask.shape:
        raise ValueError("ssim_masked: shape mismatch")
    xm = np.where(mask, x, 0.0).astype(np.float32)
    ym = np.where(mask, y, 0.0).astype(np.float32)
    return ssim_3d(xm, ym, data_range=data_range)
