"""Steering operators for MAISI-v2 latents.

Two operator families are defined per the E2.3 carry-forward:

* :func:`local_steer` — adds ``alpha * u_v`` only at latent voxels where the
  projected tumor mask is non-zero. Mathematically::

      z' = z + alpha * u_v ⊗ mask_lat

* :func:`global_steer` — adds ``alpha * u_v`` everywhere. Used as a control /
  comparator only; E2.3 selected ``"strongly_local"`` for every modality.

Inputs and outputs are torch tensors at latent resolution. The functions are
side-effect-free and dtype-preserving (the addition is performed in the input
dtype so no fp32 ↔ fp16 casts leak out).
"""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor


def _broadcast_direction(u_v: Tensor, c: int) -> Tensor:
    """Reshape a length-C direction to ``(C, 1, 1, 1)`` for spatial broadcasting."""
    if u_v.ndim != 1:
        raise ValueError(f"u_v must be 1-D (length C); got shape {tuple(u_v.shape)}")
    if u_v.numel() != c:
        raise ValueError(f"u_v length {u_v.numel()} does not match latent channels {c}")
    return u_v.view(c, 1, 1, 1)


def _ensure_unbatched(z: Tensor, mask_lat: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
    """Drop a leading batch axis if present, returning the (C,H',W',D') view.

    Steering is defined per-anchor; batching is the engine's responsibility, not
    the operator's.
    """
    if z.ndim == 5:
        if z.shape[0] != 1:
            raise ValueError(f"steering operators expect a single anchor; got batch={z.shape[0]}")
        z = z[0]
    if z.ndim != 4:
        raise ValueError(f"z must be 4-D (C,H',W',D') or (1,C,H',W',D'); got {tuple(z.shape)}")
    if mask_lat is not None:
        if mask_lat.ndim == 5:
            if mask_lat.shape[:2] != (1, 1):
                raise ValueError(
                    f"mask_lat must be (1,1,H',W',D') or (H',W',D'); got {tuple(mask_lat.shape)}"
                )
            mask_lat = mask_lat[0, 0]
        elif mask_lat.ndim == 4:
            if mask_lat.shape[0] != 1:
                raise ValueError(f"mask_lat batch axis must be 1; got {mask_lat.shape[0]}")
            mask_lat = mask_lat[0]
        if mask_lat.ndim != 3:
            raise ValueError(f"mask_lat must reduce to 3-D (H',W',D'); got {tuple(mask_lat.shape)}")
        if mask_lat.shape != z.shape[1:]:
            raise ValueError(
                f"mask_lat spatial shape {tuple(mask_lat.shape)} != z spatial "
                f"shape {tuple(z.shape[1:])}"
            )
    return z, mask_lat


def local_steer(
    z: Tensor,
    mask_lat: Tensor,
    u_v: Tensor,
    alpha: float,
) -> Tensor:
    """Add ``alpha * u_v`` to ``z`` only at latent voxels where ``mask_lat == 1``.

    Parameters
    ----------
    z
        Latent tensor of shape ``(C, H', W', D')`` or ``(1, C, H', W', D')``.
    mask_lat
        Binary mask at latent resolution, shape ``(H', W', D')`` /
        ``(1, H', W', D')`` / ``(1, 1, H', W', D')``. Cast to ``z.dtype``.
    u_v
        Direction vector of length ``C`` (unit-norm by convention; not enforced).
    alpha
        Scalar magnitude. ``alpha = delta_log_v / direction_norm``.

    Returns
    -------
    Tensor
        Steered latent in the same shape and dtype as the un-batched ``z``.
    """
    z_un, mask_un = _ensure_unbatched(z, mask_lat)
    assert mask_un is not None
    c = z_un.shape[0]
    u = _broadcast_direction(u_v, c).to(dtype=z_un.dtype, device=z_un.device)
    m = mask_un.to(dtype=z_un.dtype, device=z_un.device)[None]  # (1, H', W', D')
    return z_un + (alpha * u) * m


def global_steer(
    z: Tensor,
    mask_lat: Tensor | None,
    u_v: Tensor,
    alpha: float,
) -> Tensor:
    """Add ``alpha * u_v`` to every latent voxel.

    ``mask_lat`` is accepted for signature compatibility with :func:`local_steer`
    but ignored. Used only as a comparator; E2.3 carry-forward selected the
    local operator for all four modalities.
    """
    del mask_lat
    z_un, _ = _ensure_unbatched(z)
    c = z_un.shape[0]
    u = _broadcast_direction(u_v, c).to(dtype=z_un.dtype, device=z_un.device)
    return z_un + alpha * u


SteerFn = Callable[[Tensor, Tensor, Tensor, float], Tensor]
STEER_OPERATORS: dict[str, SteerFn] = {
    "local": local_steer,
    "global": global_steer,
    "strongly_local": local_steer,  # alias — E2.3 vocabulary
}
