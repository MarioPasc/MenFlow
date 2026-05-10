"""Translate target ``Δlog V`` into latent-space step magnitudes.

The naive ridge probe of E2.1 produces a direction ``w`` that satisfies::

    w^T (z_tumor) ≈ log V  (up to a bias term)

so a unit step along ``w`` corresponds to ``||w||`` units of ``log V``. To move
``log V`` by ``Δ``, perturb the latent by::

    alpha = Δ / ||w||           (along the unit direction u_v = w / ||w||)

The corresponding inverse, used to interpret a measured latent perturbation as
a predicted ``Δ log V``, is :func:`alpha_to_delta`.
"""

from __future__ import annotations


def delta_to_alpha(delta_log_v: float, direction_norm: float) -> float:
    """Convert a target ``Δlog V`` into an alpha along the unit direction.

    Parameters
    ----------
    delta_log_v
        Target change in ``log V`` (natural log, units of nats; volume change
        factor is ``exp(delta_log_v)``).
    direction_norm
        ``||w||`` from the ridge probe — the un-normalized coefficient norm
        carried in ``E2_1_recoverability/per_modality/{mod}/direction.npz`` as
        ``direction_norm``.

    Returns
    -------
    float
        Magnitude ``alpha`` to multiply the unit direction by.

    Raises
    ------
    ValueError
        If ``direction_norm`` is non-positive.
    """
    if direction_norm <= 0.0:
        raise ValueError(f"direction_norm must be positive; got {direction_norm}")
    return float(delta_log_v) / float(direction_norm)


def alpha_to_delta(alpha: float, direction_norm: float) -> float:
    """Inverse of :func:`delta_to_alpha`."""
    return float(alpha) * float(direction_norm)
