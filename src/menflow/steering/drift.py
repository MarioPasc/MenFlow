"""Off-manifold drift metric for steered latents.

After steering ``z' = z + alpha * u_v * mask_lat``, decoding to ``x_hat`` and
re-encoding to ``z_re``, the relative L2 distance between the *intended* latent
and the encoder's reading of the decoder's image::

    drift = ||z' - z_re|| / ||z'||

quantifies how far the perturbation has pushed the system off the encoder's
training manifold. Values close to 0 indicate the decoder + encoder agree on
the requested edit; values approaching 1 indicate the perturbation is being
silently corrected by the autoencoder.
"""

from __future__ import annotations

import torch
from torch import Tensor


def off_manifold_drift(z_steered: Tensor, z_re_encoded: Tensor) -> float:
    """Return ``||z_steered - z_re_encoded|| / ||z_steered||``.

    Parameters
    ----------
    z_steered
        Intended steered latent (output of :func:`menflow.steering.operator.local_steer`).
    z_re_encoded
        Encoder output after decoding ``z_steered`` and re-encoding the result.

    Returns
    -------
    float
        Non-negative scalar; 0 when the autoencoder is exactly idempotent on the
        steered latent. Computed in float32 to avoid fp16 underflow on small
        differences.
    """
    if z_steered.shape != z_re_encoded.shape:
        raise ValueError(
            f"shape mismatch: z_steered={tuple(z_steered.shape)} vs "
            f"z_re_encoded={tuple(z_re_encoded.shape)}"
        )
    a = z_steered.detach().to(torch.float32)
    b = z_re_encoded.detach().to(torch.float32)
    num = torch.linalg.vector_norm(a - b)
    den = torch.linalg.vector_norm(a).clamp_min(1e-12)
    return float((num / den).item())
