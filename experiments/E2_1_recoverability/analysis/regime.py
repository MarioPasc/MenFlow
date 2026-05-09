"""Regime classification for E2.1 outcomes.

Decision rule taken from ``docs/E2/E2_1_recoverability.md`` §5:

- R1            : ``r2_lin >= 0.6 and gap <= 0.10``
- borderline_R1 : ``r2_lin in [0.5, 0.6) and gap <= 0.10``
- R2            : ``r2_lin >= 0.5 and gap > 0.10``  OR  ``r2_lin < 0.5 and r2_mlp >= 0.5``
- R3            : ``r2_mlp < 0.5``  (chain terminates → Path D)
"""

from __future__ import annotations


def classify_regime(r2_lin: float, r2_mlp: float) -> str:
    """Return a regime label string."""
    gap = r2_mlp - r2_lin
    if r2_mlp < 0.5 and r2_lin < 0.5:
        return "R3"
    if r2_lin >= 0.6 and gap <= 0.10:
        return "R1"
    if r2_lin >= 0.5 and gap <= 0.10:
        return "borderline_R1"
    return "R2"
