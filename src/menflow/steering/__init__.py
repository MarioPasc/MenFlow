"""Latent-space steering primitives for MAISI-v2 (E2.4 onward).

The library is dataset- and model-agnostic. It operates on the same
``(B, C, H', W', D')`` latent layout used everywhere else in MenFlow.
"""

from menflow.steering.anchors import StratifiedAnchor, stratified_anchor_indices
from menflow.steering.calibration import alpha_to_delta, delta_to_alpha
from menflow.steering.drift import off_manifold_drift
from menflow.steering.mask_dilation import dilate_mask_lat
from menflow.steering.operator import STEER_OPERATORS, global_steer, local_steer

__all__ = [
    "STEER_OPERATORS",
    "StratifiedAnchor",
    "alpha_to_delta",
    "delta_to_alpha",
    "dilate_mask_lat",
    "global_steer",
    "local_steer",
    "off_manifold_drift",
    "stratified_anchor_indices",
]
