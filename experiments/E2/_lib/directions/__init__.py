"""Direction estimation utilities for latent-space tumor-volume analysis.

Three families of directions are provided:

- **Naive**: normalized Ridge probe coefficient (``u_naive``).
- **TCAV**: matched-control logistic direction (``u_TCAV``).
- **FWL**: Frisch-Waugh-Lovell residualized Ridge direction (``u_FWL``).
"""

from experiments.E2._lib.directions.fwl import fwl_direction, fwl_residualize
from experiments.E2._lib.directions.naive import naive_volume_direction
from experiments.E2._lib.directions.tcav import (
    InsufficientMatchedControlsError,
    TCAVResult,
    matched_control_tcav,
)

__all__ = [
    "naive_volume_direction",
    "matched_control_tcav",
    "TCAVResult",
    "InsufficientMatchedControlsError",
    "fwl_residualize",
    "fwl_direction",
]
