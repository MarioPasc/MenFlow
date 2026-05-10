"""Generic BraTS-MEN segmentation routine.

Public entry points:

* :class:`SegmentBratsEngine` — orchestrates per-subject Docker invocation.
* :class:`SegmentBratsRoutineConfig` — frozen-dataclass config (loaded from YAML).
"""

from routines.segment_brats.engine.segment_brats_engine import (
    SegmentBratsEngine,
    SegmentBratsRoutineConfig,
)

__all__ = ["SegmentBratsEngine", "SegmentBratsRoutineConfig"]
