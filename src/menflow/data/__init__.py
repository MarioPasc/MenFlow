"""Data ingestion, conversion, and HDF5 schema definitions."""

from __future__ import annotations

from menflow.data.h5_converter import H5Converter, ScanData, ScanRecord
from menflow.data.h5_schema import (
    H5Schema,
    H5SchemaError,
    SchemaViolation,
    validate_h5,
)

__all__ = [
    "H5Converter",
    "H5Schema",
    "H5SchemaError",
    "ScanData",
    "ScanRecord",
    "SchemaViolation",
    "validate_h5",
]
