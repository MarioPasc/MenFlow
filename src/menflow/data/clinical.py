"""Clinical covariate loader for BraTS-MEN-2023 supplementary xlsx.

Reads the official supplementary clinical data spreadsheet, bins histological
grade, and provides a join utility to align covariates with scan arrays loaded
from the unified MenFlow HDF5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Mapping from known grade string outliers to canonical bin
_GRADE_OUTLIER_MAP: dict[str, str] = {
    "Anaplastic hemangiopericytoma, III": "3",
}


def _bin_grade(raw: object) -> str:
    """Convert a raw Grade cell value to a canonical grade bin string.

    Parameters
    ----------
    raw:
        Value from the ``Grade`` column; may be int, float (NaN), or str.

    Returns
    -------
    str
        One of ``"1"``, ``"2"``, ``"3"``, or ``"unknown"``.
    """
    if pd.isna(raw):
        return "unknown"
    if isinstance(raw, (int, float)):
        i = int(raw)
        if i in (1, 2, 3):
            return str(i)
        logger.warning("Unrecognised numeric grade %s; mapped to 'unknown'", raw)
        return "unknown"
    s = str(raw).strip()
    if s in ("1", "2", "3"):
        return s
    if s in _GRADE_OUTLIER_MAP:
        mapped = _GRADE_OUTLIER_MAP[s]
        logger.debug("Grade outlier '%s' mapped to '%s'", s, mapped)
        return mapped
    logger.warning("Unrecognised grade string '%s'; mapped to 'unknown'", s)
    return "unknown"


@dataclass(frozen=True, slots=True)
class ClinicalCovariates:
    """Parsed clinical covariate table indexed by scan ID.

    Attributes
    ----------
    table:
        :class:`pandas.DataFrame` indexed by ``scan_id`` (str).  Columns:
        ``age`` (float), ``sex`` (str or NaN), ``grade_bin`` (str),
        ``split`` (str).
    """

    table: pd.DataFrame


def load_brats_men_clinical(xlsx_path: Path) -> ClinicalCovariates:
    """Load the BraTS-MEN-2023 supplementary clinical xlsx.

    Parameters
    ----------
    xlsx_path:
        Path to the supplementary clinical data Excel file.  Expected columns:
        ``BraTS ID``, ``Split``, ``Grade``, ``Age``, ``Sex``, plus per-modality
        resolution/slice-thickness columns (ignored).

    Returns
    -------
    ClinicalCovariates
        Parsed and binned covariate table.

    Raises
    ------
    FileNotFoundError
        If ``xlsx_path`` does not exist.
    KeyError
        If any required column is absent.
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"xlsx not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path)

    # Strip whitespace from BraTS ID and set as index
    df["BraTS ID"] = df["BraTS ID"].astype(str).str.strip()
    df = df.set_index("BraTS ID")

    for col in ("Split", "Grade", "Age", "Sex"):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in {xlsx_path}")

    table = pd.DataFrame(index=df.index)
    table.index.name = "scan_id"
    table["age"] = pd.to_numeric(df["Age"], errors="coerce")
    table["sex"] = df["Sex"].where(df["Sex"].isin(["Male", "Female"]), other=pd.NA)
    table["grade_bin"] = df["Grade"].apply(_bin_grade)
    table["split"] = df["Split"].astype(str).str.strip()

    n_nan_age = int(table["age"].isna().sum())
    n_nan_sex = int(table["sex"].isna().sum())
    n_unknown_grade = int((table["grade_bin"] == "unknown").sum())
    logger.info(
        "load_brats_men_clinical: %d rows loaded; NaN age=%d, NaN sex=%d, unknown grade=%d",
        len(table),
        n_nan_age,
        n_nan_sex,
        n_unknown_grade,
    )

    return ClinicalCovariates(table=table)


def join_clinical_to_scans(
    scan_ids: np.ndarray,
    covariates: ClinicalCovariates,
) -> pd.DataFrame:
    """Align clinical covariates to a scan array in order.

    Parameters
    ----------
    scan_ids:
        1-D array of scan ID strings, shape ``(N,)``, matching the ordering
        of rows in a latent features H5 or image array.
    covariates:
        Parsed covariate table from :func:`load_brats_men_clinical`.

    Returns
    -------
    pd.DataFrame
        RangeIndex 0..N-1 DataFrame with columns ``age``, ``sex``,
        ``grade_bin``, ``split``.  Rows for scan IDs absent from the xlsx
        receive NaN / ``"unknown"`` values.
    """
    scan_ids = np.asarray(scan_ids, dtype=str)
    rows = covariates.table.reindex(scan_ids)
    n_missing = int(
        rows["age"].isna().sum() - covariates.table["age"].isna().reindex(scan_ids).sum()
    )
    # Simpler: count IDs not in the table index
    n_not_found = int(np.sum(~np.isin(scan_ids, covariates.table.index)))
    if n_not_found > 0:
        logger.warning(
            "join_clinical_to_scans: %d scan_ids not found in clinical table; "
            "their rows will be NaN/unknown",
            n_not_found,
        )
    result = rows.reset_index(drop=True)
    return result
