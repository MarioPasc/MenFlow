"""Tests for load_brats_men_clinical and join_clinical_to_scans."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from menflow.data.clinical import (
    ClinicalCovariates,
    join_clinical_to_scans,
    load_brats_men_clinical,
)

# ---------------------------------------------------------------------------
# Fixture: tiny xlsx in tmp_path
# ---------------------------------------------------------------------------


@pytest.fixture()
def clinical_xlsx(tmp_path: Path) -> Path:
    """Build a 5-row xlsx fixture covering edge cases:

    - Row 0: complete, grade=1
    - Row 1: NaN Grade
    - Row 2: outlier grade string "Anaplastic hemangiopericytoma, III"
    - Row 3: NaN Age
    - Row 4: NaN Sex
    """
    brats_ids = [
        "BraTS-MEN-00001-000",
        "BraTS-MEN-00002-000",
        "BraTS-MEN-00003-000",
        "BraTS-MEN-00004-000",
        "BraTS-MEN-00005-000",
    ]
    df = pd.DataFrame(
        {
            "BraTS ID": brats_ids,
            "Split": ["Training", "Training", "Validation", "Training", "Validation"],
            "Grade": [1, np.nan, "Anaplastic hemangiopericytoma, III", 2, 3],
            "Age": [55.0, 62.0, 48.0, np.nan, 71.0],
            "Sex": ["Male", "Female", "Male", "Female", np.nan],
        }
    )
    path = tmp_path / "clinical.xlsx"
    df.to_excel(path, index=False)
    return path


# ---------------------------------------------------------------------------
# load_brats_men_clinical tests
# ---------------------------------------------------------------------------


def test_load_returns_clinical_covariates(clinical_xlsx: Path) -> None:
    """Return type must be ClinicalCovariates."""
    cov = load_brats_men_clinical(clinical_xlsx)
    assert isinstance(cov, ClinicalCovariates)


def test_index_matches_brats_ids(clinical_xlsx: Path) -> None:
    """table.index must match the BraTS IDs from the xlsx."""
    cov = load_brats_men_clinical(clinical_xlsx)
    expected_ids = [
        "BraTS-MEN-00001-000",
        "BraTS-MEN-00002-000",
        "BraTS-MEN-00003-000",
        "BraTS-MEN-00004-000",
        "BraTS-MEN-00005-000",
    ]
    assert list(cov.table.index) == expected_ids


def test_grade_binning_nan_to_unknown(clinical_xlsx: Path) -> None:
    """NaN Grade must be mapped to 'unknown'."""
    cov = load_brats_men_clinical(clinical_xlsx)
    assert cov.table.loc["BraTS-MEN-00002-000", "grade_bin"] == "unknown"


def test_grade_binning_outlier_string_to_3(clinical_xlsx: Path) -> None:
    """'Anaplastic hemangiopericytoma, III' must be mapped to '3'."""
    cov = load_brats_men_clinical(clinical_xlsx)
    assert cov.table.loc["BraTS-MEN-00003-000", "grade_bin"] == "3"


def test_grade_binning_integer_to_string(clinical_xlsx: Path) -> None:
    """Integer grades 1/2/3 must be returned as string '1'/'2'/'3'."""
    cov = load_brats_men_clinical(clinical_xlsx)
    assert cov.table.loc["BraTS-MEN-00001-000", "grade_bin"] == "1"
    assert cov.table.loc["BraTS-MEN-00004-000", "grade_bin"] == "2"
    assert cov.table.loc["BraTS-MEN-00005-000", "grade_bin"] == "3"


def test_nan_age_preserved(clinical_xlsx: Path) -> None:
    """Row with NaN Age must have NaN in table['age']."""
    cov = load_brats_men_clinical(clinical_xlsx)
    assert pd.isna(cov.table.loc["BraTS-MEN-00004-000", "age"])


def test_nan_sex_preserved(clinical_xlsx: Path) -> None:
    """Row with NaN Sex must have NA in table['sex']."""
    cov = load_brats_men_clinical(clinical_xlsx)
    assert pd.isna(cov.table.loc["BraTS-MEN-00005-000", "sex"])


def test_missing_file_raises(tmp_path: Path) -> None:
    """Non-existent path must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_brats_men_clinical(tmp_path / "does_not_exist.xlsx")


# ---------------------------------------------------------------------------
# join_clinical_to_scans tests
# ---------------------------------------------------------------------------


def test_join_returns_2_rows_for_2_ids(clinical_xlsx: Path) -> None:
    """join result must have as many rows as the input scan_ids array."""
    cov = load_brats_men_clinical(clinical_xlsx)
    ids = np.array(["BraTS-MEN-00001-000", "BraTS-MEN-UNKNOWN-999"])
    result = join_clinical_to_scans(ids, cov)
    assert len(result) == 2


def test_join_known_id_has_data(clinical_xlsx: Path) -> None:
    """A scan_id present in the xlsx must get its data correctly."""
    cov = load_brats_men_clinical(clinical_xlsx)
    ids = np.array(["BraTS-MEN-00001-000", "BraTS-MEN-UNKNOWN-999"])
    result = join_clinical_to_scans(ids, cov)
    assert result.loc[0, "grade_bin"] == "1"
    np.testing.assert_allclose(result.loc[0, "age"], 55.0)


def test_join_unknown_id_all_nan(clinical_xlsx: Path) -> None:
    """A scan_id absent from the xlsx must get NaN for numeric columns."""
    cov = load_brats_men_clinical(clinical_xlsx)
    ids = np.array(["BraTS-MEN-00001-000", "BraTS-MEN-UNKNOWN-999"])
    result = join_clinical_to_scans(ids, cov)
    # age must be NaN for unknown scan
    assert pd.isna(result.loc[1, "age"])


def test_join_range_index(clinical_xlsx: Path) -> None:
    """join result must have a RangeIndex (not the scan_id index)."""
    cov = load_brats_men_clinical(clinical_xlsx)
    ids = np.array(["BraTS-MEN-00001-000", "BraTS-MEN-00002-000"])
    result = join_clinical_to_scans(ids, cov)
    assert list(result.index) == [0, 1]
