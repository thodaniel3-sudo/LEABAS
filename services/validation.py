"""
LEBAS - Dataset Validation Service

File:
    services/validation.py

Purpose:
    Performs detailed validation and data-quality checks on the LEBAS
    laboratory dataset.

Important design principles:
    - No experimental values are hard-coded.
    - Sieve sizes are completely dynamic.
    - 150 µm is a valid sieve size.
    - Repeated sieve rows are expected.
    - Process-level values are not double-counted.
    - Validation does not modify the user's DataFrame.
    - Warnings do not automatically make the dataset invalid.
    - Errors identify problems that can affect reliable calculations.

Public API includes:
    validate_dataset()
    validate_dataframe()
    validate_dataset_file()
    check_required_columns()
    check_missing_values()
    check_duplicate_rows()
    check_numeric_columns()
    check_mass_values()
    check_percentage_ranges()
    check_sieve_data()
    check_sample_structure()
    check_process_consistency()
    check_sieve_mass_consistency()
    check_beneficiation_balance()
    check_size_reduction_data()
    get_validation_summary()
    get_data_quality_report()
    is_dataset_valid()
    format_validation_report()
    DatasetValidator
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# ============================================================================
# EXPECTED SCHEMA
# ============================================================================

EXPECTED_COLUMNS = [
    "sample_id",
    "material_type",
    "experiment_date",
    "operator",
    "replicate",
    "feed_mass",
    "feed_grade",
    "feed_moisture",
    "crusher_type",
    "crush_time",
    "crush_power",
    "crush_feed_size",
    "crush_product_size",
    "grinding_time",
    "grinding_mill_type",
    "grinding_media_mass",
    "grinding_speed",
    "grinding_feed_size",
    "grinding_product_size",
    "sieve_size",
    "mass_retained",
    "sieve_shaking_time",
    "beneficiation_method",
    "beneficiation_time",
    "reagent_type",
    "reagent_dosage",
    "ph_level",
    "pulp_density",
    "temperature",
    "concentrate_mass",
    "concentrate_grade",
    "tailings_mass",
    "tailings_grade",
    "losses_mass",
    "water_added",
    "solid_recovery",
]


# ============================================================================
# REQUIRED COLUMNS
# ============================================================================

# Only these are absolutely required for a usable LEBAS dataset.
# Other columns may legitimately be unavailable for a particular experiment.
REQUIRED_COLUMNS = [
    "sample_id",
    "replicate",
]


# ============================================================================
# NUMERIC COLUMNS
# ============================================================================

NUMERIC_COLUMNS = [
    "replicate",
    "feed_mass",
    "feed_grade",
    "feed_moisture",
    "crush_time",
    "crush_power",
    "crush_feed_size",
    "crush_product_size",
    "grinding_time",
    "grinding_media_mass",
    "grinding_speed",
    "grinding_feed_size",
    "grinding_product_size",
    "sieve_size",
    "mass_retained",
    "sieve_shaking_time",
    "beneficiation_time",
    "reagent_dosage",
    "ph_level",
    "pulp_density",
    "temperature",
    "concentrate_mass",
    "concentrate_grade",
    "tailings_mass",
    "tailings_grade",
    "losses_mass",
    "water_added",
    "solid_recovery",
]


# ============================================================================
# PERCENTAGE-LIKE COLUMNS
# ============================================================================

PERCENTAGE_COLUMNS = [
    "feed_grade",
    "feed_moisture",
    "concentrate_grade",
    "tailings_grade",
    "solid_recovery",
]


# ============================================================================
# MASS COLUMNS
# ============================================================================

MASS_COLUMNS = [
    "feed_mass",
    "grinding_media_mass",
    "mass_retained",
    "concentrate_mass",
    "tailings_mass",
    "losses_mass",
    "water_added",
]


# ============================================================================
# PROCESS-LEVEL COLUMNS
# ============================================================================

PROCESS_LEVEL_COLUMNS = [
    "feed_mass",
    "feed_grade",
    "feed_moisture",
    "crusher_type",
    "crush_time",
    "crush_power",
    "crush_feed_size",
    "crush_product_size",
    "grinding_time",
    "grinding_mill_type",
    "grinding_media_mass",
    "grinding_speed",
    "grinding_feed_size",
    "grinding_product_size",
    "beneficiation_method",
    "beneficiation_time",
    "reagent_type",
    "reagent_dosage",
    "ph_level",
    "pulp_density",
    "temperature",
    "concentrate_mass",
    "concentrate_grade",
    "tailings_mass",
    "tailings_grade",
    "losses_mass",
    "water_added",
    "solid_recovery",
]


# ============================================================================
# VALIDATION RESULT
# ============================================================================

@dataclass
class ValidationResult:
    """
    Standard validation result used by the LEBAS application.

    Attributes:
        valid:
            True when no error-level problems were detected.

        errors:
            Error-level validation issues.

        warnings:
            Warning-level validation issues.

        info:
            Informational messages.

        details:
            Detailed results from individual validation checks.

        counts:
            Useful summary statistics.
    """

    valid: bool = True

    errors: List[Dict[str, Any]] = field(
        default_factory=list
    )

    warnings: List[Dict[str, Any]] = field(
        default_factory=list
    )

    info: List[Dict[str, Any]] = field(
        default_factory=list
    )

    details: Dict[str, Any] = field(
        default_factory=dict
    )

    counts: Dict[str, Any] = field(
        default_factory=dict
    )

    def add_error(
        self,
        message: str,
        category: str = "general",
        rows: Optional[List[int]] = None,
        column: Optional[str] = None,
    ) -> None:
        """Add an error-level issue."""

        issue = {
            "severity": "error",
            "category": category,
            "message": message,
        }

        if rows is not None:
            issue["rows"] = rows

        if column is not None:
            issue["column"] = column

        self.errors.append(issue)
        self.valid = False

    def add_warning(
        self,
        message: str,
        category: str = "general",
        rows: Optional[List[int]] = None,
        column: Optional[str] = None,
    ) -> None:
        """Add a warning-level issue."""

        issue = {
            "severity": "warning",
            "category": category,
            "message": message,
        }

        if rows is not None:
            issue["rows"] = rows

        if column is not None:
            issue["column"] = column

        self.warnings.append(issue)

    def add_info(
        self,
        message: str,
        category: str = "general",
    ) -> None:
        """Add an informational message."""

        self.info.append(
            {
                "category": category,
                "message": message,
            }
        )

    def finalize(self) -> "ValidationResult":
        """Update the validity flag and return this object."""

        self.valid = len(self.errors) == 0

        self.counts["errors"] = len(self.errors)
        self.counts["warnings"] = len(self.warnings)
        self.counts["info"] = len(self.info)

        return self

    @property
    def error_count(self) -> int:
        """Return the number of errors."""

        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Return the number of warnings."""

        return len(self.warnings)


# ============================================================================
# HELPERS
# ============================================================================

def _safe_numeric(series: pd.Series) -> pd.Series:
    """Safely convert a pandas Series to numeric values."""

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def _is_blank(value: Any) -> bool:
    """Return True when a value is missing or blank."""

    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass

    return str(value).strip() == ""


def _normalise_text(value: Any) -> str:
    """Normalize a value for text comparison."""

    if _is_blank(value):
        return ""

    return str(value).strip().lower()


def _normalise_replicate(value: Any) -> str:
    """Normalize replicate values for comparison."""

    if _is_blank(value):
        return ""

    try:
        numeric = float(value)

        if numeric.is_integer():
            return str(int(numeric))

    except (ValueError, TypeError):
        pass

    return str(value).strip()


def _add_issue(
    container: List[Dict[str, Any]],
    severity: str,
    category: str,
    message: str,
    rows: Optional[List[int]] = None,
    column: Optional[str] = None,
) -> None:
    """Add a structured issue to a list."""

    issue = {
        "severity": severity,
        "category": category,
        "message": message,
    }

    if rows is not None:
        issue["rows"] = rows

    if column is not None:
        issue["column"] = column

    container.append(issue)


def _process_level_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create one process-level row per sample/replicate.

    This is extremely important because one laboratory experiment may
    contain several sieve rows.

    Example:

        Sample 1 / Replicate 1 / 500 µm
        Sample 1 / Replicate 1 / 300 µm
        Sample 1 / Replicate 1 / 212 µm
        Sample 1 / Replicate 1 / 150 µm

    These are four sieve observations but ONE process-level experiment.
    """

    if df.empty:
        return df.copy()

    if not {
        "sample_id",
        "replicate",
    }.issubset(df.columns):
        return df.copy()

    working = df.copy()

    completeness_columns = [
        column
        for column in PROCESS_LEVEL_COLUMNS
        if column in working.columns
    ]

    if completeness_columns:
        working["_process_completeness"] = (
            working[completeness_columns]
            .notna()
            .sum(axis=1)
        )
    else:
        working["_process_completeness"] = 0

    working = working.sort_values(
        "_process_completeness",
        ascending=False,
        kind="stable",
    )

    process = (
        working
        .groupby(
            [
                "sample_id",
                "replicate",
            ],
            as_index=False,
            sort=False,
            dropna=False,
        )
        .first()
    )

    if "_process_completeness" in process.columns:
        process = process.drop(
            columns=["_process_completeness"],
            errors="ignore",
        )

    return process.reset_index(drop=True)


def _basic_statistics(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Calculate basic dataset statistics."""

    statistics: Dict[str, Any] = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_cells": int(
            df.isna().sum().sum()
        ),
        "exact_duplicate_rows": int(
            df.duplicated().sum()
        ),
    }

    if "sample_id" in df.columns:
        sample_values = (
            df["sample_id"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        sample_values = sample_values[
            sample_values != ""
        ]

        statistics["unique_samples"] = int(
            sample_values.nunique()
        )
    else:
        statistics["unique_samples"] = 0

    if "replicate" in df.columns:
        replicate_values = (
            df["replicate"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        replicate_values = replicate_values[
            replicate_values != ""
        ]

        statistics["unique_replicates"] = int(
            replicate_values.nunique()
        )
    else:
        statistics["unique_replicates"] = 0

    if {
        "sample_id",
        "replicate",
    }.issubset(df.columns):

        process = df[
            [
                "sample_id",
                "replicate",
            ]
        ].drop_duplicates()

        statistics["process_records"] = int(
            len(process)
        )
    else:
        statistics["process_records"] = 0

    if "sieve_size" in df.columns:

        sieve = _safe_numeric(
            df["sieve_size"]
        )

        sieve = sieve[
            sieve > 0
        ].dropna()

        statistics["unique_sieve_sizes"] = int(
            sieve.nunique()
        )

        statistics["minimum_sieve_size"] = (
            float(sieve.min())
            if not sieve.empty
            else None
        )

        statistics["maximum_sieve_size"] = (
            float(sieve.max())
            if not sieve.empty
            else None
        )

    else:

        statistics["unique_sieve_sizes"] = 0
        statistics["minimum_sieve_size"] = None
        statistics["maximum_sieve_size"] = None

    return statistics


# ============================================================================
# REQUIRED COLUMN CHECK
# ============================================================================

def check_required_columns(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Check required and expected columns."""

    columns = set(df.columns)

    missing_required = [
        column
        for column in REQUIRED_COLUMNS
        if column not in columns
    ]

    missing_expected = [
        column
        for column in EXPECTED_COLUMNS
        if column not in columns
    ]

    missing_optional = [
        column
        for column in missing_expected
        if column not in REQUIRED_COLUMNS
    ]

    issues: List[Dict[str, Any]] = []

    for column in missing_required:

        _add_issue(
            issues,
            "error",
            "schema",
            f"Required column is missing: {column}",
            column=column,
        )

    for column in missing_optional:

        _add_issue(
            issues,
            "warning",
            "schema",
            (
                f"Expected optional column is missing: "
                f"{column}"
            ),
            column=column,
        )

    return {
        "valid": len(missing_required) == 0,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "issues": issues,
    }


# ============================================================================
# MISSING VALUE CHECK
# ============================================================================

def check_missing_values(
    df: pd.DataFrame,
    important_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Check missing values without modifying the dataset."""

    if important_columns is None:
        important_columns = [
            "sample_id",
            "replicate",
            "feed_mass",
            "sieve_size",
            "mass_retained",
            "concentrate_mass",
        ]

    issues: List[Dict[str, Any]] = []
    missing_counts: Dict[str, int] = {}

    for column in df.columns:

        count = int(
            df[column].isna().sum()
        )

        if count > 0:
            missing_counts[column] = count

    for column in important_columns:

        if column not in df.columns:
            continue

        count = int(
            df[column].isna().sum()
        )

        if count == 0:
            continue

        if column == "sample_id":
            severity = "error"
        else:
            severity = "warning"

        _add_issue(
            issues,
            severity,
            "missing_values",
            (
                f"{count} row(s) have missing "
                f"values in '{column}'."
            ),
            column=column,
        )

    return {
        "valid": not any(
            issue["severity"] == "error"
            for issue in issues
        ),
        "missing_counts": missing_counts,
        "issues": issues,
    }


# ============================================================================
# DUPLICATE CHECK
# ============================================================================

def check_duplicate_rows(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Check exact duplicate rows.

    Repeated sieve rows are NOT treated as duplicates unless every
    column in the row is identical.
    """

    if df.empty:
        return {
            "valid": True,
            "duplicate_count": 0,
            "duplicate_rows": [],
            "issues": [],
        }

    duplicate_mask = df.duplicated(
        keep=False
    )

    duplicate_rows = (
        df.index[duplicate_mask].tolist()
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    issues: List[Dict[str, Any]] = []

    if duplicate_count > 0:

        _add_issue(
            issues,
            "warning",
            "duplicates",
            (
                f"{duplicate_count} row(s) are exact "
                "duplicates of another row."
            ),
            rows=duplicate_rows[:100],
        )

    return {
        "valid": True,
        "duplicate_count": duplicate_count,
        "duplicate_rows": duplicate_rows,
        "issues": issues,
    }


# ============================================================================
# NUMERIC CHECK
# ============================================================================

def check_numeric_columns(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Check numeric columns for invalid non-numeric values."""

    issues: List[Dict[str, Any]] = []
    invalid_counts: Dict[str, int] = {}

    for column in NUMERIC_COLUMNS:

        if column not in df.columns:
            continue

        original = df[column]

        converted = _safe_numeric(
            original
        )

        non_empty = (
            original.notna()
            & (
                original.astype(str)
                .str.strip()
                != ""
            )
        )

        invalid_mask = (
            non_empty
            & converted.isna()
        )

        count = int(
            invalid_mask.sum()
        )

        if count <= 0:
            continue

        invalid_counts[column] = count

        rows = (
            df.index[invalid_mask]
            .tolist()
        )

        _add_issue(
            issues,
            "error",
            "numeric_data",
            (
                f"{count} non-numeric value(s) "
                f"found in '{column}'."
            ),
            rows=rows[:100],
            column=column,
        )

    return {
        "valid": not any(
            issue["severity"] == "error"
            for issue in issues
        ),
        "invalid_counts": invalid_counts,
        "issues": issues,
    }


# ============================================================================
# DATE CHECK
# ============================================================================

def check_date_values(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Check experiment_date values."""

    issues: List[Dict[str, Any]] = []

    if "experiment_date" not in df.columns:
        return {
            "valid": True,
            "issues": issues,
            "invalid_count": 0,
        }

    original = df["experiment_date"]

    non_empty = (
        original.notna()
        & (
            original.astype(str)
            .str.strip()
            != ""
        )
    )

    if not non_empty.any():
        return {
            "valid": True,
            "issues": issues,
            "invalid_count": 0,
        }

    converted = pd.to_datetime(
        original,
        errors="coerce",
    )

    invalid_mask = (
        non_empty
        & converted.isna()
    )

    count = int(
        invalid_mask.sum()
    )

    if count > 0:

        _add_issue(
            issues,
            "warning",
            "date_values",
            (
                f"{count} experiment_date value(s) "
                "could not be interpreted as dates."
            ),
            rows=df.index[
                invalid_mask
            ].tolist()[:100],
            column="experiment_date",
        )

    return {
        "valid": True,
        "issues": issues,
        "invalid_count": count,
    }


# ============================================================================
# MASS CHECK
# ============================================================================

def check_mass_values(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Check physical mass values."""

    issues: List[Dict[str, Any]] = []
    negative_counts: Dict[str, int] = {}

    for column in MASS_COLUMNS:

        if column not in df.columns:
            continue

        values = _safe_numeric(
            df[column]
        )

        mask = (
            values.notna()
            & (
                values < 0
            )
        )

        count = int(
            mask.sum()
        )

        if count <= 0:
            continue

        negative_counts[column] = count

        _add_issue(
            issues,
            "error",
            "mass_values",
            (
                f"{count} negative mass value(s) "
                f"found in '{column}'."
            ),
            rows=df.index[
                mask
            ].tolist()[:100],
            column=column,
        )

    # Feed mass must be positive when supplied.
    if "feed_mass" in df.columns:

        feed = _safe_numeric(
            df["feed_mass"]
        )

        zero_or_negative = (
            feed.notna()
            & (
                feed <= 0
            )
        )

        count = int(
            zero_or_negative.sum()
        )

        if count > 0:

            _add_issue(
                issues,
                "error",
                "mass_values",
                (
                    f"{count} feed_mass value(s) "
                    "are zero or negative."
                ),
                rows=df.index[
                    zero_or_negative
                ].tolist()[:100],
                column="feed_mass",
            )

    return {
        "valid": not any(
            issue["severity"] == "error"
            for issue in issues
        ),
        "negative_counts": negative_counts,
        "issues": issues,
    }


# ============================================================================
# PERCENTAGE RANGE CHECK
# ============================================================================

def check_percentage_ranges(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Check percentage-like values.

    Negative values are errors.

    Values above 100 are warnings because laboratory datasets may use
    different representations or units.
    """

    issues: List[Dict[str, Any]] = []

    below_zero: Dict[str, int] = {}
    above_hundred: Dict[str, int] = {}

    for column in PERCENTAGE_COLUMNS:

        if column not in df.columns:
            continue

        values = _safe_numeric(
            df[column]
        )

        negative_mask = (
            values.notna()
            & (
                values < 0
            )
        )

        high_mask = (
            values.notna()
            & (
                values > 100
            )
        )

        negative_count = int(
            negative_mask.sum()
        )

        high_count = int(
            high_mask.sum()
        )

        if negative_count > 0:

            below_zero[column] = negative_count

            _add_issue(
                issues,
                "error",
                "percentage_range",
                (
                    f"{negative_count} negative "
                    f"value(s) found in '{column}'."
                ),
                rows=df.index[
                    negative_mask
                ].tolist()[:100],
                column=column,
            )

        if high_count > 0:

            above_hundred[column] = high_count

            _add_issue(
                issues,
                "warning",
                "percentage_range",
                (
                    f"{high_count} value(s) above 100 "
                    f"found in '{column}'. Check the "
                    "unit or representation."
                ),
                rows=df.index[
                    high_mask
                ].tolist()[:100],
                column=column,
            )

    return {
        "valid": not any(
            issue["severity"] == "error"
            for issue in issues
        ),
        "below_zero": below_zero,
        "above_hundred": above_hundred,
        "issues": issues,
    }


# ============================================================================
# SIEVE CHECK
# ============================================================================

def check_sieve_data(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Validate sieve analysis data.

    No hard-coded sieve sizes are used.

    Therefore 150 µm is valid, as are other positive sieve sizes.
    """

    issues: List[Dict[str, Any]] = []

    if "sieve_size" not in df.columns:

        _add_issue(
            issues,
            "warning",
            "sieve_analysis",
            "sieve_size column is not present.",
            column="sieve_size",
        )

        return {
            "valid": True,
            "sieve_count": 0,
            "minimum_sieve": None,
            "maximum_sieve": None,
            "sieve_sizes": [],
            "issues": issues,
        }

    sieve = _safe_numeric(
        df["sieve_size"]
    )

    invalid_mask = (
        sieve.notna()
        & (
            sieve <= 0
        )
    )

    invalid_count = int(
        invalid_mask.sum()
    )

    if invalid_count > 0:

        _add_issue(
            issues,
            "error",
            "sieve_analysis",
            (
                f"{invalid_count} sieve size value(s) "
                "are zero or negative."
            ),
            rows=df.index[
                invalid_mask
            ].tolist()[:100],
            column="sieve_size",
        )

    valid_sieve = sieve[
        sieve > 0
    ].dropna()

    unique_sieves = sorted(
        [
            float(value)
            for value in valid_sieve.unique()
        ],
        reverse=True,
    )

    if valid_sieve.empty:

        minimum_sieve = None
        maximum_sieve = None

        _add_issue(
            issues,
            "warning",
            "sieve_analysis",
            "No valid positive sieve sizes were found.",
            column="sieve_size",
        )

    else:

        minimum_sieve = float(
            valid_sieve.min()
        )

        maximum_sieve = float(
            valid_sieve.max()
        )

    if "mass_retained" in df.columns:

        retained = _safe_numeric(
            df["mass_retained"]
        )

        missing_retained = (
            retained.isna()
            & df["sieve_size"].notna()
        )

        count = int(
            missing_retained.sum()
        )

        if count > 0:

            _add_issue(
                issues,
                "warning",
                "sieve_analysis",
                (
                    f"{count} sieve row(s) have a sieve "
                    "size but no usable mass_retained value."
                ),
                rows=df.index[
                    missing_retained
                ].tolist()[:100],
                column="mass_retained",
            )

    if minimum_sieve is not None:

        logger.info(
            "Minimum detected sieve size: %s",
            minimum_sieve,
        )

        if np.isclose(
            minimum_sieve,
            150.0,
        ):
            logger.info(
                "150 µm detected as minimum sieve. "
                "This is valid."
            )

    return {
        "valid": not any(
            issue["severity"] == "error"
            for issue in issues
        ),
        "sieve_count": len(unique_sieves),
        "minimum_sieve": minimum_sieve,
        "maximum_sieve": maximum_sieve,
        "sieve_sizes": unique_sieves,
        "issues": issues,
    }


# ============================================================================
# SAMPLE / REPLICATE STRUCTURE
# ============================================================================

def check_sample_structure(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Check sample IDs, replicates and sieve combinations."""

    issues: List[Dict[str, Any]] = []

    if "sample_id" not in df.columns:

        _add_issue(
            issues,
            "error",
            "sample_structure",
            "sample_id column is missing.",
            column="sample_id",
        )

        return {
            "valid": False,
            "issues": issues,
        }

    sample = (
        df["sample_id"]
        .astype("string")
        .str.strip()
    )

    blank_sample = (
        sample.isna()
        | (
            sample == ""
        )
    )

    count = int(
        blank_sample.sum()
    )

    if count > 0:

        _add_issue(
            issues,
            "error",
            "sample_structure",
            (
                f"{count} row(s) have a blank sample_id."
            ),
            rows=df.index[
                blank_sample
            ].tolist()[:100],
            column="sample_id",
        )

    if "replicate" in df.columns:

        replicate = df["replicate"]

        blank_replicate = (
            replicate.isna()
            | (
                replicate.astype(str)
                .str.strip()
                == ""
            )
        )

        count = int(
            blank_replicate.sum()
        )

        if count > 0:

            _add_issue(
                issues,
                "warning",
                "sample_structure",
                (
                    f"{count} row(s) have no replicate value."
                ),
                rows=df.index[
                    blank_replicate
                ].tolist()[:100],
                column="replicate",
            )

    grouping_columns = [
        column
        for column in [
            "sample_id",
            "replicate",
            "sieve_size",
        ]
        if column in df.columns
    ]

    if len(grouping_columns) == 3:

        duplicate_mask = df.duplicated(
            subset=grouping_columns,
            keep=False,
        )

        count = int(
            duplicate_mask.sum()
        )

        if count > 0:

            _add_issue(
                issues,
                "warning",
                "sample_structure",
                (
                    f"{count} row(s) share the same "
                    "sample/replicate/sieve combination. "
                    "Check whether they are intentional "
                    "repeated measurements."
                ),
                rows=df.index[
                    duplicate_mask
                ].tolist()[:100],
            )

    return {
        "valid": not any(
            issue["severity"] == "error"
            for issue in issues
        ),
        "issues": issues,
    }


# ============================================================================
# PROCESS CONSISTENCY
# ============================================================================

def check_process_consistency(
    df: pd.DataFrame,
    tolerance: float = 1e-9,
) -> Dict[str, Any]:
    """
    Check whether process-level values remain consistent across
    repeated sieve rows.
    """

    issues: List[Dict[str, Any]] = []

    required = {
        "sample_id",
        "replicate",
    }

    if not required.issubset(df.columns):

        return {
            "valid": True,
            "issues": issues,
            "inconsistent_fields": [],
        }

    available_columns = [
        column
        for column in PROCESS_LEVEL_COLUMNS
        if column in df.columns
    ]

    if not available_columns:

        return {
            "valid": True,
            "issues": issues,
            "inconsistent_fields": [],
        }

    inconsistent_fields: List[
        Dict[str, Any]
    ] = []

    grouped = df.groupby(
        [
            "sample_id",
            "replicate",
        ],
        dropna=False,
    )

    for (
        sample_id,
        replicate,
    ), group in grouped:

        if len(group) <= 1:
            continue

        for column in available_columns:

            values = group[column].dropna()

            if len(values) <= 1:
                continue

            if column in NUMERIC_COLUMNS:

                numeric = _safe_numeric(
                    values
                ).dropna()

                if numeric.empty:
                    continue

                minimum = float(
                    numeric.min()
                )

                maximum = float(
                    numeric.max()
                )

                difference = abs(
                    maximum - minimum
                )

                if difference > tolerance:

                    inconsistent_fields.append(
                        {
                            "sample_id": sample_id,
                            "replicate": replicate,
                            "column": column,
                            "minimum": minimum,
                            "maximum": maximum,
                        }
                    )

            else:

                normalized = (
                    values
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .unique()
                )

                if len(normalized) > 1:

                    inconsistent_fields.append(
                        {
                            "sample_id": sample_id,
                            "replicate": replicate,
                            "column": column,
                            "values": normalized.tolist(),
                        }
                    )

    for item in inconsistent_fields[:100]:

        _add_issue(
            issues,
            "warning",
            "process_consistency",
            (
                "Process-level value differs across "
                "repeated sieve rows for sample "
                f"{item['sample_id']} replicate "
                f"{item['replicate']} in column "
                f"'{item['column']}'."
            ),
        )

    return {
        "valid": True,
        "issues": issues,
        "inconsistent_fields": inconsistent_fields,
    }


# ============================================================================
# SIEVE MASS CONSISTENCY
# ============================================================================

def check_sieve_mass_consistency(
    df: pd.DataFrame,
    tolerance_percent: float = 10.0,
) -> Dict[str, Any]:
    """
    Compare total retained sieve mass against feed mass.

    This is a warning-level quality check, not an automatic failure.
    """

    issues: List[Dict[str, Any]] = []

    required = {
        "sample_id",
        "replicate",
        "feed_mass",
        "mass_retained",
    }

    if not required.issubset(df.columns):

        return {
            "valid": True,
            "issues": issues,
            "records": [],
        }

    working = df.copy()

    working["_feed_mass"] = _safe_numeric(
        working["feed_mass"]
    )

    working["_mass_retained"] = _safe_numeric(
        working["mass_retained"]
    )

    records: List[Dict[str, Any]] = []

    grouped = working.groupby(
        [
            "sample_id",
            "replicate",
        ],
        dropna=False,
    )

    for (
        sample_id,
        replicate,
    ), group in grouped:

        feed_values = (
            group["_feed_mass"]
            .dropna()
        )

        retained_values = (
            group["_mass_retained"]
            .dropna()
        )

        if feed_values.empty:
            continue

        feed_mass = float(
            feed_values.iloc[0]
        )

        if feed_mass <= 0:
            continue

        if retained_values.empty:
            continue

        retained_mass = float(
            retained_values.sum()
        )

        recovery = (
            retained_mass
            / feed_mass
            * 100.0
        )

        difference = abs(
            recovery - 100.0
        )

        record = {
            "sample_id": sample_id,
            "replicate": replicate,
            "feed_mass": feed_mass,
            "sieve_retained_mass": retained_mass,
            "sieve_mass_recovery_percent": recovery,
            "difference_from_100_percent": difference,
        }

        records.append(record)

        if difference > tolerance_percent:

            _add_issue(
                issues,
                "warning",
                "sieve_mass_consistency",
                (
                    f"Sieve mass recovery for sample "
                    f"{sample_id}, replicate {replicate} "
                    f"is {recovery:.2f}%, outside the "
                    f"{tolerance_percent:.2f}% tolerance "
                    "around 100%."
                ),
            )

    return {
        "valid": True,
        "issues": issues,
        "records": records,
    }


# ============================================================================
# BENEFICIATION MASS BALANCE
# ============================================================================

def check_beneficiation_balance(
    df: pd.DataFrame,
    tolerance_percent: float = 5.0,
) -> Dict[str, Any]:
    """
    Check beneficiation mass balance.

    Formula:

        Balance Difference =
            Feed - Concentrate - Tailings - Losses

        Balance Error (%) =
            abs(Balance Difference)
            ----------------------- × 100
                  Feed

    Repeated sieve rows are reduced to one process-level record.
    """

    issues: List[Dict[str, Any]] = []

    required = {
        "sample_id",
        "replicate",
        "feed_mass",
        "concentrate_mass",
    }

    if not required.issubset(df.columns):

        return {
            "valid": True,
            "issues": issues,
            "records": [],
        }

    process = _process_level_dataframe(
        df
    ).copy()

    numeric_columns = [
        "feed_mass",
        "concentrate_mass",
        "tailings_mass",
        "losses_mass",
    ]

    for column in numeric_columns:

        if column not in process.columns:
            process[column] = np.nan

        process[column] = _safe_numeric(
            process[column]
        )

    records: List[Dict[str, Any]] = []

    for _, row in process.iterrows():

        feed = row["feed_mass"]
        concentrate = row["concentrate_mass"]

        if pd.isna(feed):
            continue

        if pd.isna(concentrate):
            continue

        feed = float(feed)
        concentrate = float(concentrate)

        if feed <= 0:
            continue

        tailings = row["tailings_mass"]
        losses = row["losses_mass"]

        # Missing tailings/losses are not silently interpreted as
        # measured zero unless the field itself is present and has
        # a numeric zero value.
        #
        # For the purpose of balance calculation, however, missing
        # output quantities cannot be independently verified.
        if pd.isna(tailings) and pd.isna(losses):

            _add_issue(
                issues,
                "warning",
                "beneficiation_balance",
                (
                    f"Cannot fully verify mass balance for "
                    f"sample {row['sample_id']}, replicate "
                    f"{row['replicate']} because tailings_mass "
                    "and losses_mass are both missing."
                ),
            )

            continue

        tailings_value = (
            0.0
            if pd.isna(tailings)
            else float(tailings)
        )

        losses_value = (
            0.0
            if pd.isna(losses)
            else float(losses)
        )

        output = (
            concentrate
            + tailings_value
            + losses_value
        )

        difference = (
            feed
            - output
        )

        error_percent = (
            abs(difference)
            / feed
            * 100.0
        )

        record = {
            "sample_id": row["sample_id"],
            "replicate": row["replicate"],
            "feed_mass": feed,
            "concentrate_mass": concentrate,
            "tailings_mass": tailings_value,
            "losses_mass": losses_value,
            "output_mass": output,
            "balance_difference": difference,
            "balance_error_percent": error_percent,
        }

        records.append(record)

        if error_percent > tolerance_percent:

            _add_issue(
                issues,
                "warning",
                "beneficiation_balance",
                (
                    f"Material balance error for sample "
                    f"{row['sample_id']}, replicate "
                    f"{row['replicate']} is "
                    f"{error_percent:.2f}%, above the "
                    f"{tolerance_percent:.2f}% tolerance."
                ),
            )

    return {
        "valid": True,
        "issues": issues,
        "records": records,
    }


# ============================================================================
# SIZE REDUCTION CHECK
# ============================================================================

def check_size_reduction_data(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Check crushing and grinding feed/product sizes.

    A product size larger than feed size is reported as a warning.
    """

    issues: List[Dict[str, Any]] = []

    checks = [
        (
            "crush_feed_size",
            "crush_product_size",
            "crushing",
        ),
        (
            "grinding_feed_size",
            "grinding_product_size",
            "grinding",
        ),
    ]

    for (
        feed_column,
        product_column,
        process,
    ) in checks:

        if (
            feed_column not in df.columns
            or product_column not in df.columns
        ):
            continue

        feed = _safe_numeric(
            df[feed_column]
        )

        product = _safe_numeric(
            df[product_column]
        )

        mask = (
            feed.notna()
            & product.notna()
            & (
                feed > 0
            )
            & (
                product > feed
            )
        )

        count = int(
            mask.sum()
        )

        if count > 0:

            _add_issue(
                issues,
                "warning",
                "size_reduction",
                (
                    f"{count} {process} row(s) have a "
                    "product size larger than the feed size."
                ),
                rows=df.index[
                    mask
                ].tolist()[:100],
                column=product_column,
            )

    return {
        "valid": True,
        "issues": issues,
    }


# ============================================================================
# COMPLETE VALIDATION
# ============================================================================

def validate_dataset(
    df: pd.DataFrame,
    strict: bool = False,
    sieve_mass_tolerance_percent: float = 10.0,
    balance_tolerance_percent: float = 5.0,
) -> ValidationResult:
    """
    Perform complete validation of a LEBAS DataFrame.

    Parameters
    ----------
    df:
        Dataset to validate.

    strict:
        If True, warnings are also converted into errors.

    sieve_mass_tolerance_percent:
        Allowed difference between sieve retained mass and feed mass.

    balance_tolerance_percent:
        Allowed beneficiation mass-balance error.

    Returns
    -------
    ValidationResult
        Structured validation result.

    Important:
        The DataFrame is never modified.
    """

    result = ValidationResult()

    if df is None:

        result.add_error(
            "Dataset is None.",
            category="dataset",
        )

        return result.finalize()

    if not isinstance(df, pd.DataFrame):

        result.add_error(
            "Dataset must be a pandas DataFrame.",
            category="dataset",
        )

        return result.finalize()

    if df.empty:

        result.add_error(
            "Dataset is empty.",
            category="dataset",
        )

        result.counts = {
            "rows": 0,
            "columns": len(df.columns),
        }

        return result.finalize()

    # ------------------------------------------------------------------------
    # Run individual checks.
    # ------------------------------------------------------------------------

    checks: Dict[str, Dict[str, Any]] = {}

    checks["required_columns"] = (
        check_required_columns(df)
    )

    checks["missing_values"] = (
        check_missing_values(df)
    )

    checks["duplicate_rows"] = (
        check_duplicate_rows(df)
    )

    checks["numeric_columns"] = (
        check_numeric_columns(df)
    )

    checks["date_values"] = (
        check_date_values(df)
    )

    checks["mass_values"] = (
        check_mass_values(df)
    )

    checks["percentage_ranges"] = (
        check_percentage_ranges(df)
    )

    checks["sieve_data"] = (
        check_sieve_data(df)
    )

    checks["sample_structure"] = (
        check_sample_structure(df)
    )

    checks["process_consistency"] = (
        check_process_consistency(df)
    )

    checks["sieve_mass_consistency"] = (
        check_sieve_mass_consistency(
            df,
            tolerance_percent=(
                sieve_mass_tolerance_percent
            ),
        )
    )

    checks["beneficiation_balance"] = (
        check_beneficiation_balance(
            df,
            tolerance_percent=(
                balance_tolerance_percent
            ),
        )
    )

    checks["size_reduction"] = (
        check_size_reduction_data(df)
    )

    result.details = checks

    # ------------------------------------------------------------------------
    # Collect all issues.
    # ------------------------------------------------------------------------

    for check_name, check_result in checks.items():

        for issue in check_result.get(
            "issues",
            [],
        ):

            issue_copy = dict(issue)

            issue_copy["check"] = check_name

            severity = issue_copy.get(
                "severity",
                "warning",
            )

            if severity == "error":
                result.errors.append(
                    issue_copy
                )
            else:
                result.warnings.append(
                    issue_copy
                )

    # ------------------------------------------------------------------------
    # Statistics.
    # ------------------------------------------------------------------------

    result.counts.update(
        _basic_statistics(df)
    )

    # ------------------------------------------------------------------------
    # Informational messages.
    # ------------------------------------------------------------------------

    statistics = result.counts

    result.add_info(
        (
            f"Dataset contains {statistics.get('rows', 0)} "
            "rows."
        ),
        category="dataset",
    )

    result.add_info(
        (
            f"Dataset contains "
            f"{statistics.get('unique_samples', 0)} "
            "unique sample(s)."
        ),
        category="dataset",
    )

    if statistics.get(
        "minimum_sieve_size"
    ) is not None:

        result.add_info(
            (
                "Detected sieve range: "
                f"{statistics['minimum_sieve_size']} to "
                f"{statistics['maximum_sieve_size']}."
            ),
            category="sieve_analysis",
        )

    # ------------------------------------------------------------------------
    # Strict mode.
    # ------------------------------------------------------------------------

    if strict and result.warnings:

        strict_warnings = []

        for warning in result.warnings:

            converted = dict(warning)
            converted["severity"] = "error"
            converted["strict_mode"] = True

            strict_warnings.append(
                converted
            )

        result.errors.extend(
            strict_warnings
        )

        result.add_info(
            (
                "Strict mode was enabled. "
                "Warning-level issues were treated as errors."
            ),
            category="strict_mode",
        )

    result.valid = (
        len(result.errors) == 0
    )

    result.counts["errors"] = (
        len(result.errors)
    )

    result.counts["warnings"] = (
        len(result.warnings)
    )

    result.counts["info"] = (
        len(result.info)
    )

    logger.info(
        (
            "LEBAS dataset validation completed: "
            "valid=%s, errors=%d, warnings=%d"
        ),
        result.valid,
        len(result.errors),
        len(result.warnings),
    )

    return result


# ============================================================================
# BACKWARD-COMPATIBLE VALIDATION FUNCTION
# ============================================================================

def validate_dataframe(
    df: pd.DataFrame,
    sieve_mass_tolerance_percent: float = 10.0,
    balance_tolerance_percent: float = 5.0,
) -> Dict[str, Any]:
    """
    Backward-compatible validation function.

    Older LEBAS modules may expect a dictionary instead of
    ValidationResult.

    Newer modules should use validate_dataset().
    """

    result = validate_dataset(
        df,
        strict=False,
        sieve_mass_tolerance_percent=(
            sieve_mass_tolerance_percent
        ),
        balance_tolerance_percent=(
            balance_tolerance_percent
        ),
    )

    return {
        "valid": result.valid,
        "errors": result.errors,
        "warnings": result.warnings,
        "info": result.info,
        "checks": result.details,
        "details": result.details,
        "statistics": result.counts,
        "counts": result.counts,
    }


# ============================================================================
# FILE VALIDATION
# ============================================================================

def validate_dataset_file(
    filepath: str | Path,
    strict: bool = False,
    sieve_mass_tolerance_percent: float = 10.0,
    balance_tolerance_percent: float = 5.0,
) -> ValidationResult:
    """Load and validate a CSV file."""

    path = Path(filepath).resolve()

    if not path.exists():

        result = ValidationResult()

        result.add_error(
            f"Dataset file does not exist: {path}",
            category="file",
        )

        return result.finalize()

    if not path.is_file():

        result = ValidationResult()

        result.add_error(
            f"Dataset path is not a file: {path}",
            category="file",
        )

        return result.finalize()

    try:

        df = pd.read_csv(path)

    except UnicodeDecodeError:

        try:

            df = pd.read_csv(
                path,
                encoding="latin-1",
            )

        except Exception as exc:

            result = ValidationResult()

            result.add_error(
                (
                    "Could not read dataset file: "
                    f"{exc}"
                ),
                category="file",
            )

            return result.finalize()

    except Exception as exc:

        result = ValidationResult()

        result.add_error(
            (
                "Could not read dataset file: "
                f"{exc}"
            ),
            category="file",
        )

        return result.finalize()

    # Normalize column names.
    normalized_columns = []

    for column in df.columns:

        name = str(column).strip().lower()

        name = (
            name
            .replace("-", "_")
            .replace(" ", "_")
        )

        while "__" in name:

            name = name.replace(
                "__",
                "_",
            )

        normalized_columns.append(name)

    df.columns = normalized_columns

    return validate_dataset(
        df,
        strict=strict,
        sieve_mass_tolerance_percent=(
            sieve_mass_tolerance_percent
        ),
        balance_tolerance_percent=(
            balance_tolerance_percent
        ),
    )


# ============================================================================
# DATA QUALITY REPORT
# ============================================================================

def get_data_quality_report(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Return a compact data-quality report.

    This is useful for dashboard KPI cards and report generation.
    """

    result = validate_dataset(df)

    return {
        "valid": result.valid,
        "status": (
            "VALID"
            if result.valid and not result.warnings
            else (
                "VALID WITH WARNINGS"
                if result.valid
                else "INVALID"
            )
        ),
        "error_count": len(
            result.errors
        ),
        "warning_count": len(
            result.warnings
        ),
        "info_count": len(
            result.info
        ),
        "statistics": result.counts,
        "errors": result.errors,
        "warnings": result.warnings,
        "details": result.details,
    }


# ============================================================================
# VALIDATION SUMMARY
# ============================================================================

def get_validation_summary(
    validation_report: Any,
) -> Dict[str, Any]:
    """
    Convert either a ValidationResult or dictionary report into a
    compact summary.
    """

    if validation_report is None:

        return {
            "valid": False,
            "status": "INVALID",
            "error_count": 1,
            "warning_count": 0,
            "message": "No validation report was supplied.",
        }

    if isinstance(
        validation_report,
        ValidationResult,
    ):

        errors = validation_report.errors
        warnings = validation_report.warnings
        valid = validation_report.valid

    elif isinstance(
        validation_report,
        dict,
    ):

        errors = validation_report.get(
            "errors",
            [],
        )

        warnings = validation_report.get(
            "warnings",
            [],
        )

        valid = bool(
            validation_report.get(
                "valid",
                False,
            )
        )

    else:

        return {
            "valid": False,
            "status": "INVALID",
            "error_count": 1,
            "warning_count": 0,
            "message": (
                "Unsupported validation report type."
            ),
        }

    if valid and warnings:

        status = "VALID WITH WARNINGS"

    elif valid:

        status = "VALID"

    else:

        status = "INVALID"

    if valid and not warnings:

        message = "Dataset passed validation."

    elif valid:

        message = (
            "Dataset passed validation but has warnings."
        )

    else:

        message = (
            "Dataset contains validation errors."
        )

    return {
        "valid": valid,
        "status": status,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "message": message,
    }


# ============================================================================
# BOOLEAN VALIDITY CHECK
# ============================================================================

def is_dataset_valid(
    df: pd.DataFrame,
) -> bool:
    """Return True when there are no error-level issues."""

    result = validate_dataset(df)

    return bool(
        result.valid
    )


# ============================================================================
# FORMAT VALIDATION REPORT
# ============================================================================

def format_validation_report(
    validation_report: Any,
) -> str:
    """Convert a validation report to readable text."""

    summary = get_validation_summary(
        validation_report
    )

    lines = [
        "LEBAS DATASET VALIDATION REPORT",
        "=" * 45,
        f"Status: {summary['status']}",
        f"Errors: {summary['error_count']}",
        f"Warnings: {summary['warning_count']}",
        "",
    ]

    if isinstance(
        validation_report,
        ValidationResult,
    ):

        statistics = (
            validation_report.counts
        )

        errors = (
            validation_report.errors
        )

        warnings = (
            validation_report.warnings
        )

        info = (
            validation_report.info
        )

    elif isinstance(
        validation_report,
        dict,
    ):

        statistics = (
            validation_report.get(
                "statistics",
                validation_report.get(
                    "counts",
                    {},
                ),
            )
        )

        errors = (
            validation_report.get(
                "errors",
                [],
            )
        )

        warnings = (
            validation_report.get(
                "warnings",
                [],
            )
        )

        info = (
            validation_report.get(
                "info",
                [],
            )
        )

    else:

        statistics = {}
        errors = []
        warnings = []
        info = []

    if statistics:

        lines.append(
            "DATASET STATISTICS"
        )

        lines.append(
            "-" * 30
        )

        for key, value in statistics.items():

            label = (
                str(key)
                .replace("_", " ")
                .title()
            )

            if isinstance(
                value,
                float,
            ):

                value_text = (
                    f"{value:.4f}"
                )

            else:

                value_text = str(
                    value
                )

            lines.append(
                f"{label}: {value_text}"
            )

        lines.append("")

    if errors:

        lines.append(
            "ERRORS"
        )

        lines.append(
            "-" * 30
        )

        for index, issue in enumerate(
            errors,
            start=1,
        ):

            lines.append(
                f"{index}. "
                f"{issue.get('message', '')}"
            )

        lines.append("")

    if warnings:

        lines.append(
            "WARNINGS"
        )

        lines.append(
            "-" * 30
        )

        for index, issue in enumerate(
            warnings,
            start=1,
        ):

            lines.append(
                f"{index}. "
                f"{issue.get('message', '')}"
            )

        lines.append("")

    if info:

        lines.append(
            "INFORMATION"
        )

        lines.append(
            "-" * 30
        )

        for item in info:

            if isinstance(
                item,
                dict,
            ):

                message = item.get(
                    "message",
                    "",
                )

            else:

                message = str(item)

            lines.append(
                f"- {message}"
            )

        lines.append("")

    if not errors and not warnings:

        lines.append(
            "No validation issues were detected."
        )

    return "\n".join(lines)


# ============================================================================
# DATASET VALIDATOR CLASS
# ============================================================================

class DatasetValidator:
    """
    Object-oriented interface for dataset validation.

    Useful when the GUI needs to validate the same DataFrame more
    than once.
    """

    def __init__(
        self,
        strict: bool = False,
        sieve_mass_tolerance_percent: float = 10.0,
        balance_tolerance_percent: float = 5.0,
    ) -> None:

        self.strict = bool(strict)

        self.sieve_mass_tolerance_percent = float(
            sieve_mass_tolerance_percent
        )

        self.balance_tolerance_percent = float(
            balance_tolerance_percent
        )

        self.last_result: Optional[
            ValidationResult
        ] = None

    def validate(
        self,
        df: pd.DataFrame,
    ) -> ValidationResult:
        """Validate a DataFrame."""

        self.last_result = validate_dataset(
            df,
            strict=self.strict,
            sieve_mass_tolerance_percent=(
                self.sieve_mass_tolerance_percent
            ),
            balance_tolerance_percent=(
                self.balance_tolerance_percent
            ),
        )

        return self.last_result

    def validate_file(
        self,
        filepath: str | Path,
    ) -> ValidationResult:
        """Validate a CSV file."""

        self.last_result = validate_dataset_file(
            filepath,
            strict=self.strict,
            sieve_mass_tolerance_percent=(
                self.sieve_mass_tolerance_percent
            ),
            balance_tolerance_percent=(
                self.balance_tolerance_percent
            ),
        )

        return self.last_result

    def is_valid(
        self,
        df: pd.DataFrame,
    ) -> bool:
        """Return True when validation passes."""

        return self.validate(
            df
        ).valid

    def get_summary(self) -> Dict[str, Any]:
        """Return summary of the last validation."""

        if self.last_result is None:

            return {
                "valid": False,
                "status": "NOT VALIDATED",
                "error_count": 0,
                "warning_count": 0,
                "message": (
                    "No validation has been performed yet."
                ),
            }

        return get_validation_summary(
            self.last_result
        )


# ============================================================================
# BACKWARD-COMPATIBLE ALIASES
# ============================================================================

validate_data = validate_dataframe
check_dataset = validate_dataframe
validate_file = validate_dataset_file


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "ValidationResult",
    "DatasetValidator",
    "EXPECTED_COLUMNS",
    "REQUIRED_COLUMNS",
    "NUMERIC_COLUMNS",
    "PERCENTAGE_COLUMNS",
    "MASS_COLUMNS",
    "PROCESS_LEVEL_COLUMNS",
    "check_required_columns",
    "check_missing_values",
    "check_duplicate_rows",
    "check_numeric_columns",
    "check_date_values",
    "check_mass_values",
    "check_percentage_ranges",
    "check_sieve_data",
    "check_sample_structure",
    "check_process_consistency",
    "check_sieve_mass_consistency",
    "check_beneficiation_balance",
    "check_size_reduction_data",
    "validate_dataset",
    "validate_dataframe",
    "validate_dataset_file",
    "get_data_quality_report",
    "get_validation_summary",
    "is_dataset_valid",
    "format_validation_report",
    "validate_data",
    "check_dataset",
    "validate_file",
]