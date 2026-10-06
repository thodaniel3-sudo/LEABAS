
"""
LEBAS - Statistical Calculations Module
=======================================

Laboratory Extraction and Beneficiation Analysis System (LEBAS)

This module provides statistical calculations used throughout the
LEBAS application.

Important dataset rule:
-----------------------
The LEBAS dataset may contain multiple rows for the same
sample_id + replicate because sieve analysis normally has one row
per sieve size.

Therefore, process-level statistics MUST NOT count every sieve row
as a separate experiment.

This module:
    1. Deduplicates process-level records using sample_id + replicate.
    2. Keeps sieve data separate for particle-size statistics.
    3. Calculates mean, median, standard deviation, minimum, maximum,
       coefficient of variation and count.
    4. Calculates beneficiation recovery statistics.
    5. Calculates mass-balance statistics.
    6. Calculates size-reduction statistics.
    7. Provides calculate_statistics(), which is required by the
       Dashboard.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------

PROCESS_KEY_COLUMNS = ["sample_id", "replicate"]

NUMERIC_COLUMNS = [
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


# ---------------------------------------------------------------------
# BASIC HELPERS
# ---------------------------------------------------------------------

def _safe_numeric(series: pd.Series) -> pd.Series:
    """
    Convert a pandas Series to numeric values safely.
    Invalid values become NaN.
    """
    return pd.to_numeric(series, errors="coerce")


def _safe_float(value: Any) -> Optional[float]:
    """
    Safely convert a value to float.
    Returns None when conversion is impossible.
    """
    try:
        if value is None:
            return None

        if pd.isna(value):
            return None

        result = float(value)

        if not math.isfinite(result):
            return None

        return result

    except (TypeError, ValueError):
        return None


def _round_value(value: Any, decimals: int = 4) -> Optional[float]:
    """
    Convert a numeric value to a rounded Python float.
    """
    number = _safe_float(value)

    if number is None:
        return None

    return round(number, decimals)


def _normalise_replicate(series: pd.Series) -> pd.Series:
    """
    Normalise replicate values so that values such as 1 and '1'
    are treated consistently when creating process-level keys.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    result = series.astype(str).str.strip()

    mask = numeric.notna()

    result.loc[mask] = numeric.loc[mask].apply(
        lambda x: str(int(x)) if float(x).is_integer() else str(float(x))
    )

    return result


def _prepare_dataframe(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Safely prepare a copy of the supplied DataFrame.
    """
    if df is None:
        return pd.DataFrame()

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Statistics functions require a pandas DataFrame.")

    data = df.copy()

    # Remove completely empty rows.
    data = data.dropna(how="all").reset_index(drop=True)

    # Convert known numeric columns.
    for column in NUMERIC_COLUMNS:
        if column in data.columns:
            data[column] = _safe_numeric(data[column])

    return data


def _process_level_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per sample_id + replicate.

    This is extremely important for LEBAS because sieve analysis can
    produce several rows for one experiment.

    Example:

        S001 / Rep 1 / 1000 um
        S001 / Rep 1 / 710 um
        S001 / Rep 1 / 500 um
        S001 / Rep 1 / 150 um

    These four rows represent ONE experiment, not four experiments.

    If sample_id and replicate exist, the first process-level record
    for each combination is retained.
    """
    data = _prepare_dataframe(df)

    if data.empty:
        return data

    if all(column in data.columns for column in PROCESS_KEY_COLUMNS):
        data["_statistics_replicate_key"] = _normalise_replicate(
            data["replicate"]
        )

        data = data.drop_duplicates(
            subset=["sample_id", "_statistics_replicate_key"],
            keep="first",
        ).copy()

        data.drop(
            columns=["_statistics_replicate_key"],
            inplace=True,
            errors="ignore",
        )

    else:
        # If the expected key is unavailable, preserve the rows.
        data = data.drop_duplicates().copy()

    return data.reset_index(drop=True)


# ---------------------------------------------------------------------
# DESCRIPTIVE STATISTICS
# ---------------------------------------------------------------------

def calculate_descriptive_statistics(
    series: pd.Series,
    decimals: int = 4,
) -> Dict[str, Any]:
    """
    Calculate standard descriptive statistics for one numeric series.

    Returns:
        count
        mean
        median
        std
        variance
        min
        max
        range
        coefficient_of_variation
    """
    if series is None:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "variance": None,
            "min": None,
            "max": None,
            "range": None,
            "coefficient_of_variation": None,
        }

    numeric = _safe_numeric(series).dropna()

    if numeric.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "variance": None,
            "min": None,
            "max": None,
            "range": None,
            "coefficient_of_variation": None,
        }

    count = int(numeric.count())
    mean = float(numeric.mean())
    median = float(numeric.median())

    # pandas uses sample standard deviation by default.
    std = float(numeric.std(ddof=1)) if count > 1 else 0.0

    variance = (
        float(numeric.var(ddof=1))
        if count > 1
        else 0.0
    )

    minimum = float(numeric.min())
    maximum = float(numeric.max())
    data_range = maximum - minimum

    if mean != 0:
        coefficient_of_variation = abs(std / mean) * 100.0
    else:
        coefficient_of_variation = None

    return {
        "count": count,
        "mean": _round_value(mean, decimals),
        "median": _round_value(median, decimals),
        "std": _round_value(std, decimals),
        "variance": _round_value(variance, decimals),
        "min": _round_value(minimum, decimals),
        "max": _round_value(maximum, decimals),
        "range": _round_value(data_range, decimals),
        "coefficient_of_variation": _round_value(
            coefficient_of_variation,
            decimals,
        ),
    }


def calculate_column_statistics(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    decimals: int = 4,
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate descriptive statistics for multiple columns.
    """
    data = _prepare_dataframe(df)

    if data.empty:
        return {}

    if columns is None:
        columns = [
            column
            for column in data.columns
            if column in NUMERIC_COLUMNS
        ]

    statistics: Dict[str, Dict[str, Any]] = {}

    for column in columns:
        if column not in data.columns:
            continue

        statistics[column] = calculate_descriptive_statistics(
            data[column],
            decimals=decimals,
        )

    return statistics


# ---------------------------------------------------------------------
# SAMPLE / DATASET STATISTICS
# ---------------------------------------------------------------------

def calculate_sample_statistics(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Calculate overall sample-level statistics.

    Sieve rows are deduplicated so that one sample/replicate is counted
    only once.
    """
    data = _process_level_dataframe(df)

    if data.empty:
        return {
            "sample_count": 0,
            "replicate_count": 0,
            "material_type_count": 0,
            "material_types": [],
            "operators": [],
            "date_min": None,
            "date_max": None,
        }

    sample_count = (
        int(data["sample_id"].nunique())
        if "sample_id" in data.columns
        else 0
    )

    replicate_count = (
        int(len(data))
        if all(column in data.columns for column in PROCESS_KEY_COLUMNS)
        else 0
    )

    material_types: List[str] = []

    if "material_type" in data.columns:
        material_types = sorted(
            data["material_type"]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .unique()
            .tolist()
        )

    operators: List[str] = []

    if "operator" in data.columns:
        operators = sorted(
            data["operator"]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .unique()
            .tolist()
        )

    date_min = None
    date_max = None

    if "experiment_date" in data.columns:
        dates = pd.to_datetime(
            data["experiment_date"],
            errors="coerce",
        ).dropna()

        if not dates.empty:
            date_min = dates.min().strftime("%Y-%m-%d")
            date_max = dates.max().strftime("%Y-%m-%d")

    return {
        "sample_count": sample_count,
        "replicate_count": replicate_count,
        "material_type_count": len(material_types),
        "material_types": material_types,
        "operators": operators,
        "operator_count": len(operators),
        "date_min": date_min,
        "date_max": date_max,
    }


# ---------------------------------------------------------------------
# CRUSHING STATISTICS
# ---------------------------------------------------------------------

def calculate_crushing_statistics(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Calculate crushing performance statistics.

    Reduction ratio:
        Feed size / Product size

    Size reduction percentage:
        ((Feed size - Product size) / Feed size) * 100
    """
    data = _process_level_dataframe(df)

    result = {
        "records": 0,
        "mean_feed_size": None,
        "mean_product_size": None,
        "mean_reduction_ratio": None,
        "mean_size_reduction_percent": None,
        "mean_crush_time": None,
        "mean_crush_power": None,
        "reduction_ratio_statistics": {},
        "size_reduction_statistics": {},
    }

    if data.empty:
        return result

    feed_col = "crush_feed_size"
    product_col = "crush_product_size"

    if feed_col in data.columns and product_col in data.columns:
        feed = _safe_numeric(data[feed_col])
        product = _safe_numeric(data[product_col])

        valid = (
            feed.notna()
            & product.notna()
            & (feed > 0)
            & (product > 0)
        )

        if valid.any():
            feed_valid = feed[valid]
            product_valid = product[valid]

            ratio = feed_valid / product_valid

            reduction = (
                (feed_valid - product_valid)
                / feed_valid
            ) * 100.0

            result["records"] = int(valid.sum())
            result["mean_feed_size"] = _round_value(
                feed_valid.mean()
            )
            result["mean_product_size"] = _round_value(
                product_valid.mean()
            )
            result["mean_reduction_ratio"] = _round_value(
                ratio.mean()
            )
            result["mean_size_reduction_percent"] = _round_value(
                reduction.mean()
            )

            result["reduction_ratio_statistics"] = (
                calculate_descriptive_statistics(ratio)
            )

            result["size_reduction_statistics"] = (
                calculate_descriptive_statistics(reduction)
            )

    if "crush_time" in data.columns:
        stats = calculate_descriptive_statistics(
            data["crush_time"]
        )
        result["mean_crush_time"] = stats["mean"]

    if "crush_power" in data.columns:
        stats = calculate_descriptive_statistics(
            data["crush_power"]
        )
        result["mean_crush_power"] = stats["mean"]

    return result


# ---------------------------------------------------------------------
# GRINDING STATISTICS
# ---------------------------------------------------------------------

def calculate_grinding_statistics(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Calculate grinding performance statistics.
    """
    data = _process_level_dataframe(df)

    result = {
        "records": 0,
        "mean_feed_size": None,
        "mean_product_size": None,
        "mean_reduction_ratio": None,
        "mean_size_reduction_percent": None,
        "mean_grinding_time": None,
        "mean_grinding_speed": None,
        "mean_media_mass": None,
        "reduction_ratio_statistics": {},
        "size_reduction_statistics": {},
    }

    if data.empty:
        return result

    feed_col = "grinding_feed_size"
    product_col = "grinding_product_size"

    if feed_col in data.columns and product_col in data.columns:
        feed = _safe_numeric(data[feed_col])
        product = _safe_numeric(data[product_col])

        valid = (
            feed.notna()
            & product.notna()
            & (feed > 0)
            & (product > 0)
        )

        if valid.any():
            feed_valid = feed[valid]
            product_valid = product[valid]

            ratio = feed_valid / product_valid

            reduction = (
                (feed_valid - product_valid)
                / feed_valid
            ) * 100.0

            result["records"] = int(valid.sum())
            result["mean_feed_size"] = _round_value(
                feed_valid.mean()
            )
            result["mean_product_size"] = _round_value(
                product_valid.mean()
            )
            result["mean_reduction_ratio"] = _round_value(
                ratio.mean()
            )
            result["mean_size_reduction_percent"] = _round_value(
                reduction.mean()
            )

            result["reduction_ratio_statistics"] = (
                calculate_descriptive_statistics(ratio)
            )

            result["size_reduction_statistics"] = (
                calculate_descriptive_statistics(reduction)
            )

    if "grinding_time" in data.columns:
        result["mean_grinding_time"] = (
            calculate_descriptive_statistics(
                data["grinding_time"]
            )["mean"]
        )

    if "grinding_speed" in data.columns:
        result["mean_grinding_speed"] = (
            calculate_descriptive_statistics(
                data["grinding_speed"]
            )["mean"]
        )

    if "grinding_media_mass" in data.columns:
        result["mean_media_mass"] = (
            calculate_descriptive_statistics(
                data["grinding_media_mass"]
            )["mean"]
        )

    return result


# ---------------------------------------------------------------------
# SIEVE ANALYSIS STATISTICS
# ---------------------------------------------------------------------

def calculate_sieve_statistics(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Calculate statistics for sieve analysis.

    The function dynamically reads the sieve sizes contained in the
    dataset. It does NOT hard-code sieve sizes.

    Therefore, a laboratory sieve size such as 150 micrometres is
    accepted normally.
    """
    data = _prepare_dataframe(df)

    result: Dict[str, Any] = {
        "sieve_records": 0,
        "sample_count": 0,
        "sieve_sizes": [],
        "smallest_sieve_um": None,
        "largest_sieve_um": None,
        "total_retained_mass": None,
        "sieve_size_statistics": {},
        "mass_retained_statistics": {},
        "sieve_summary": [],
    }

    if data.empty:
        return result

    if "sieve_size" not in data.columns:
        return result

    sieve = _safe_numeric(data["sieve_size"])

    valid = sieve.notna() & (sieve > 0)

    if not valid.any():
        return result

    valid_data = data.loc[valid].copy()
    sieve = _safe_numeric(valid_data["sieve_size"])

    sizes = sorted(
        sieve.dropna().unique().tolist(),
        reverse=True,
    )

    result["sieve_records"] = int(len(valid_data))
    result["sieve_sizes"] = [
        _round_value(size, 3)
        for size in sizes
    ]

    result["smallest_sieve_um"] = _round_value(
        min(sizes),
        3,
    )

    result["largest_sieve_um"] = _round_value(
        max(sizes),
        3,
    )

    result["sieve_size_statistics"] = (
        calculate_descriptive_statistics(sieve)
    )

    if "mass_retained" in valid_data.columns:
        retained = _safe_numeric(
            valid_data["mass_retained"]
        )

        retained_valid = retained.dropna()

        if not retained_valid.empty:
            result["total_retained_mass"] = _round_value(
                retained_valid.sum()
            )

            result["mass_retained_statistics"] = (
                calculate_descriptive_statistics(
                    retained_valid
                )
            )

    # Create a sieve-level summary.
    if "mass_retained" in valid_data.columns:
        grouped = (
            valid_data
            .assign(
                _sieve_size=sieve,
                _mass_retained=_safe_numeric(
                    valid_data["mass_retained"]
                ),
            )
            .groupby("_sieve_size", dropna=True)
        )

        total_mass = _safe_float(
            valid_data["_mass_retained"].sum()
        )

        for sieve_size, group in grouped:
            mass = _safe_float(
                group["_mass_retained"].sum()
            )

            percentage_retained = None

            if (
                mass is not None
                and total_mass is not None
                and total_mass > 0
            ):
                percentage_retained = (
                    mass / total_mass
                ) * 100.0

            result["sieve_summary"].append(
                {
                    "sieve_size_um": _round_value(
                        sieve_size,
                        3,
                    ),
                    "mass_retained_g": _round_value(
                        mass,
                        4,
                    ),
                    "percent_retained": _round_value(
                        percentage_retained,
                        4,
                    ),
                }
            )

        result["sieve_summary"] = sorted(
            result["sieve_summary"],
            key=lambda item: item["sieve_size_um"],
            reverse=True,
        )

    if "sample_id" in valid_data.columns:
        result["sample_count"] = int(
            valid_data["sample_id"].nunique()
        )

    return result


# ---------------------------------------------------------------------
# BENEFICIATION STATISTICS
# ---------------------------------------------------------------------

def calculate_beneficiation_statistics(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Calculate beneficiation performance statistics.

    Calculations:

        Mass recovery (%) =
            concentrate mass / feed mass * 100

        Yield (%) =
            concentrate mass / feed mass * 100

        Grade recovery (%) =
            (concentrate mass * concentrate grade)
            /
            (feed mass * feed grade)
            * 100

        Enrichment ratio =
            concentrate grade / feed grade

        Mass balance difference =
            feed mass
            - concentrate mass
            - tailings mass
            - losses mass

        Balance error (%) =
            absolute balance difference / feed mass * 100

    The calculated values are kept separate from the user's
    dataset-provided 'solid_recovery' column.
    """
    data = _process_level_dataframe(df)

    result: Dict[str, Any] = {
        "records": 0,
        "mean_mass_recovery": None,
        "mean_yield": None,
        "mean_grade_recovery": None,
        "mean_enrichment_ratio": None,
        "mean_balance_difference": None,
        "mean_balance_error_percent": None,
        "provided_solid_recovery_mean": None,
        "mass_recovery_statistics": {},
        "yield_statistics": {},
        "grade_recovery_statistics": {},
        "enrichment_ratio_statistics": {},
        "balance_difference_statistics": {},
        "balance_error_statistics": {},
    }

    if data.empty:
        return result

    feed = (
        _safe_numeric(data["feed_mass"])
        if "feed_mass" in data.columns
        else pd.Series(dtype=float)
    )

    concentrate = (
        _safe_numeric(data["concentrate_mass"])
        if "concentrate_mass" in data.columns
        else pd.Series(dtype=float)
    )

    valid_recovery = (
        feed.notna()
        & concentrate.notna()
        & (feed > 0)
        & (concentrate >= 0)
    )

    if valid_recovery.any():
        mass_recovery = (
            concentrate[valid_recovery]
            / feed[valid_recovery]
        ) * 100.0

        result["records"] = int(
            valid_recovery.sum()
        )

        result["mass_recovery_statistics"] = (
            calculate_descriptive_statistics(
                mass_recovery
            )
        )

        result["yield_statistics"] = (
            calculate_descriptive_statistics(
                mass_recovery
            )
        )

        result["mean_mass_recovery"] = (
            result["mass_recovery_statistics"]["mean"]
        )

        result["mean_yield"] = (
            result["yield_statistics"]["mean"]
        )

    # -------------------------------------------------------------
    # Grade recovery and enrichment
    # -------------------------------------------------------------

    required_grade_columns = [
        "feed_mass",
        "feed_grade",
        "concentrate_mass",
        "concentrate_grade",
    ]

    if all(
        column in data.columns
        for column in required_grade_columns
    ):
        feed_mass = _safe_numeric(
            data["feed_mass"]
        )

        feed_grade = _safe_numeric(
            data["feed_grade"]
        )

        concentrate_mass = _safe_numeric(
            data["concentrate_mass"]
        )

        concentrate_grade = _safe_numeric(
            data["concentrate_grade"]
        )

        valid_grade = (
            feed_mass.notna()
            & feed_grade.notna()
            & concentrate_mass.notna()
            & concentrate_grade.notna()
            & (feed_mass > 0)
            & (feed_grade > 0)
            & (concentrate_mass >= 0)
            & (concentrate_grade >= 0)
        )

        if valid_grade.any():
            grade_recovery = (
                (
                    concentrate_mass[valid_grade]
                    * concentrate_grade[valid_grade]
                )
                /
                (
                    feed_mass[valid_grade]
                    * feed_grade[valid_grade]
                )
            ) * 100.0

            enrichment_ratio = (
                concentrate_grade[valid_grade]
                /
                feed_grade[valid_grade]
            )

            result["grade_recovery_statistics"] = (
                calculate_descriptive_statistics(
                    grade_recovery
                )
            )

            result["enrichment_ratio_statistics"] = (
                calculate_descriptive_statistics(
                    enrichment_ratio
                )
            )

            result["mean_grade_recovery"] = (
                result["grade_recovery_statistics"]["mean"]
            )

            result["mean_enrichment_ratio"] = (
                result["enrichment_ratio_statistics"]["mean"]
            )

    # -------------------------------------------------------------
    # Mass balance
    # -------------------------------------------------------------

    balance_columns = [
        "feed_mass",
        "concentrate_mass",
        "tailings_mass",
        "losses_mass",
    ]

    if all(
        column in data.columns
        for column in balance_columns
    ):
        feed_mass = _safe_numeric(
            data["feed_mass"]
        )

        concentrate_mass = _safe_numeric(
            data["concentrate_mass"]
        )

        tailings_mass = _safe_numeric(
            data["tailings_mass"]
        )

        losses_mass = _safe_numeric(
            data["losses_mass"]
        )

        valid_balance = (
            feed_mass.notna()
            & concentrate_mass.notna()
            & tailings_mass.notna()
            & losses_mass.notna()
            & (feed_mass > 0)
        )

        if valid_balance.any():
            balance_difference = (
                feed_mass[valid_balance]
                - concentrate_mass[valid_balance]
                - tailings_mass[valid_balance]
                - losses_mass[valid_balance]
            )

            balance_error = (
                balance_difference.abs()
                /
                feed_mass[valid_balance]
            ) * 100.0

            result["balance_difference_statistics"] = (
                calculate_descriptive_statistics(
                    balance_difference
                )
            )

            result["balance_error_statistics"] = (
                calculate_descriptive_statistics(
                    balance_error
                )
            )

            result["mean_balance_difference"] = (
                result["balance_difference_statistics"]["mean"]
            )

            result["mean_balance_error_percent"] = (
                result["balance_error_statistics"]["mean"]
            )

    # -------------------------------------------------------------
    # User-provided solid recovery
    # -------------------------------------------------------------

    if "solid_recovery" in data.columns:
        provided = _safe_numeric(
            data["solid_recovery"]
        ).dropna()

        if not provided.empty:
            result["provided_solid_recovery_mean"] = (
                _round_value(provided.mean())
            )

    return result


# ---------------------------------------------------------------------
# MATERIAL BALANCE STATISTICS
# ---------------------------------------------------------------------

def calculate_material_balance_statistics(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Calculate overall material balance statistics.
    """
    data = _process_level_dataframe(df)

    result = {
        "records": 0,
        "mean_feed_mass": None,
        "mean_concentrate_mass": None,
        "mean_tailings_mass": None,
        "mean_losses_mass": None,
        "mean_output_mass": None,
        "mean_balance_difference": None,
        "mean_closure_percent": None,
        "mean_mass_recovery_percent": None,
        "closure_statistics": {},
        "balance_difference_statistics": {},
    }

    if data.empty:
        return result

    required = [
        "feed_mass",
        "concentrate_mass",
        "tailings_mass",
        "losses_mass",
    ]

    if not all(column in data.columns for column in required):
        return result

    feed = _safe_numeric(data["feed_mass"])
    concentrate = _safe_numeric(
        data["concentrate_mass"]
    )
    tailings = _safe_numeric(
        data["tailings_mass"]
    )
    losses = _safe_numeric(
        data["losses_mass"]
    )

    valid = (
        feed.notna()
        & concentrate.notna()
        & tailings.notna()
        & losses.notna()
        & (feed > 0)
    )

    if not valid.any():
        return result

    feed = feed[valid]
    concentrate = concentrate[valid]
    tailings = tailings[valid]
    losses = losses[valid]

    output = concentrate + tailings + losses

    balance_difference = feed - output

    closure = (
        output / feed
    ) * 100.0

    mass_recovery = (
        concentrate / feed
    ) * 100.0

    result["records"] = int(valid.sum())

    result["mean_feed_mass"] = _round_value(
        feed.mean()
    )

    result["mean_concentrate_mass"] = _round_value(
        concentrate.mean()
    )

    result["mean_tailings_mass"] = _round_value(
        tailings.mean()
    )

    result["mean_losses_mass"] = _round_value(
        losses.mean()
    )

    result["mean_output_mass"] = _round_value(
        output.mean()
    )

    result["mean_balance_difference"] = _round_value(
        balance_difference.mean()
    )

    result["mean_closure_percent"] = _round_value(
        closure.mean()
    )

    result["mean_mass_recovery_percent"] = _round_value(
        mass_recovery.mean()
    )

    result["closure_statistics"] = (
        calculate_descriptive_statistics(
            closure
        )
    )

    result["balance_difference_statistics"] = (
        calculate_descriptive_statistics(
            balance_difference
        )
    )

    return result


# ---------------------------------------------------------------------
# GENERIC DATASET STATISTICS
# ---------------------------------------------------------------------

def calculate_statistics(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    MAIN STATISTICS FUNCTION USED BY THE LEBAS DASHBOARD.

    This function combines the major statistical calculations into
    one dictionary.

    Returns a structure containing:

        dataset
        samples
        numeric
        crushing
        grinding
        sieve
        beneficiation
        material_balance

    The function is intentionally defensive so that the Dashboard
    does not crash when a dataset is missing an optional column.
    """
    data = _prepare_dataframe(df)

    if data.empty:
        return {
            "dataset": {
                "rows": 0,
                "columns": 0,
                "process_records": 0,
            },
            "samples": calculate_sample_statistics(data),
            "numeric": {},
            "crushing": calculate_crushing_statistics(data),
            "grinding": calculate_grinding_statistics(data),
            "sieve": calculate_sieve_statistics(data),
            "beneficiation": calculate_beneficiation_statistics(data),
            "material_balance": calculate_material_balance_statistics(
                data
            ),
        }

    process_data = _process_level_dataframe(data)

    numeric_statistics = calculate_column_statistics(
        process_data
    )

    # Add sieve statistics separately because sieve rows represent
    # fractions rather than independent experiments.
    sieve_statistics = calculate_sieve_statistics(data)

    sample_statistics = calculate_sample_statistics(data)

    crushing_statistics = calculate_crushing_statistics(
        data
    )

    grinding_statistics = calculate_grinding_statistics(
        data
    )

    beneficiation_statistics = (
        calculate_beneficiation_statistics(data)
    )

    material_balance_statistics = (
        calculate_material_balance_statistics(data)
    )

    return {
        "dataset": {
            "rows": int(len(data)),
            "columns": int(len(data.columns)),
            "process_records": int(len(process_data)),
            "duplicate_sieve_rows_excluded": int(
                max(len(data) - len(process_data), 0)
            ),
        },

        "samples": sample_statistics,

        "numeric": numeric_statistics,

        "crushing": crushing_statistics,

        "grinding": grinding_statistics,

        "sieve": sieve_statistics,

        "beneficiation": beneficiation_statistics,

        "material_balance": material_balance_statistics,
    }


# ---------------------------------------------------------------------
# COMPATIBILITY FUNCTIONS
# ---------------------------------------------------------------------

def get_statistics(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compatibility alias for calculate_statistics().
    """
    return calculate_statistics(df)


def calculate_dataset_statistics(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compatibility alias for calculate_statistics().
    """
    return calculate_statistics(df)


def get_descriptive_statistics(
    series: pd.Series,
) -> Dict[str, Any]:
    """
    Compatibility wrapper for one-column statistics.
    """
    return calculate_descriptive_statistics(series)


def calculate_mean(
    series: pd.Series,
) -> Optional[float]:
    """
    Calculate the arithmetic mean.
    """
    numeric = _safe_numeric(series).dropna()

    if numeric.empty:
        return None

    return _round_value(numeric.mean())


def calculate_median(
    series: pd.Series,
) -> Optional[float]:
    """
    Calculate the median.
    """
    numeric = _safe_numeric(series).dropna()

    if numeric.empty:
        return None

    return _round_value(numeric.median())


def calculate_standard_deviation(
    series: pd.Series,
) -> Optional[float]:
    """
    Calculate sample standard deviation.

    Uses ddof=1, which is appropriate when the laboratory measurements
    represent a sample of observations.
    """
    numeric = _safe_numeric(series).dropna()

    if numeric.empty:
        return None

    if len(numeric) == 1:
        return 0.0

    return _round_value(
        numeric.std(ddof=1)
    )


def calculate_variance(
    series: pd.Series,
) -> Optional[float]:
    """
    Calculate sample variance.
    """
    numeric = _safe_numeric(series).dropna()

    if numeric.empty:
        return None

    if len(numeric) == 1:
        return 0.0

    return _round_value(
        numeric.var(ddof=1)
    )


def calculate_coefficient_of_variation(
    series: pd.Series,
) -> Optional[float]:
    """
    Calculate coefficient of variation as a percentage.
    """
    numeric = _safe_numeric(series).dropna()

    if numeric.empty:
        return None

    mean = float(numeric.mean())

    if mean == 0:
        return None

    std = (
        float(numeric.std(ddof=1))
        if len(numeric) > 1
        else 0.0
    )

    return _round_value(
        abs(std / mean) * 100.0
    )


# ---------------------------------------------------------------------
# GROUPED STATISTICS
# ---------------------------------------------------------------------

def calculate_grouped_statistics(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
) -> pd.DataFrame:
    """
    Calculate grouped mean, median, standard deviation, minimum,
    maximum and count.

    Example:

        calculate_grouped_statistics(
            df,
            "material_type",
            "concentrate_grade"
        )
    """
    data = _prepare_dataframe(df)

    if data.empty:
        return pd.DataFrame(
            columns=[
                group_column,
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max",
            ]
        )

    if group_column not in data.columns:
        return pd.DataFrame()

    if value_column not in data.columns:
        return pd.DataFrame()

    working = data[
        [group_column, value_column]
    ].copy()

    working[value_column] = _safe_numeric(
        working[value_column]
    )

    working = working.dropna(
        subset=[group_column, value_column]
    )

    if working.empty:
        return pd.DataFrame()

    grouped = (
        working
        .groupby(group_column)[value_column]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
        )
        .reset_index()
    )

    grouped["std"] = grouped["std"].fillna(0.0)

    numeric_result_columns = [
        "mean",
        "median",
        "std",
        "min",
        "max",
    ]

    for column in numeric_result_columns:
        grouped[column] = grouped[column].round(4)

    return grouped


# ---------------------------------------------------------------------
# FORMATTING
# ---------------------------------------------------------------------

def format_statistics_summary(
    statistics: Dict[str, Any],
) -> str:
    """
    Convert the main statistics dictionary into a readable text
    summary for reports or debugging.
    """
    if not statistics:
        return "No statistics available."

    dataset = statistics.get("dataset", {})
    samples = statistics.get("samples", {})
    crushing = statistics.get("crushing", {})
    grinding = statistics.get("grinding", {})
    sieve = statistics.get("sieve", {})
    beneficiation = statistics.get(
        "beneficiation",
        {},
    )
    balance = statistics.get(
        "material_balance",
        {},
    )

    lines = [
        "LEBAS STATISTICAL SUMMARY",
        "=" * 60,
        "",
        "DATASET",
        "-" * 60,
        f"Rows: {dataset.get('rows', 0)}",
        f"Columns: {dataset.get('columns', 0)}",
        f"Process records: {dataset.get('process_records', 0)}",
        "",
        "SAMPLES",
        "-" * 60,
        f"Samples: {samples.get('sample_count', 0)}",
        f"Replicates: {samples.get('replicate_count', 0)}",
        f"Material types: {samples.get('material_type_count', 0)}",
        "",
        "CRUSHING",
        "-" * 60,
        f"Mean reduction ratio: "
        f"{crushing.get('mean_reduction_ratio')}",
        f"Mean size reduction (%): "
        f"{crushing.get('mean_size_reduction_percent')}",
        "",
        "GRINDING",
        "-" * 60,
        f"Mean reduction ratio: "
        f"{grinding.get('mean_reduction_ratio')}",
        f"Mean size reduction (%): "
        f"{grinding.get('mean_size_reduction_percent')}",
        "",
        "SIEVE ANALYSIS",
        "-" * 60,
        f"Smallest sieve (µm): "
        f"{sieve.get('smallest_sieve_um')}",
        f"Largest sieve (µm): "
        f"{sieve.get('largest_sieve_um')}",
        f"Sieve records: "
        f"{sieve.get('sieve_records', 0)}",
        "",
        "BENEFICIATION",
        "-" * 60,
        f"Mean mass recovery (%): "
        f"{beneficiation.get('mean_mass_recovery')}",
        f"Mean grade recovery (%): "
        f"{beneficiation.get('mean_grade_recovery')}",
        f"Mean enrichment ratio: "
        f"{beneficiation.get('mean_enrichment_ratio')}",
        "",
        "MATERIAL BALANCE",
        "-" * 60,
        f"Mean closure (%): "
        f"{balance.get('mean_closure_percent')}",
        f"Mean balance difference: "
        f"{balance.get('mean_balance_difference')}",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------

__all__ = [
    # Main function required by Dashboard
    "calculate_statistics",

    # Main statistics functions
    "calculate_descriptive_statistics",
    "calculate_column_statistics",
    "calculate_sample_statistics",
    "calculate_crushing_statistics",
    "calculate_grinding_statistics",
    "calculate_sieve_statistics",
    "calculate_beneficiation_statistics",
    "calculate_material_balance_statistics",

    # Compatibility functions
    "get_statistics",
    "calculate_dataset_statistics",
    "get_descriptive_statistics",
    "calculate_mean",
    "calculate_median",
    "calculate_standard_deviation",
    "calculate_variance",
    "calculate_coefficient_of_variATION",

    # Grouping and formatting
    "calculate_grouped_statistics",
    "format_statistics_summary",
]