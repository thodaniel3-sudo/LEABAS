"""
LEBAS Material Balance Calculation Module

Laboratory Extraction and Beneficiation Analysis System

This module provides robust material-balance calculations for the LEBAS
application.

Important:
    The LEBAS dataset contains multiple rows for a single experimental run
    because sieve-analysis measurements are stored as separate rows.

    Therefore, process-level values such as feed_mass, concentrate_mass,
    tailings_mass, and losses_mass must NOT be blindly summed across every
    CSV row.

    This module groups records by:
        sample_id + replicate

    and uses one valid process-level record per experimental run.

The module does not modify the original dataset.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Required and optional columns
# ---------------------------------------------------------------------------

IDENTIFICATION_COLUMNS = [
    "sample_id",
    "replicate",
]

MATERIAL_BALANCE_COLUMNS = [
    "feed_mass",
    "concentrate_mass",
    "tailings_mass",
    "losses_mass",
]

OPTIONAL_COLUMNS = [
    "feed_grade",
    "concentrate_grade",
    "tailings_grade",
    "solid_recovery",
]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _safe_numeric(series: pd.Series) -> pd.Series:
    """
    Convert a pandas Series to numeric values safely.

    Invalid values are converted to NaN instead of causing the application
    to crash.
    """
    return pd.to_numeric(series, errors="coerce")


def _clean_dataframe(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Return a cleaned copy of the supplied DataFrame.

    The original DataFrame is never modified.
    """
    if df is None:
        return pd.DataFrame()

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        return df.copy()

    cleaned = df.copy()

    # Remove completely empty rows.
    cleaned = cleaned.dropna(how="all")

    # Normalize column names.
    cleaned.columns = [
        str(column).strip()
        for column in cleaned.columns
    ]

    # Convert material-balance columns to numeric where present.
    for column in MATERIAL_BALANCE_COLUMNS + OPTIONAL_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = _safe_numeric(cleaned[column])

    return cleaned


def _get_group_columns(df: pd.DataFrame) -> list[str]:
    """
    Determine the best available grouping columns.

    Preferred grouping:
        sample_id + replicate

    If replicate is unavailable:
        sample_id

    If sample_id is unavailable:
        use a synthetic single-run grouping.

    This allows the module to remain robust even when a user supplies a
    dataset with some optional identification information missing.
    """
    if "sample_id" in df.columns:
        if "replicate" in df.columns:
            return ["sample_id", "replicate"]

        return ["sample_id"]

    return []


def _first_valid(series: pd.Series) -> Any:
    """
    Return the first non-null value from a Series.

    If no valid value exists, return NaN.
    """
    valid = series.dropna()

    if valid.empty:
        return np.nan

    return valid.iloc[0]


def _aggregate_process_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate process-level values without double-counting sieve rows.

    A single experimental run may appear many times in dataset.csv because
    each sieve size occupies a separate row.

    Example:

        SAMPLE-001, replicate 1, sieve 2000
        SAMPLE-001, replicate 1, sieve 1400
        SAMPLE-001, replicate 1, sieve 1000
        ...

    feed_mass may be repeated on every row.

    Summing those rows would incorrectly multiply the actual feed mass.

    Therefore, this function takes one valid process-level value per
    sample/replicate.

    Returns:
        A DataFrame containing one row per experimental run.
    """

    if df.empty:
        return pd.DataFrame()

    working = df.copy()

    group_columns = _get_group_columns(working)

    # If sample_id does not exist, treat the whole dataframe as one run.
    if not group_columns:
        result = {}

        for column in MATERIAL_BALANCE_COLUMNS + OPTIONAL_COLUMNS:
            if column in working.columns:
                result[column] = _first_valid(working[column])

        return pd.DataFrame([result])

    # Preserve first occurrence of each experimental run.
    #
    # We cannot simply use drop_duplicates() because some rows may have
    # missing process values while another row in the same run contains the
    # valid value.
    records = []

    grouped = working.groupby(
        group_columns,
        dropna=False,
        sort=False
    )

    for group_key, group in grouped:

        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        record: Dict[str, Any] = {}

        for index, column in enumerate(group_columns):
            record[column] = group_key[index]

        for column in MATERIAL_BALANCE_COLUMNS + OPTIONAL_COLUMNS:
            if column in group.columns:
                record[column] = _first_valid(group[column])
            else:
                record[column] = np.nan

        records.append(record)

    return pd.DataFrame(records)


def _calculate_single_balance(
    row: pd.Series,
) -> Dict[str, Any]:
    """
    Calculate material balance for a single experimental run.
    """

    feed_mass = row.get("feed_mass", np.nan)
    concentrate_mass = row.get("concentrate_mass", np.nan)
    tailings_mass = row.get("tailings_mass", np.nan)
    losses_mass = row.get("losses_mass", np.nan)

    # Convert values safely.
    try:
        feed_mass = float(feed_mass)
    except (TypeError, ValueError):
        feed_mass = np.nan

    try:
        concentrate_mass = float(concentrate_mass)
    except (TypeError, ValueError):
        concentrate_mass = np.nan

    try:
        tailings_mass = float(tailings_mass)
    except (TypeError, ValueError):
        tailings_mass = np.nan

    try:
        losses_mass = float(losses_mass)
    except (TypeError, ValueError):
        losses_mass = np.nan

    # Output mass can only be calculated when all required output streams
    # are available.
    output_values = [
        concentrate_mass,
        tailings_mass,
        losses_mass,
    ]

    if all(pd.notna(value) for value in output_values):
        output_mass = (
            concentrate_mass
            + tailings_mass
            + losses_mass
        )
    else:
        output_mass = np.nan

    # Difference between feed and calculated output.
    if pd.notna(feed_mass) and pd.notna(output_mass):
        balance_difference = feed_mass - output_mass
    else:
        balance_difference = np.nan

    # Material balance closure.
    if (
        pd.notna(feed_mass)
        and feed_mass != 0
        and pd.notna(output_mass)
    ):
        balance_closure = (
            output_mass / feed_mass
        ) * 100.0
    else:
        balance_closure = np.nan

    # Concentrate mass recovery / concentrate yield.
    if (
        pd.notna(feed_mass)
        and feed_mass != 0
        and pd.notna(concentrate_mass)
    ):
        mass_recovery = (
            concentrate_mass / feed_mass
        ) * 100.0
    else:
        mass_recovery = np.nan

    # Optional reported solid recovery from the dataset.
    reported_solid_recovery = row.get(
        "solid_recovery",
        np.nan
    )

    try:
        if pd.notna(reported_solid_recovery):
            reported_solid_recovery = float(
                reported_solid_recovery
            )
    except (TypeError, ValueError):
        reported_solid_recovery = np.nan

    return {
        "feed_mass_g": feed_mass,
        "concentrate_mass_g": concentrate_mass,
        "tailings_mass_g": tailings_mass,
        "losses_mass_g": losses_mass,
        "output_mass_g": output_mass,
        "balance_difference_g": balance_difference,
        "balance_closure_percent": balance_closure,
        "mass_recovery_percent": mass_recovery,
        "reported_solid_recovery_percent": reported_solid_recovery,
    }


# ---------------------------------------------------------------------------
# Public calculation functions
# ---------------------------------------------------------------------------

def calculate_material_balance(
    df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Calculate material balance for every experimental run in the dataset.

    Parameters
    ----------
    df:
        LEBAS dataset as a pandas DataFrame.

    Returns
    -------
    pandas.DataFrame
        One row per sample/replicate containing material-balance results.

    Output columns include:

        sample_id
        replicate
        feed_mass_g
        concentrate_mass_g
        tailings_mass_g
        losses_mass_g
        output_mass_g
        balance_difference_g
        balance_closure_percent
        mass_recovery_percent
        reported_solid_recovery_percent

    Notes
    -----
    The function prevents double-counting caused by repeated sieve-analysis
    rows.
    """

    cleaned = _clean_dataframe(df)

    if cleaned.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "replicate",
                "feed_mass_g",
                "concentrate_mass_g",
                "tailings_mass_g",
                "losses_mass_g",
                "output_mass_g",
                "balance_difference_g",
                "balance_closure_percent",
                "mass_recovery_percent",
                "reported_solid_recovery_percent",
            ]
        )

    # Check whether the fundamental columns exist.
    missing = [
        column
        for column in MATERIAL_BALANCE_COLUMNS
        if column not in cleaned.columns
    ]

    if missing:
        logger.warning(
            "Material balance columns missing: %s",
            ", ".join(missing),
        )

    aggregated = _aggregate_process_values(cleaned)

    if aggregated.empty:
        return pd.DataFrame()

    results = []

    for _, row in aggregated.iterrows():

        balance = _calculate_single_balance(row)

        result: Dict[str, Any] = {}

        if "sample_id" in aggregated.columns:
            result["sample_id"] = row.get(
                "sample_id",
                np.nan,
            )

        if "replicate" in aggregated.columns:
            result["replicate"] = row.get(
                "replicate",
                np.nan,
            )

        result.update(balance)

        # Preserve useful optional laboratory information.
        if "feed_grade" in aggregated.columns:
            result["feed_grade_percent"] = row.get(
                "feed_grade",
                np.nan,
            )

        if "concentrate_grade" in aggregated.columns:
            result["concentrate_grade_percent"] = row.get(
                "concentrate_grade",
                np.nan,
            )

        if "tailings_grade" in aggregated.columns:
            result["tailings_grade_percent"] = row.get(
                "tailings_grade",
                np.nan,
            )

        results.append(result)

    result_df = pd.DataFrame(results)

    # Keep numeric output clean.
    numeric_columns = [
        "feed_mass_g",
        "concentrate_mass_g",
        "tailings_mass_g",
        "losses_mass_g",
        "output_mass_g",
        "balance_difference_g",
        "balance_closure_percent",
        "mass_recovery_percent",
        "reported_solid_recovery_percent",
        "feed_grade_percent",
        "concentrate_grade_percent",
        "tailings_grade_percent",
    ]

    for column in numeric_columns:
        if column in result_df.columns:
            result_df[column] = pd.to_numeric(
                result_df[column],
                errors="coerce",
            )

    return result_df


def calculate_sample_material_balance(
    df: Optional[pd.DataFrame],
    sample_id: Any,
    replicate: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Calculate material balance for one sample.

    Parameters
    ----------
    df:
        LEBAS dataset.

    sample_id:
        Sample identifier.

    replicate:
        Optional replicate identifier.

    Returns
    -------
    dict
        Material-balance result for the selected experimental run.

    Raises
    ------
    ValueError
        If the requested sample does not exist.
    """

    cleaned = _clean_dataframe(df)

    if cleaned.empty:
        raise ValueError("The dataset is empty.")

    if "sample_id" not in cleaned.columns:
        raise ValueError(
            "The dataset does not contain a 'sample_id' column."
        )

    # Compare as strings to avoid issues such as:
    # 1 vs "1"
    # SAMPLE-001 vs categorical values
    sample_mask = (
        cleaned["sample_id"]
        .astype(str)
        .str.strip()
        == str(sample_id).strip()
    )

    sample_df = cleaned.loc[sample_mask].copy()

    if sample_df.empty:
        raise ValueError(
            f"Sample '{sample_id}' was not found in the dataset."
        )

    if (
        replicate is not None
        and "replicate" in sample_df.columns
    ):
        replicate_mask = (
            sample_df["replicate"]
            .astype(str)
            .str.strip()
            == str(replicate).strip()
        )

        sample_df = sample_df.loc[replicate_mask].copy()

        if sample_df.empty:
            raise ValueError(
                f"Replicate '{replicate}' was not found "
                f"for sample '{sample_id}'."
            )

    balance_df = calculate_material_balance(sample_df)

    if balance_df.empty:
        raise ValueError(
            f"Material balance could not be calculated "
            f"for sample '{sample_id}'."
        )

    # Return the first run because the function has already filtered to the
    # requested sample/replicate.
    result = balance_df.iloc[0].to_dict()

    return result


def calculate_total_material_balance(
    df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """
    Calculate overall material-balance statistics without double-counting
    repeated sieve rows.

    Returns
    -------
    dict
        Overall feed, concentrate, tailings, losses, output, difference,
        closure and mass recovery.

    Important:
        Only valid, non-null process-level values are included in totals.
    """

    balance_df = calculate_material_balance(df)

    if balance_df.empty:
        return {
            "number_of_runs": 0,
            "total_feed_mass_g": np.nan,
            "total_concentrate_mass_g": np.nan,
            "total_tailings_mass_g": np.nan,
            "total_losses_mass_g": np.nan,
            "total_output_mass_g": np.nan,
            "total_balance_difference_g": np.nan,
            "overall_balance_closure_percent": np.nan,
            "overall_mass_recovery_percent": np.nan,
        }

    feed = balance_df["feed_mass_g"]
    concentrate = balance_df["concentrate_mass_g"]
    tailings = balance_df["tailings_mass_g"]
    losses = balance_df["losses_mass_g"]

    total_feed = feed.sum(min_count=1)
    total_concentrate = concentrate.sum(min_count=1)
    total_tailings = tailings.sum(min_count=1)
    total_losses = losses.sum(min_count=1)

    output_components = [
        total_concentrate,
        total_tailings,
        total_losses,
    ]

    if all(pd.notna(value) for value in output_components):
        total_output = (
            total_concentrate
            + total_tailings
            + total_losses
        )
    else:
        total_output = np.nan

    if pd.notna(total_feed) and pd.notna(total_output):
        total_difference = (
            total_feed - total_output
        )
    else:
        total_difference = np.nan

    if (
        pd.notna(total_feed)
        and total_feed != 0
        and pd.notna(total_output)
    ):
        overall_closure = (
            total_output / total_feed
        ) * 100.0
    else:
        overall_closure = np.nan

    if (
        pd.notna(total_feed)
        and total_feed != 0
        and pd.notna(total_concentrate)
    ):
        overall_recovery = (
            total_concentrate / total_feed
        ) * 100.0
    else:
        overall_recovery = np.nan

    return {
        "number_of_runs": len(balance_df),
        "total_feed_mass_g": total_feed,
        "total_concentrate_mass_g": total_concentrate,
        "total_tailings_mass_g": total_tailings,
        "total_losses_mass_g": total_losses,
        "total_output_mass_g": total_output,
        "total_balance_difference_g": total_difference,
        "overall_balance_closure_percent": overall_closure,
        "overall_mass_recovery_percent": overall_recovery,
    }


def get_material_balance_summary(
    df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """
    Return a convenient summary dictionary for use by the UI and reports.

    This function is intentionally lightweight and safe to call from
    dashboard, results and report-generation modules.
    """

    totals = calculate_total_material_balance(df)

    summary = {
        "number_of_experimental_runs": totals[
            "number_of_runs"
        ],
        "total_feed_mass_g": totals[
            "total_feed_mass_g"
        ],
        "total_concentrate_mass_g": totals[
            "total_concentrate_mass_g"
        ],
        "total_tailings_mass_g": totals[
            "total_tailings_mass_g"
        ],
        "total_losses_mass_g": totals[
            "total_losses_mass_g"
        ],
        "total_output_mass_g": totals[
            "total_output_mass_g"
        ],
        "balance_difference_g": totals[
            "total_balance_difference_g"
        ],
        "balance_closure_percent": totals[
            "overall_balance_closure_percent"
        ],
        "mass_recovery_percent": totals[
            "overall_mass_recovery_percent"
        ],
    }

    return summary


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_material_balance(
    df: Optional[pd.DataFrame],
    tolerance_percent: float = 5.0,
) -> pd.DataFrame:
    """
    Evaluate whether each experimental run has an acceptable material
    balance closure.

    Parameters
    ----------
    df:
        LEBAS dataset.

    tolerance_percent:
        Maximum absolute deviation from 100% closure considered acceptable.

        Example:
            tolerance_percent = 5

        means closure from 95% to 105% is considered acceptable.

    Returns
    -------
    pandas.DataFrame
        Material-balance results with additional validation columns.
    """

    if tolerance_percent < 0:
        raise ValueError(
            "tolerance_percent cannot be negative."
        )

    results = calculate_material_balance(df)

    if results.empty:
        return results

    results = results.copy()

    results["closure_deviation_percent"] = np.nan
    results["balance_status"] = "Not Available"

    valid_mask = results[
        "balance_closure_percent"
    ].notna()

    results.loc[
        valid_mask,
        "closure_deviation_percent"
    ] = (
        results.loc[
            valid_mask,
            "balance_closure_percent"
        ] - 100.0
    ).abs()

    acceptable_mask = (
        valid_mask
        & (
            results[
                "closure_deviation_percent"
            ]
            <= tolerance_percent
        )
    )

    results.loc[
        acceptable_mask,
        "balance_status"
    ] = "Acceptable"

    results.loc[
        valid_mask & ~acceptable_mask,
        "balance_status"
    ] = "Check Balance"

    return results


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

def calculate_balance(
    df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Backward-compatible alias for calculate_material_balance().
    """
    return calculate_material_balance(df)


def get_balance_summary(
    df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """
    Backward-compatible alias for get_material_balance_summary().
    """
    return get_material_balance_summary(df)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "calculate_material_balance",
    "calculate_sample_material_balance",
    "calculate_total_material_balance",
    "get_material_balance_summary",
    "validate_material_balance",
    "calculate_balance",
    "get_balance_summary",
]