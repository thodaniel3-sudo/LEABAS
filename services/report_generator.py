"""
LEBAS - Laboratory Report Generator

File:
    services/report_generator.py

Purpose:
    Generates professional laboratory analysis reports from the
    current LEBAS dataset.

Design principles:
    - Completely dataset-driven.
    - No experimental results are hard-coded.
    - Repeated sieve rows are not double-counted in process summaries.
    - Sieve sizes are read dynamically from the dataset.
    - The real laboratory dataset can use 150 µm as its smallest sieve.
    - Missing measurements are reported rather than invented.
    - Reports can be generated as TXT and Excel-compatible analysis data.
    - The module can be used by the Reports GUI.

Main report sections:
    1. Dataset information
    2. Sample information
    3. Crushing analysis
    4. Grinding analysis
    5. Sieve analysis
    6. Beneficiation analysis
    7. Material balance
    8. Data quality
    9. Overall observations

Public functions:
    generate_report()
    generate_text_report()
    generate_summary()
    generate_sample_summary()
    save_report()
    save_text_report()
"""


from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# ============================================================================
# PROJECT DIRECTORIES
# ============================================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent

REPORTS_DIR = PROJECT_ROOT / "reports"


# ============================================================================
# PROCESS COLUMNS
# ============================================================================

PROCESS_COLUMNS = [
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
# GENERAL HELPERS
# ============================================================================

def _safe_numeric(
    series: pd.Series,
) -> pd.Series:
    """Convert a Series to numeric values safely."""

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def _mean(
    series: Optional[pd.Series],
) -> Optional[float]:
    """Return a safe mean."""

    if series is None:
        return None

    values = _safe_numeric(
        series
    ).dropna()

    if values.empty:
        return None

    return float(
        values.mean()
    )


def _sum(
    series: Optional[pd.Series],
) -> Optional[float]:
    """Return a safe sum."""

    if series is None:
        return None

    values = _safe_numeric(
        series
    ).dropna()

    if values.empty:
        return None

    return float(
        values.sum()
    )


def _minimum(
    series: Optional[pd.Series],
) -> Optional[float]:
    """Return safe minimum."""

    if series is None:
        return None

    values = _safe_numeric(
        series
    ).dropna()

    if values.empty:
        return None

    return float(
        values.min()
    )


def _maximum(
    series: Optional[pd.Series],
) -> Optional[float]:
    """Return safe maximum."""

    if series is None:
        return None

    values = _safe_numeric(
        series
    ).dropna()

    if values.empty:
        return None

    return float(
        values.max()
    )


def _format_number(
    value: Any,
    decimals: int = 2,
) -> str:
    """Format numeric values for report text."""

    if value is None:
        return "N/A"

    try:

        numeric = float(
            value
        )

        if not np.isfinite(
            numeric
        ):
            return "N/A"

        return f"{numeric:.{decimals}f}"

    except (
        ValueError,
        TypeError,
    ):

        return str(
            value
        )


def _format_percent(
    value: Any,
    decimals: int = 2,
) -> str:
    """Format a percentage value."""

    formatted = _format_number(
        value,
        decimals,
    )

    if formatted == "N/A":
        return formatted

    return f"{formatted}%"


def _clean_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return a clean copy without changing the source DataFrame."""

    if df is None:

        raise ValueError(
            "Dataset cannot be None."
        )

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        raise TypeError(
            "Dataset must be a pandas DataFrame."
        )

    if df.empty:

        raise ValueError(
            "Dataset is empty."
        )

    result = df.copy()

    result.columns = [
        str(column).strip()
        for column in result.columns
    ]

    return result


# ============================================================================
# PROCESS-LEVEL REDUCTION
# ============================================================================

def _prepare_process_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reduce repeated sieve rows to one process-level row per
    sample_id + replicate.

    This is critical for avoiding double counting.
    """

    working = _clean_dataframe(
        df
    )

    if not {
        "sample_id",
        "replicate",
    }.issubset(
        working.columns
    ):

        return working

    available = [
        column
        for column in PROCESS_COLUMNS
        if column in working.columns
    ]

    if available:

        working[
            "_completeness"
        ] = working[
            available
        ].notna().sum(axis=1)

        working = working.sort_values(
            "_completeness",
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

    process = process.drop(
        columns=[
            "_completeness"
        ],
        errors="ignore",
    )

    return process.reset_index(
        drop=True
    )


# ============================================================================
# CALCULATED BENEFICIATION DATA
# ============================================================================

def _calculate_beneficiation_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate beneficiation indicators on process-level records.
    """

    process = _prepare_process_dataframe(
        df
    )

    numeric_columns = [
        "feed_mass",
        "feed_grade",
        "concentrate_mass",
        "concentrate_grade",
        "tailings_mass",
        "tailings_grade",
        "losses_mass",
        "solid_recovery",
    ]

    for column in numeric_columns:

        if column in process.columns:

            process[
                column
            ] = _safe_numeric(
                process[
                    column
                ]
            )

    if {
        "feed_mass",
        "concentrate_mass",
    }.issubset(
        process.columns
    ):

        feed = process[
            "feed_mass"
        ]

        concentrate = process[
            "concentrate_mass"
        ]

        process[
            "mass_recovery"
        ] = np.where(
            feed > 0,
            concentrate
            / feed
            * 100.0,
            np.nan,
        )

        process[
            "yield"
        ] = process[
            "mass_recovery"
        ]

    if {
        "feed_mass",
        "feed_grade",
        "concentrate_mass",
        "concentrate_grade",
    }.issubset(
        process.columns
    ):

        denominator = (
            process[
                "feed_mass"
            ]
            * process[
                "feed_grade"
            ]
        )

        numerator = (
            process[
                "concentrate_mass"
            ]
            * process[
                "concentrate_grade"
            ]
        )

        process[
            "grade_recovery"
        ] = np.where(
            denominator != 0,
            numerator
            / denominator
            * 100.0,
            np.nan,
        )

        process[
            "enrichment_ratio"
        ] = np.where(
            process[
                "feed_grade"
            ] != 0,
            process[
                "concentrate_grade"
            ]
            / process[
                "feed_grade"
            ],
            np.nan,
        )

    if "feed_mass" in process.columns:

        concentrate = (
            process[
                "concentrate_mass"
            ]
            if "concentrate_mass"
            in process.columns
            else pd.Series(
                0.0,
                index=process.index,
            )
        )

        tailings = (
            process[
                "tailings_mass"
            ]
            if "tailings_mass"
            in process.columns
            else pd.Series(
                0.0,
                index=process.index,
            )
        )

        losses = (
            process[
                "losses_mass"
            ]
            if "losses_mass"
            in process.columns
            else pd.Series(
                0.0,
                index=process.index,
            )
        )

        process[
            "output_mass"
        ] = (
            concentrate.fillna(0)
            + tailings.fillna(0)
            + losses.fillna(0)
        )

        process[
            "mass_balance"
        ] = (
            process[
                "feed_mass"
            ]
            - process[
                "output_mass"
            ]
        )

        process[
            "closure_percent"
        ] = np.where(
            process[
                "feed_mass"
            ] != 0,
            process[
                "output_mass"
            ]
            / process[
                "feed_mass"
            ]
            * 100.0,
            np.nan,
        )

        process[
            "balance_error_percent"
        ] = np.where(
            process[
                "feed_mass"
            ] != 0,
            process[
                "mass_balance"
            ].abs()
            / process[
                "feed_mass"
            ].abs()
            * 100.0,
            np.nan,
        )

    return process


# ============================================================================
# SIEVE DATA
# ============================================================================

def _prepare_sieve_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare sieve records and calculate percentage retained and passing.

    Sieve sizes are dynamically read from the dataset.
    """

    working = _clean_dataframe(
        df
    )

    required = {
        "sample_id",
        "replicate",
        "sieve_size",
        "mass_retained",
    }

    if not required.issubset(
        working.columns
    ):

        return pd.DataFrame()

    working[
        "sieve_size"
    ] = _safe_numeric(
        working[
            "sieve_size"
        ]
    )

    working[
        "mass_retained"
    ] = _safe_numeric(
        working[
            "mass_retained"
        ]
    )

    working = working[
        working[
            "sieve_size"
        ] > 0
    ].copy()

    if working.empty:
        return working

    group_columns = [
        "sample_id",
        "replicate",
    ]

    total_retained = (
        working
        .groupby(
            group_columns,
            dropna=False,
        )[
            "mass_retained"
        ]
        .transform(
            "sum"
        )
    )

    working[
        "percent_retained"
    ] = np.where(
        total_retained != 0,
        working[
            "mass_retained"
        ]
        / total_retained
        * 100.0,
        np.nan,
    )

    working = working.sort_values(
        [
            "sample_id",
            "replicate",
            "sieve_size",
        ],
        ascending=[
            True,
            True,
            False,
        ],
        kind="stable",
    )

    working[
        "cumulative_percent_retained"
    ] = (
        working
        .groupby(
            group_columns,
            dropna=False,
        )[
            "percent_retained"
        ]
        .cumsum()
    )

    working[
        "percent_passing"
    ] = (
        100.0
        - working[
            "cumulative_percent_retained"
        ]
    )

    return working.reset_index(
        drop=True
    )


# ============================================================================
# DATASET SUMMARY
# ============================================================================

def generate_summary(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Generate a comprehensive summary dictionary.

    The returned dictionary is designed to be reusable by the GUI,
    reports, charts, and export modules.
    """

    data = _clean_dataframe(
        df
    )

    process = _calculate_beneficiation_data(
        data
    )

    summary: Dict[
        str,
        Any,
    ] = {}

    # ------------------------------------------------------------------------
    # GENERAL DATA
    # ------------------------------------------------------------------------

    summary[
        "row_count"
    ] = int(
        len(data)
    )

    summary[
        "column_count"
    ] = int(
        len(data.columns)
    )

    if "sample_id" in data.columns:

        summary[
            "sample_count"
        ] = int(
            data[
                "sample_id"
            ]
            .dropna()
            .astype(str)
            .str.strip()
            .nunique()
        )

    else:

        summary[
            "sample_count"
        ] = 0

    if "replicate" in data.columns:

        summary[
            "replicate_count"
        ] = int(
            data[
                "replicate"
            ]
            .dropna()
            .astype(str)
            .str.strip()
            .nunique()
        )

    else:

        summary[
            "replicate_count"
        ] = 0

    if "material_type" in data.columns:

        materials = (
            data[
                "material_type"
            ]
            .dropna()
            .astype(str)
            .str.strip()
        )

        summary[
            "material_types"
        ] = sorted(
            materials.unique().tolist()
        )

    else:

        summary[
            "material_types"
        ] = []

    # ------------------------------------------------------------------------
    # DATE RANGE
    # ------------------------------------------------------------------------

    if "experiment_date" in data.columns:

        dates = pd.to_datetime(
            data[
                "experiment_date"
            ],
            errors="coerce",
        ).dropna()

        if not dates.empty:

            summary[
                "start_date"
            ] = dates.min()

            summary[
                "end_date"
            ] = dates.max()

        else:

            summary[
                "start_date"
            ] = None

            summary[
                "end_date"
            ] = None

    else:

        summary[
            "start_date"
        ] = None

        summary[
            "end_date"
        ] = None

    # ------------------------------------------------------------------------
    # FEED DATA
    # ------------------------------------------------------------------------

    if "feed_mass" in process.columns:

        summary[
            "total_feed_mass"
        ] = _sum(
            process[
                "feed_mass"
            ]
        )

        summary[
            "average_feed_mass"
        ] = _mean(
            process[
                "feed_mass"
            ]
        )

    else:

        summary[
            "total_feed_mass"
        ] = None

        summary[
            "average_feed_mass"
        ] = None

    if "feed_grade" in process.columns:

        summary[
            "average_feed_grade"
        ] = _mean(
            process[
                "feed_grade"
            ]
        )

    else:

        summary[
            "average_feed_grade"
        ] = None

    # ------------------------------------------------------------------------
    # CRUSHING
    # ------------------------------------------------------------------------

    if "crush_time" in process.columns:

        summary[
            "average_crush_time"
        ] = _mean(
            process[
                "crush_time"
            ]
        )

    else:

        summary[
            "average_crush_time"
        ] = None

    if {
        "crush_feed_size",
        "crush_product_size",
    }.issubset(
        process.columns
    ):

        feed_size = _safe_numeric(
            process[
                "crush_feed_size"
            ]
        )

        product_size = _safe_numeric(
            process[
                "crush_product_size"
            ]
        )

        ratio = np.where(
            product_size > 0,
            feed_size
            / product_size,
            np.nan,
        )

        reduction = np.where(
            feed_size > 0,
            (
                feed_size
                - product_size
            )
            / feed_size
            * 100.0,
            np.nan,
        )

        summary[
            "average_crush_reduction_ratio"
        ] = _mean(
            pd.Series(
                ratio
            )
        )

        summary[
            "average_crush_size_reduction"
        ] = _mean(
            pd.Series(
                reduction
            )
        )

    else:

        summary[
            "average_crush_reduction_ratio"
        ] = None

        summary[
            "average_crush_size_reduction"
        ] = None

    # ------------------------------------------------------------------------
    # GRINDING
    # ------------------------------------------------------------------------

    if "grinding_time" in process.columns:

        summary[
            "average_grinding_time"
        ] = _mean(
            process[
                "grinding_time"
            ]
        )

    else:

        summary[
            "average_grinding_time"
        ] = None

    if {
        "grinding_feed_size",
        "grinding_product_size",
    }.issubset(
        process.columns
    ):

        feed_size = _safe_numeric(
            process[
                "grinding_feed_size"
            ]
        )

        product_size = _safe_numeric(
            process[
                "grinding_product_size"
            ]
        )

        ratio = np.where(
            product_size > 0,
            feed_size
            / product_size,
            np.nan,
        )

        reduction = np.where(
            feed_size > 0,
            (
                feed_size
                - product_size
            )
            / feed_size
            * 100.0,
            np.nan,
        )

        summary[
            "average_grinding_reduction_ratio"
        ] = _mean(
            pd.Series(
                ratio
            )
        )

        summary[
            "average_grinding_size_reduction"
        ] = _mean(
            pd.Series(
                reduction
            )
        )

    else:

        summary[
            "average_grinding_reduction_ratio"
        ] = None

        summary[
            "average_grinding_size_reduction"
        ] = None

    # ------------------------------------------------------------------------
    # SIEVE
    # ------------------------------------------------------------------------

    sieve = _prepare_sieve_data(
        data
    )

    summary[
        "sieve_row_count"
    ] = int(
        len(sieve)
    )

    if not sieve.empty:

        summary[
            "sieve_size_count"
        ] = int(
            sieve[
                "sieve_size"
            ]
            .nunique()
        )

        summary[
            "minimum_sieve_size"
        ] = _minimum(
            sieve[
                "sieve_size"
            ]
        )

        summary[
            "maximum_sieve_size"
        ] = _maximum(
            sieve[
                "sieve_size"
            ]
        )

        summary[
            "sieve_sizes"
        ] = sorted(
            [
                float(value)
                for value in sieve[
                    "sieve_size"
                ].dropna().unique()
            ],
            reverse=True,
        )

    else:

        summary[
            "sieve_size_count"
        ] = 0

        summary[
            "minimum_sieve_size"
        ] = None

        summary[
            "maximum_sieve_size"
        ] = None

        summary[
            "sieve_sizes"
        ] = []

    # ------------------------------------------------------------------------
    # BENEFICIATION
    # ------------------------------------------------------------------------

    if "concentrate_mass" in process.columns:

        summary[
            "total_concentrate_mass"
        ] = _sum(
            process[
                "concentrate_mass"
            ]
        )

    else:

        summary[
            "total_concentrate_mass"
        ] = None

    if "tailings_mass" in process.columns:

        summary[
            "total_tailings_mass"
        ] = _sum(
            process[
                "tailings_mass"
            ]
        )

    else:

        summary[
            "total_tailings_mass"
        ] = None

    if "mass_recovery" in process.columns:

        summary[
            "average_mass_recovery"
        ] = _mean(
            process[
                "mass_recovery"
            ]
        )

    else:

        summary[
            "average_mass_recovery"
        ] = None

    if "yield" in process.columns:

        summary[
            "average_yield"
        ] = _mean(
            process[
                "yield"
            ]
        )

    else:

        summary[
            "average_yield"
        ] = None

    if "grade_recovery" in process.columns:

        summary[
            "average_grade_recovery"
        ] = _mean(
            process[
                "grade_recovery"
            ]
        )

    else:

        summary[
            "average_grade_recovery"
        ] = None

    if "enrichment_ratio" in process.columns:

        summary[
            "average_enrichment_ratio"
        ] = _mean(
            process[
                "enrichment_ratio"
            ]
        )

    else:

        summary[
            "average_enrichment_ratio"
        ] = None

    if "concentrate_grade" in process.columns:

        summary[
            "average_concentrate_grade"
        ] = _mean(
            process[
                "concentrate_grade"
            ]
        )

    else:

        summary[
            "average_concentrate_grade"
        ] = None

    # ------------------------------------------------------------------------
    # MATERIAL BALANCE
    # ------------------------------------------------------------------------

    if "closure_percent" in process.columns:

        summary[
            "average_closure"
        ] = _mean(
            process[
                "closure_percent"
            ]
        )

    else:

        summary[
            "average_closure"
        ] = None

    if "balance_error_percent" in process.columns:

        summary[
            "average_balance_error"
        ] = _mean(
            process[
                "balance_error_percent"
            ]
        )

        summary[
            "maximum_balance_error"
        ] = _maximum(
            process[
                "balance_error_percent"
            ]
        )

    else:

        summary[
            "average_balance_error"
        ] = None

        summary[
            "maximum_balance_error"
        ] = None

    return summary


# ============================================================================
# SAMPLE SUMMARY
# ============================================================================

def generate_sample_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate one summary row for each sample/replicate.
    """

    process = _calculate_beneficiation_data(
        df
    )

    if not {
        "sample_id",
        "replicate",
    }.issubset(
        process.columns
    ):

        return process

    preferred_columns = [
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
        "crush_feed_size",
        "crush_product_size",
        "grinding_mill_type",
        "grinding_time",
        "grinding_feed_size",
        "grinding_product_size",
        "beneficiation_method",
        "beneficiation_time",
        "concentrate_mass",
        "concentrate_grade",
        "tailings_mass",
        "tailings_grade",
        "losses_mass",
        "mass_recovery",
        "yield",
        "grade_recovery",
        "enrichment_ratio",
        "mass_balance",
        "closure_percent",
        "balance_error_percent",
        "solid_recovery",
    ]

    available = [
        column
        for column in preferred_columns
        if column in process.columns
    ]

    return process[
        available
    ].copy()


# ============================================================================
# TEXT REPORT
# ============================================================================

def generate_text_report(
    df: pd.DataFrame,
    title: str = (
        "LEBAS LABORATORY ANALYSIS REPORT"
    ),
) -> str:
    """
    Generate a professional plain-text laboratory report.
    """

    data = _clean_dataframe(
        df
    )

    process = _calculate_beneficiation_data(
        data
    )

    sieve = _prepare_sieve_data(
        data
    )

    summary = generate_summary(
        data
    )

    lines: List[str] = []

    separator = "=" * 78
    sub_separator = "-" * 78

    lines.append(
        separator
    )

    lines.append(
        title.center(78)
    )

    lines.append(
        separator
    )

    lines.append(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    lines.append(
        "System: LEBAS - Laboratory Extraction and Beneficiation Analysis System"
    )

    lines.append("")

    # ========================================================================
    # 1. DATASET OVERVIEW
    # ========================================================================

    lines.append(
        "1. DATASET OVERVIEW"
    )

    lines.append(
        sub_separator
    )

    lines.append(
        f"Total dataset rows: {summary['row_count']}"
    )

    lines.append(
        f"Total columns: {summary['column_count']}"
    )

    lines.append(
        f"Unique samples: {summary['sample_count']}"
    )

    lines.append(
        f"Unique replicate identifiers: {summary['replicate_count']}"
    )

    materials = summary.get(
        "material_types",
        [],
    )

    lines.append(
        "Material types: "
        + (
            ", ".join(materials)
            if materials
            else "N/A"
        )
    )

    if (
        summary.get("start_date")
        is not None
    ):

        lines.append(
            "Experiment date range: "
            f"{summary['start_date'].strftime('%Y-%m-%d')} "
            f"to "
            f"{summary['end_date'].strftime('%Y-%m-%d')}"
        )

    else:

        lines.append(
            "Experiment date range: N/A"
        )

    lines.append(
        f"Total feed mass: {_format_number(summary['total_feed_mass'])}"
    )

    lines.append(
        f"Average feed mass: {_format_number(summary['average_feed_mass'])}"
    )

    lines.append(
        f"Average feed grade: {_format_number(summary['average_feed_grade'])}"
    )

    lines.append("")

    # ========================================================================
    # 2. CRUSHING
    # ========================================================================

    lines.append(
        "2. CRUSHING ANALYSIS"
    )

    lines.append(
        sub_separator
    )

    lines.append(
        f"Average crushing time: "
        f"{_format_number(summary['average_crush_time'])}"
    )

    lines.append(
        f"Average crushing reduction ratio: "
        f"{_format_number(summary['average_crush_reduction_ratio'])}"
    )

    lines.append(
        f"Average crushing size reduction: "
        f"{_format_percent(summary['average_crush_size_reduction'])}"
    )

    if {
        "crush_feed_size",
        "crush_product_size",
    }.issubset(
        process.columns
    ):

        valid = process[
            [
                "crush_feed_size",
                "crush_product_size",
            ]
        ].dropna()

        lines.append(
            f"Crushing records evaluated: {len(valid)}"
        )

    else:

        lines.append(
            "Crushing records evaluated: N/A"
        )

    lines.append("")

    # ========================================================================
    # 3. GRINDING
    # ========================================================================

    lines.append(
        "3. GRINDING ANALYSIS"
    )

    lines.append(
        sub_separator
    )

    lines.append(
        f"Average grinding time: "
        f"{_format_number(summary['average_grinding_time'])}"
    )

    lines.append(
        f"Average grinding reduction ratio: "
        f"{_format_number(summary['average_grinding_reduction_ratio'])}"
    )

    lines.append(
        f"Average grinding size reduction: "
        f"{_format_percent(summary['average_grinding_size_reduction'])}"
    )

    if {
        "grinding_feed_size",
        "grinding_product_size",
    }.issubset(
        process.columns
    ):

        valid = process[
            [
                "grinding_feed_size",
                "grinding_product_size",
            ]
        ].dropna()

        lines.append(
            f"Grinding records evaluated: {len(valid)}"
        )

    else:

        lines.append(
            "Grinding records evaluated: N/A"
        )

    lines.append("")

    # ========================================================================
    # 4. SIEVE ANALYSIS
    # ========================================================================

    lines.append(
        "4. SIEVE ANALYSIS"
    )

    lines.append(
        sub_separator
    )

    lines.append(
        f"Sieve records: {summary['sieve_row_count']}"
    )

    lines.append(
        f"Number of unique sieve sizes: "
        f"{summary['sieve_size_count']}"
    )

    lines.append(
        f"Smallest detected sieve size: "
        f"{_format_number(summary['minimum_sieve_size'])}"
    )

    lines.append(
        f"Largest detected sieve size: "
        f"{_format_number(summary['maximum_sieve_size'])}"
    )

    sieve_sizes = summary.get(
        "sieve_sizes",
        [],
    )

    if sieve_sizes:

        sieve_text = ", ".join(
            _format_number(
                value,
                2,
            )
            for value in sieve_sizes
        )

        lines.append(
            f"Detected sieve sizes: {sieve_text}"
        )

    else:

        lines.append(
            "Detected sieve sizes: N/A"
        )

    if not sieve.empty:

        lines.append("")

        lines.append(
            "Sieve distribution by sample:"
        )

        sample_groups = sieve.groupby(
            [
                "sample_id",
                "replicate",
            ],
            dropna=False,
        )

        for (
            sample_id,
            replicate,
        ), group in sample_groups:

            lines.append(
                f"  Sample {sample_id}, "
                f"Replicate {replicate}: "
                f"{len(group)} sieve record(s)"
            )

    lines.append("")

    # ========================================================================
    # 5. BENEFICIATION
    # ========================================================================

    lines.append(
        "5. BENEFICIATION ANALYSIS"
    )

    lines.append(
        sub_separator
    )

    lines.append(
        f"Total concentrate mass: "
        f"{_format_number(summary['total_concentrate_mass'])}"
    )

    lines.append(
        f"Total tailings mass: "
        f"{_format_number(summary['total_tailings_mass'])}"
    )

    lines.append(
        f"Average mass recovery: "
        f"{_format_percent(summary['average_mass_recovery'])}"
    )

    lines.append(
        f"Average yield: "
        f"{_format_percent(summary['average_yield'])}"
    )

    lines.append(
        f"Average grade recovery: "
        f"{_format_percent(summary['average_grade_recovery'])}"
    )

    lines.append(
        f"Average enrichment ratio: "
        f"{_format_number(summary['average_enrichment_ratio'])}"
    )

    lines.append(
        f"Average concentrate grade: "
        f"{_format_number(summary['average_concentrate_grade'])}"
    )

    if "beneficiation_method" in process.columns:

        methods = (
            process[
                "beneficiation_method"
            ]
            .dropna()
            .astype(str)
            .str.strip()
        )

        methods = sorted(
            methods.unique().tolist()
        )

        lines.append(
            "Beneficiation method(s): "
            + (
                ", ".join(methods)
                if methods
                else "N/A"
            )
        )

    lines.append("")

    # ========================================================================
    # 6. MATERIAL BALANCE
    # ========================================================================

    lines.append(
        "6. MATERIAL BALANCE"
    )

    lines.append(
        sub_separator
    )

    lines.append(
        f"Average mass closure: "
        f"{_format_percent(summary['average_closure'])}"
    )

    lines.append(
        f"Average balance error: "
        f"{_format_percent(summary['average_balance_error'])}"
    )

    lines.append(
        f"Maximum balance error: "
        f"{_format_percent(summary['maximum_balance_error'])}"
    )

    if {
        "sample_id",
        "replicate",
        "balance_error_percent",
    }.issubset(
        process.columns
    ):

        lines.append("")

        lines.append(
            "Sample material-balance results:"
        )

        balance_columns = [
            "sample_id",
            "replicate",
            "closure_percent",
            "balance_error_percent",
        ]

        for _, row in process[
            balance_columns
        ].iterrows():

            lines.append(
                "  "
                f"{row['sample_id']} / Rep {row['replicate']}: "
                f"closure="
                f"{_format_percent(row['closure_percent'])}, "
                f"error="
                f"{_format_percent(row['balance_error_percent'])}"
            )

    lines.append("")

    # ========================================================================
    # 7. SAMPLE RESULTS
    # ========================================================================

    lines.append(
        "7. SAMPLE-LEVEL RESULTS"
    )

    lines.append(
        sub_separator
    )

    sample_summary = generate_sample_summary(
        data
    )

    if sample_summary.empty:

        lines.append(
            "No process-level sample results are available."
        )

    else:

        for _, row in sample_summary.iterrows():

            sample_id = row.get(
                "sample_id",
                "N/A",
            )

            replicate = row.get(
                "replicate",
                "N/A",
            )

            lines.append(
                f"Sample: {sample_id} | "
                f"Replicate: {replicate}"
            )

            if "feed_mass" in row.index:

                lines.append(
                    "  Feed mass: "
                    f"{_format_number(row['feed_mass'])}"
                )

            if "feed_grade" in row.index:

                lines.append(
                    "  Feed grade: "
                    f"{_format_number(row['feed_grade'])}"
                )

            if "concentrate_mass" in row.index:

                lines.append(
                    "  Concentrate mass: "
                    f"{_format_number(row['concentrate_mass'])}"
                )

            if "concentrate_grade" in row.index:

                lines.append(
                    "  Concentrate grade: "
                    f"{_format_number(row['concentrate_grade'])}"
                )

            if "mass_recovery" in row.index:

                lines.append(
                    "  Mass recovery: "
                    f"{_format_percent(row['mass_recovery'])}"
                )

            if "grade_recovery" in row.index:

                lines.append(
                    "  Grade recovery: "
                    f"{_format_percent(row['grade_recovery'])}"
                )

            if "enrichment_ratio" in row.index:

                lines.append(
                    "  Enrichment ratio: "
                    f"{_format_number(row['enrichment_ratio'])}"
                )

            lines.append("")

    # ========================================================================
    # 8. DATA QUALITY
    # ========================================================================

    lines.append(
        "8. DATA QUALITY"
    )

    lines.append(
        sub_separator
    )

    missing_cells = int(
        data.isna().sum().sum()
    )

    duplicate_rows = int(
        data.duplicated().sum()
    )

    lines.append(
        f"Missing cells: {missing_cells}"
    )

    lines.append(
        f"Exact duplicate rows: {duplicate_rows}"
    )

    if "sample_id" in data.columns:

        blank_samples = int(
            data[
                "sample_id"
            ]
            .isna()
            .sum()
        )

        lines.append(
            f"Rows with missing sample ID: {blank_samples}"
        )

    if "sieve_size" in data.columns:

        sieve_numeric = _safe_numeric(
            data[
                "sieve_size"
            ]
        )

        invalid_sieves = int(
            (
                sieve_numeric <= 0
            ).sum()
        )

        lines.append(
            f"Invalid sieve-size entries: {invalid_sieves}"
        )

    lines.append("")

    # ========================================================================
    # 9. OBSERVATIONS
    # ========================================================================

    lines.append(
        "9. OVERALL OBSERVATIONS"
    )

    lines.append(
        sub_separator
    )

    observations = _generate_observations(
        data,
        process,
        sieve,
        summary,
    )

    if observations:

        for observation in observations:

            lines.append(
                f"- {observation}"
            )

    else:

        lines.append(
            "- No automatic observations were generated."
        )

    lines.append("")

    lines.append(
        separator
    )

    lines.append(
        "END OF REPORT".center(78)
    )

    lines.append(
        separator
    )

    return "\n".join(
        lines
    )


# ============================================================================
# AUTOMATIC OBSERVATIONS
# ============================================================================

def _generate_observations(
    df: pd.DataFrame,
    process: pd.DataFrame,
    sieve: pd.DataFrame,
    summary: Dict[str, Any],
) -> List[str]:
    """
    Generate cautious automatic observations.

    These are descriptive observations only. They do not replace
    scientific interpretation of laboratory results.
    """

    observations: List[str] = []

    # ------------------------------------------------------------------------
    # DATASET SIZE
    # ------------------------------------------------------------------------

    if summary["sample_count"] > 0:

        observations.append(
            f"The dataset contains {summary['sample_count']} unique sample(s)."
        )

    # ------------------------------------------------------------------------
    # SIEVE RANGE
    # ------------------------------------------------------------------------

    if (
        summary["minimum_sieve_size"]
        is not None
    ):

        observations.append(
            "The sieve-analysis range was detected dynamically from "
            f"{_format_number(summary['minimum_sieve_size'])} to "
            f"{_format_number(summary['maximum_sieve_size'])}."
        )

    # ------------------------------------------------------------------------
    # CRUSHING
    # ------------------------------------------------------------------------

    if (
        summary["average_crush_size_reduction"]
        is not None
    ):

        observations.append(
            "The calculated average crushing size reduction was "
            f"{_format_percent(summary['average_crush_size_reduction'])}."
        )

    # ------------------------------------------------------------------------
    # GRINDING
    # ------------------------------------------------------------------------

    if (
        summary["average_grinding_size_reduction"]
        is not None
    ):

        observations.append(
            "The calculated average grinding size reduction was "
            f"{_format_percent(summary['average_grinding_size_reduction'])}."
        )

    # ------------------------------------------------------------------------
    # CONCENTRATE GRADE
    # ------------------------------------------------------------------------

    if (
        summary["average_feed_grade"]
        is not None
        and summary["average_concentrate_grade"]
        is not None
    ):

        feed_grade = summary[
            "average_feed_grade"
        ]

        concentrate_grade = summary[
            "average_concentrate_grade"
        ]

        if concentrate_grade > feed_grade:

            observations.append(
                "The average concentrate grade is higher than the "
                "average feed grade."
            )

        elif concentrate_grade < feed_grade:

            observations.append(
                "The average concentrate grade is lower than the "
                "average feed grade."
            )

        else:

            observations.append(
                "The average concentrate grade is approximately equal "
                "to the average feed grade."
            )

    # ------------------------------------------------------------------------
    # MATERIAL BALANCE
    # ------------------------------------------------------------------------

    if (
        summary["average_balance_error"]
        is not None
    ):

        observations.append(
            "The average calculated material-balance error was "
            f"{_format_percent(summary['average_balance_error'])}."
        )

    # ------------------------------------------------------------------------
    # MISSING DATA
    # ------------------------------------------------------------------------

    missing_cells = int(
        df.isna().sum().sum()
    )

    if missing_cells > 0:

        observations.append(
            f"The dataset contains {missing_cells} missing cell(s); "
            "these should be reviewed before final scientific interpretation."
        )

    return observations


# ============================================================================
# SAVE TEXT REPORT
# ============================================================================

def ensure_reports_directory(
    directory: Optional[str | Path] = None,
) -> Path:
    """
    Create and return the report directory.
    """

    if directory is None:

        path = REPORTS_DIR

    else:

        path = Path(
            directory
        ).expanduser().resolve()

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def _sanitize_filename(
    filename: str,
) -> str:
    """
    Make a filename safe for Windows.
    """

    unsafe = '<>:"/\\|?*'

    cleaned = "".join(
        "_"
        if character in unsafe
        else character
        for character in str(filename)
    )

    return (
        cleaned.strip()
        or "LEBAS_report"
    )


def _unique_path(
    path: Path,
) -> Path:
    """
    Generate a unique path instead of overwriting an existing report.
    """

    if not path.exists():
        return path

    counter = 1

    while True:

        candidate = path.with_name(
            f"{path.stem}_{counter}{path.suffix}"
        )

        if not candidate.exists():

            return candidate

        counter += 1


def save_text_report(
    df: pd.DataFrame,
    filename: str | Path = "LEBAS_report.txt",
    directory: Optional[str | Path] = None,
    unique_filename: bool = True,
) -> Path:
    """
    Generate and save a text report.
    """

    report = generate_text_report(
        df
    )

    output_dir = ensure_reports_directory(
        directory
    )

    filename = _sanitize_filename(
        Path(
            str(filename)
        ).name
    )

    if not filename.lower().endswith(
        ".txt"
    ):

        filename += ".txt"

    output_path = (
        output_dir
        / filename
    )

    if unique_filename:

        output_path = _unique_path(
            output_path
        )

    output_path.write_text(
        report,
        encoding="utf-8",
    )

    logger.info(
        "LEBAS text report saved: %s",
        output_path,
    )

    return output_path


# ============================================================================
# EXCEL REPORT
# ============================================================================

def save_excel_report(
    df: pd.DataFrame,
    filename: str | Path = "LEBAS_report.xlsx",
    directory: Optional[str | Path] = None,
    unique_filename: bool = True,
) -> Path:
    """
    Generate a structured Excel laboratory report.

    Sheets:
        Summary
        Sample Results
        Sieve Analysis
        Raw Data
    """

    data = _clean_dataframe(
        df
    )

    summary = generate_summary(
        data
    )

    sample_summary = generate_sample_summary(
        data
    )

    sieve = _prepare_sieve_data(
        data
    )

    output_dir = ensure_reports_directory(
        directory
    )

    filename = _sanitize_filename(
        Path(
            str(filename)
        ).name
    )

    if not filename.lower().endswith(
        ".xlsx"
    ):

        filename += ".xlsx"

    output_path = (
        output_dir
        / filename
    )

    if unique_filename:

        output_path = _unique_path(
            output_path
        )

    try:

        with pd.ExcelWriter(
            output_path,
            engine="openpyxl",
        ) as writer:

            # --------------------------------------------------------------
            # SUMMARY
            # --------------------------------------------------------------

            summary_rows = []

            for key, value in summary.items():

                if isinstance(
                    value,
                    list,
                ):

                    value = ", ".join(
                        str(item)
                        for item in value
                    )

                elif isinstance(
                    value,
                    pd.Timestamp,
                ):

                    value = value.strftime(
                        "%Y-%m-%d"
                    )

                summary_rows.append(
                    {
                        "Metric": str(key)
                        .replace(
                            "_",
                            " ",
                        )
                        .title(),
                        "Value": value,
                    }
                )

            summary_df = pd.DataFrame(
                summary_rows
            )

            summary_df.to_excel(
                writer,
                sheet_name="Summary",
                index=False,
            )

            # --------------------------------------------------------------
            # SAMPLE RESULTS
            # --------------------------------------------------------------

            if sample_summary.empty:

                pd.DataFrame(
                    {
                        "Information": [
                            "No process-level sample results available."
                        ]
                    }
                ).to_excel(
                    writer,
                    sheet_name="Sample Results",
                    index=False,
                )

            else:

                sample_summary.to_excel(
                    writer,
                    sheet_name="Sample Results",
                    index=False,
                )

            # --------------------------------------------------------------
            # SIEVE ANALYSIS
            # --------------------------------------------------------------

            if sieve.empty:

                pd.DataFrame(
                    {
                        "Information": [
                            "No valid sieve-analysis records available."
                        ]
                    }
                ).to_excel(
                    writer,
                    sheet_name="Sieve Analysis",
                    index=False,
                )

            else:

                sieve.to_excel(
                    writer,
                    sheet_name="Sieve Analysis",
                    index=False,
                )

            # --------------------------------------------------------------
            # RAW DATA
            # --------------------------------------------------------------

            data.to_excel(
                writer,
                sheet_name="Raw Data",
                index=False,
            )

            # --------------------------------------------------------------
            # FORMAT WORKBOOK
            # --------------------------------------------------------------

            for sheet_name, worksheet in writer.sheets.items():

                worksheet.freeze_panes = "A2"

                for cell in worksheet[1]:

                    try:

                        cell.font = cell.font.copy(
                            bold=True
                        )

                    except Exception:

                        pass

                # Determine reasonable column widths.
                for column_cells in worksheet.columns:

                    try:

                        column_letter = (
                            column_cells[0]
                            .column_letter
                        )

                        max_length = 0

                        for cell in column_cells[:100]:

                            value = cell.value

                            if value is None:
                                continue

                            max_length = max(
                                max_length,
                                len(
                                    str(value)
                                ),
                            )

                        worksheet.column_dimensions[
                            column_letter
                        ].width = min(
                            max(
                                max_length + 2,
                                10,
                            ),
                            40,
                        )

                    except Exception:

                        continue

    except ImportError as exc:

        raise ImportError(
            "Excel report generation requires openpyxl. "
            "Install it with: pip install openpyxl"
        ) from exc

    logger.info(
        "LEBAS Excel report saved: %s",
        output_path,
    )

    return output_path


# ============================================================================
# GENERIC REPORT GENERATOR
# ============================================================================

def generate_report(
    df: pd.DataFrame,
    output_format: str = "txt",
    filename: Optional[str | Path] = None,
    directory: Optional[str | Path] = None,
) -> Path:
    """
    Generate and save a report.

    Supported formats:
        txt
        text
        xlsx
        excel
    """

    format_name = str(
        output_format
    ).strip().lower()

    if format_name in {
        "txt",
        "text",
    }:

        if filename is None:

            filename = "LEBAS_report.txt"

        return save_text_report(
            df,
            filename=filename,
            directory=directory,
        )

    if format_name in {
        "xlsx",
        "excel",
    }:

        if filename is None:

            filename = "LEBAS_report.xlsx"

        return save_excel_report(
            df,
            filename=filename,
            directory=directory,
        )

    raise ValueError(
        "Unsupported report format. "
        "Use 'txt' or 'xlsx'."
    )


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

create_report = generate_report

create_text_report = generate_text_report

save_report = save_text_report

generate_lab_report = generate_text_report


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "PROJECT_ROOT",
    "REPORTS_DIR",
    "generate_summary",
    "generate_sample_summary",
    "generate_text_report",
    "save_text_report",
    "save_excel_report",
    "generate_report",
    "create_report",
    "create_text_report",
    "save_report",
    "generate_lab_report",
]