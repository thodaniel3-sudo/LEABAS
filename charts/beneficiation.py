"""
LEBAS - Beneficiation Chart Module

File:
    charts/beneficiation.py

Purpose:
    Provides reusable chart functions for beneficiation analysis.

The module is fully dataset-driven and works with the LEBAS dataset.csv
schema. It does not contain hard-coded experimental results.

Important:
    - Repeated sieve rows are NOT double-counted.
    - Process-level calculations are performed once per sample/replicate.
    - Missing optional columns are handled safely.
    - Charts automatically use whatever samples and measurements exist.
    - No particular beneficiation method is hard-coded.

Main calculations:
    Mass Recovery
    Yield
    Grade Recovery
    Enrichment Ratio
    Mass Balance
    Balance Error

Main chart functions:
    prepare_beneficiation_data()
    calculate_beneficiation_metrics()
    create_beneficiation_figure()
    create_grade_figure()
    create_yield_figure()
    create_enrichment_figure()
    create_mass_balance_figure()
    create_beneficiation_comparison_figure()
    embed_beneficiation_chart()
    get_beneficiation_summary()
    save_beneficiation_chart()
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:
    FigureCanvasTkAgg = None


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# REQUIRED COLUMNS
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "sample_id",
    "replicate",
    "feed_mass",
    "concentrate_mass",
}


# ---------------------------------------------------------------------------
# OPTIONAL COLUMNS
# ---------------------------------------------------------------------------

OPTIONAL_COLUMNS = {
    "material_type",
    "experiment_date",
    "operator",
    "feed_grade",
    "concentrate_grade",
    "tailings_mass",
    "tailings_grade",
    "losses_mass",
    "beneficiation_method",
    "beneficiation_time",
    "reagent_type",
    "reagent_dosage",
    "ph_level",
    "pulp_density",
    "temperature",
    "solid_recovery",
}


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

def _validate_dataframe(df: pd.DataFrame) -> None:
    """Validate the minimum required dataset structure."""

    if df is None:
        raise ValueError("Dataset cannot be None.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Dataset must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    missing = REQUIRED_COLUMNS.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "Dataset is missing required beneficiation columns: "
            + ", ".join(sorted(missing))
        )


# ---------------------------------------------------------------------------
# NUMERIC HELPERS
# ---------------------------------------------------------------------------

def _safe_numeric(series: pd.Series) -> pd.Series:
    """Convert values to numeric, replacing invalid values with NaN."""
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def _normalise_replicate(value) -> str:
    """Normalize replicate values for reliable grouping/filtering."""

    if pd.isna(value):
        return ""

    try:
        number = float(value)

        if number.is_integer():
            return str(int(number))

    except (ValueError, TypeError):
        pass

    return str(value).strip()


def _first_valid(series: pd.Series):
    """Return the first non-null value."""

    valid = series.dropna()

    if valid.empty:
        return np.nan

    return valid.iloc[0]


# ---------------------------------------------------------------------------
# PROCESS-LEVEL DATA PREPARATION
# ---------------------------------------------------------------------------

def _prepare_process_records(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reduce repeated sieve-level records to one process record per
    sample and replicate.

    This is critical because a single beneficiation experiment can
    appear on many rows in dataset.csv due to sieve analysis.

    Process-level values such as feed mass and concentrate mass must
    therefore NOT be summed across all rows.
    """

    data = df.copy()

    data["sample_id"] = (
        data["sample_id"]
        .astype(str)
        .str.strip()
    )

    data["_replicate"] = data[
        "replicate"
    ].apply(
        _normalise_replicate
    )

    numeric_columns = [
        "feed_mass",
        "feed_grade",
        "concentrate_mass",
        "concentrate_grade",
        "tailings_mass",
        "tailings_grade",
        "losses_mass",
        "beneficiation_time",
        "reagent_dosage",
        "ph_level",
        "pulp_density",
        "temperature",
        "solid_recovery",
    ]

    for column in numeric_columns:

        if column in data.columns:
            data[column] = _safe_numeric(
                data[column]
            )

    # Calculate completeness so that when the same process information
    # is repeated across sieve rows, the most complete record is retained.
    available_numeric = [
        column
        for column in numeric_columns
        if column in data.columns
    ]

    available_text = [
        column
        for column in [
            "material_type",
            "experiment_date",
            "operator",
            "beneficiation_method",
            "reagent_type",
        ]
        if column in data.columns
    ]

    completeness_columns = (
        available_numeric
        + available_text
    )

    if completeness_columns:
        data["_completeness"] = data[
            completeness_columns
        ].notna().sum(axis=1)
    else:
        data["_completeness"] = 0

    data = data.sort_values(
        by="_completeness",
        ascending=False,
    )

    grouped = (
        data.groupby(
            [
                "sample_id",
                "_replicate",
            ],
            as_index=False,
            sort=False,
        )
        .first()
    )

    grouped = grouped.rename(
        columns={
            "_replicate": "replicate"
        }
    )

    return grouped


# ---------------------------------------------------------------------------
# FILTERING
# ---------------------------------------------------------------------------

def _filter_records(
    df: pd.DataFrame,
    sample_id: Optional[str] = None,
    replicate: Optional[str] = None,
) -> pd.DataFrame:
    """Filter process-level records."""

    result = df.copy()

    if (
        sample_id is not None
        and str(sample_id).strip()
    ):

        target = str(
            sample_id
        ).strip()

        result = result[
            result["sample_id"]
            .astype(str)
            .str.strip()
            == target
        ]

    if (
        replicate is not None
        and str(replicate).strip()
    ):

        target = _normalise_replicate(
            replicate
        )

        result = result[
            result["replicate"]
            .apply(_normalise_replicate)
            == target
        ]

    return result


# ---------------------------------------------------------------------------
# BENEFICIATION METRICS
# ---------------------------------------------------------------------------

def prepare_beneficiation_data(
    df: pd.DataFrame,
    sample_id: Optional[str] = None,
    replicate: Optional[str] = None,
) -> pd.DataFrame:
    """
    Prepare calculated beneficiation metrics.

    Calculations:

        Mass Recovery (%)
            = Concentrate Mass / Feed Mass × 100

        Yield (%)
            = Concentrate Mass / Feed Mass × 100

        Grade Recovery (%)
            = (Concentrate Mass × Concentrate Grade)
              ---------------------------------------- × 100
                    (Feed Mass × Feed Grade)

        Enrichment Ratio
            = Concentrate Grade / Feed Grade

        Mass Balance
            = Feed Mass
              - Concentrate Mass
              - Tailings Mass
              - Losses Mass

        Balance Error (%)
            = |Mass Balance| / Feed Mass × 100

    The calculations are performed once for each sample/replicate.
    """

    _validate_dataframe(df)

    data = _prepare_process_records(
        df
    )

    data = _filter_records(
        data,
        sample_id=sample_id,
        replicate=replicate,
    )

    if data.empty:
        return data.reset_index(
            drop=True
        )

    data = data.copy()

    # ---------------------------------------------------------------
    # Required masses
    # ---------------------------------------------------------------

    feed_mass = _safe_numeric(
        data["feed_mass"]
    )

    concentrate_mass = _safe_numeric(
        data["concentrate_mass"]
    )

    # ---------------------------------------------------------------
    # Mass recovery
    # ---------------------------------------------------------------

    data["mass_recovery"] = np.where(
        feed_mass > 0,
        (
            concentrate_mass
            / feed_mass
            * 100.0
        ),
        np.nan,
    )

    # ---------------------------------------------------------------
    # Yield
    # ---------------------------------------------------------------

    data["yield"] = np.where(
        feed_mass > 0,
        (
            concentrate_mass
            / feed_mass
            * 100.0
        ),
        np.nan,
    )

    # ---------------------------------------------------------------
    # Grade recovery and enrichment
    # ---------------------------------------------------------------

    if (
        "feed_grade" in data.columns
        and "concentrate_grade" in data.columns
    ):

        feed_grade = _safe_numeric(
            data["feed_grade"]
        )

        concentrate_grade = _safe_numeric(
            data["concentrate_grade"]
        )

        grade_input = (
            feed_mass
            * feed_grade
        )

        grade_output = (
            concentrate_mass
            * concentrate_grade
        )

        data["grade_recovery"] = np.where(
            grade_input > 0,
            (
                grade_output
                / grade_input
                * 100.0
            ),
            np.nan,
        )

        data["enrichment_ratio"] = np.where(
            feed_grade > 0,
            (
                concentrate_grade
                / feed_grade
            ),
            np.nan,
        )

    else:

        data["grade_recovery"] = np.nan

        data["enrichment_ratio"] = np.nan

    # ---------------------------------------------------------------
    # Tailings and losses
    # ---------------------------------------------------------------

    if "tailings_mass" in data.columns:

        tailings_mass = _safe_numeric(
            data["tailings_mass"]
        )

    else:

        tailings_mass = pd.Series(
            np.zeros(len(data)),
            index=data.index,
        )

    if "losses_mass" in data.columns:

        losses_mass = _safe_numeric(
            data["losses_mass"]
        )

    else:

        losses_mass = pd.Series(
            np.zeros(len(data)),
            index=data.index,
        )

    # ---------------------------------------------------------------
    # Material balance
    # ---------------------------------------------------------------

    data["mass_balance"] = (
        feed_mass
        - concentrate_mass
        - tailings_mass
        - losses_mass
    )

    data["balance_error_percent"] = np.where(
        feed_mass > 0,
        (
            data["mass_balance"].abs()
            / feed_mass
            * 100.0
        ),
        np.nan,
    )

    # ---------------------------------------------------------------
    # Clean infinite values
    # ---------------------------------------------------------------

    calculated_columns = [
        "mass_recovery",
        "yield",
        "grade_recovery",
        "enrichment_ratio",
        "mass_balance",
        "balance_error_percent",
    ]

    for column in calculated_columns:

        data[column] = data[
            column
        ].replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

    return data.reset_index(
        drop=True
    )


def calculate_beneficiation_metrics(
    df: pd.DataFrame,
    sample_id: Optional[str] = None,
    replicate: Optional[str] = None,
) -> pd.DataFrame:
    """Calculate beneficiation metrics."""

    return prepare_beneficiation_data(
        df,
        sample_id=sample_id,
        replicate=replicate,
    )


def calculate_beneficiation(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compatibility wrapper used by other LEBAS modules.

    Returns all calculated beneficiation records.
    """

    return calculate_beneficiation_metrics(
        df
    )


# ---------------------------------------------------------------------------
# SINGLE SAMPLE CHART
# ---------------------------------------------------------------------------

def create_beneficiation_figure(
    df: pd.DataFrame,
    sample_id: Optional[str] = None,
    replicate: Optional[str] = None,
    ax=None,
    title: Optional[str] = None,
) -> Figure:
    """
    Create a beneficiation performance chart.

    Displays:
        Mass Recovery
        Grade Recovery, if available
        Yield

    If no sample is selected, values are averaged by sample.
    """

    data = calculate_beneficiation_metrics(
        df,
        sample_id=sample_id,
        replicate=replicate,
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(9, 5.5)
        )
    else:
        fig = ax.figure

    ax.clear()

    if data.empty:

        ax.text(
            0.5,
            0.5,
            "No beneficiation data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )

        ax.set_title(
            title
            or "Beneficiation Performance"
        )

        fig.tight_layout()

        return fig

    # ---------------------------------------------------------------
    # Determine grouping
    # ---------------------------------------------------------------

    if sample_id is None:

        grouped = (
            data
            .groupby("sample_id")
            .agg(
                mass_recovery=(
                    "mass_recovery",
                    "mean",
                ),
                grade_recovery=(
                    "grade_recovery",
                    "mean",
                ),
                yield_value=(
                    "yield",
                    "mean",
                ),
            )
        )

        labels = (
            grouped.index
            .astype(str)
            .tolist()
        )

    else:

        grouped = data.copy()

        labels = (
            grouped["replicate"]
            .astype(str)
            .tolist()
        )

    # ---------------------------------------------------------------
    # Available metrics
    # ---------------------------------------------------------------

    metric_columns = []

    if grouped[
        "mass_recovery"
    ].notna().any():

        metric_columns.append(
            "mass_recovery"
        )

    if grouped[
        "grade_recovery"
    ].notna().any():

        metric_columns.append(
            "grade_recovery"
        )

    if grouped[
        "yield_value"
    ].notna().any():

        metric_columns.append(
            "yield_value"
        )

    if not metric_columns:

        ax.text(
            0.5,
            0.5,
            "No valid beneficiation metrics available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        ax.set_title(
            title
            or "Beneficiation Performance"
        )

        fig.tight_layout()

        return fig

    # ---------------------------------------------------------------
    # Plot grouped bars
    # ---------------------------------------------------------------

    x = np.arange(
        len(labels)
    )

    width = (
        0.8
        / len(metric_columns)
    )

    metric_labels = {
        "mass_recovery": "Mass Recovery",
        "grade_recovery": "Grade Recovery",
        "yield_value": "Yield",
    }

    for index, metric in enumerate(
        metric_columns
    ):

        values = grouped[
            metric
        ].to_numpy()

        positions = (
            x
            - 0.4
            + width / 2
            + index * width
        )

        ax.bar(
            positions,
            values,
            width=width,
            label=metric_labels[
                metric
            ],
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels,
        rotation=0,
    )

    if sample_id is None:

        ax.set_xlabel(
            "Sample ID"
        )

    else:

        ax.set_xlabel(
            "Replicate"
        )

    ax.set_ylabel(
        "Percentage (%)"
    )

    ax.set_title(
        title
        or "Beneficiation Performance"
    )

    ax.set_ylim(
        bottom=0
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# FEED AND CONCENTRATE GRADE
# ---------------------------------------------------------------------------

def create_grade_figure(
    df: pd.DataFrame,
    ax=None,
    title: str = "Feed and Concentrate Grade",
) -> Figure:
    """
    Compare feed grade with concentrate grade by sample.

    This chart is only populated when both grade columns are available.
    """

    data = calculate_beneficiation_metrics(
        df
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(10, 6)
        )
    else:
        fig = ax.figure

    ax.clear()

    required = [
        "feed_grade",
        "concentrate_grade",
    ]

    if any(
        column not in data.columns
        for column in required
    ):

        ax.text(
            0.5,
            0.5,
            "Feed/concentrate grade data unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        ax.set_title(title)

        fig.tight_layout()

        return fig

    valid = data.dropna(
        subset=required
    )

    if valid.empty:

        ax.text(
            0.5,
            0.5,
            "No valid grade data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        ax.set_title(title)

        fig.tight_layout()

        return fig

    grouped = (
        valid
        .groupby("sample_id")
        .agg(
            feed_grade=(
                "feed_grade",
                "mean",
            ),
            concentrate_grade=(
                "concentrate_grade",
                "mean",
            ),
        )
    )

    x = np.arange(
        len(grouped)
    )

    width = 0.35

    ax.bar(
        x - width / 2,
        grouped["feed_grade"],
        width=width,
        label="Feed Grade",
    )

    ax.bar(
        x + width / 2,
        grouped["concentrate_grade"],
        width=width,
        label="Concentrate Grade",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        grouped.index.astype(str)
    )

    ax.set_xlabel(
        "Sample ID"
    )

    ax.set_ylabel(
        "Grade"
    )

    ax.set_title(
        title
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# YIELD CHART
# ---------------------------------------------------------------------------

def create_yield_figure(
    df: pd.DataFrame,
    ax=None,
    title: str = "Beneficiation Yield",
) -> Figure:
    """Create a sample-by-sample yield chart."""

    data = calculate_beneficiation_metrics(
        df
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(9, 5.5)
        )
    else:
        fig = ax.figure

    ax.clear()

    valid = data.dropna(
        subset=["yield"]
    )

    if valid.empty:

        ax.text(
            0.5,
            0.5,
            "No yield data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        ax.set_title(title)

        fig.tight_layout()

        return fig

    grouped = (
        valid
        .groupby("sample_id")[
            "yield"
        ]
        .mean()
    )

    ax.bar(
        grouped.index.astype(str),
        grouped.values,
    )

    ax.set_xlabel(
        "Sample ID"
    )

    ax.set_ylabel(
        "Yield (%)"
    )

    ax.set_title(
        title
    )

    ax.set_ylim(
        bottom=0
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# ENRICHMENT RATIO
# ---------------------------------------------------------------------------

def create_enrichment_figure(
    df: pd.DataFrame,
    ax=None,
    title: str = "Enrichment Ratio",
) -> Figure:
    """
    Create an enrichment-ratio chart.

    Enrichment ratio:

        Concentrate Grade
        -----------------
            Feed Grade
    """

    data = calculate_beneficiation_metrics(
        df
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(9, 5.5)
        )
    else:
        fig = ax.figure

    ax.clear()

    valid = data.dropna(
        subset=["enrichment_ratio"]
    )

    if valid.empty:

        ax.text(
            0.5,
            0.5,
            "No enrichment-ratio data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        ax.set_title(title)

        fig.tight_layout()

        return fig

    grouped = (
        valid
        .groupby("sample_id")[
            "enrichment_ratio"
        ]
        .mean()
    )

    ax.bar(
        grouped.index.astype(str),
        grouped.values,
    )

    ax.set_xlabel(
        "Sample ID"
    )

    ax.set_ylabel(
        "Enrichment Ratio"
    )

    ax.set_title(
        title
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# MASS BALANCE CHART
# ---------------------------------------------------------------------------

def create_mass_balance_figure(
    df: pd.DataFrame,
    ax=None,
    title: str = "Beneficiation Mass Balance",
) -> Figure:
    """
    Compare feed mass with concentrate, tailings and losses.

    Missing tailings or losses columns are treated as zero for the
    purpose of the displayed balance.
    """

    data = calculate_beneficiation_metrics(
        df
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(10, 6)
        )
    else:
        fig = ax.figure

    ax.clear()

    if data.empty:

        ax.text(
            0.5,
            0.5,
            "No mass-balance data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        ax.set_title(title)

        fig.tight_layout()

        return fig

    columns = {
        "Feed Mass": "feed_mass",
        "Concentrate Mass": "concentrate_mass",
    }

    if "tailings_mass" in data.columns:
        columns[
            "Tailings Mass"
        ] = "tailings_mass"

    if "losses_mass" in data.columns:
        columns[
            "Losses Mass"
        ] = "losses_mass"

    x = np.arange(
        len(data)
    )

    width = (
        0.8
        / len(columns)
    )

    for index, (
        label,
        column,
    ) in enumerate(
        columns.items()
    ):

        values = _safe_numeric(
            data[column]
        )

        positions = (
            x
            - 0.4
            + width / 2
            + index * width
        )

        ax.bar(
            positions,
            values,
            width=width,
            label=label,
        )

    ax.set_xticks(
        x
    )

    labels = []

    for _, row in data.iterrows():

        sample = str(
            row["sample_id"]
        )

        replicate = str(
            row["replicate"]
        ).strip()

        if replicate:
            labels.append(
                f"{sample}\nR{replicate}"
            )
        else:
            labels.append(
                sample
            )

    ax.set_xticklabels(
        labels
    )

    ax.set_xlabel(
        "Sample / Replicate"
    )

    ax.set_ylabel(
        "Mass"
    )

    ax.set_title(
        title
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# BALANCE ERROR CHART
# ---------------------------------------------------------------------------

def create_balance_error_figure(
    df: pd.DataFrame,
    ax=None,
    title: str = "Material Balance Error",
) -> Figure:
    """Plot material balance error for each sample."""

    data = calculate_beneficiation_metrics(
        df
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(9, 5.5)
        )
    else:
        fig = ax.figure

    ax.clear()

    valid = data.dropna(
        subset=[
            "balance_error_percent"
        ]
    )

    if valid.empty:

        ax.text(
            0.5,
            0.5,
            "No balance-error data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        ax.set_title(title)

        fig.tight_layout()

        return fig

    labels = []

    for _, row in valid.iterrows():

        sample = str(
            row["sample_id"]
        )

        replicate = str(
            row["replicate"]
        ).strip()

        if replicate:
            labels.append(
                f"{sample}\nR{replicate}"
            )
        else:
            labels.append(
                sample
            )

    ax.bar(
        labels,
        valid[
            "balance_error_percent"
        ],
    )

    ax.set_xlabel(
        "Sample / Replicate"
    )

    ax.set_ylabel(
        "Balance Error (%)"
    )

    ax.set_title(
        title
    )

    ax.set_ylim(
        bottom=0
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# COMPARISON CHART
# ---------------------------------------------------------------------------

def create_beneficiation_comparison_figure(
    df: pd.DataFrame,
    ax=None,
    title: str = "Beneficiation Comparison",
) -> Figure:
    """
    Create a combined comparison chart.

    Metrics:
        Mass Recovery
        Grade Recovery
        Enrichment Ratio

    Only metrics actually available in the dataset are displayed.
    """

    data = calculate_beneficiation_metrics(
        df
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(10, 6)
        )
    else:
        fig = ax.figure

    ax.clear()

    if data.empty:

        ax.text(
            0.5,
            0.5,
            "No beneficiation data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        ax.set_title(title)

        fig.tight_layout()

        return fig

    grouped = (
        data
        .groupby("sample_id")
        .agg(
            mass_recovery=(
                "mass_recovery",
                "mean",
            ),
            grade_recovery=(
                "grade_recovery",
                "mean",
            ),
            enrichment_ratio=(
                "enrichment_ratio",
                "mean",
            ),
        )
    )

    available = []

    if grouped[
        "mass_recovery"
    ].notna().any():

        available.append(
            "mass_recovery"
        )

    if grouped[
        "grade_recovery"
    ].notna().any():

        available.append(
            "grade_recovery"
        )

    if grouped[
        "enrichment_ratio"
    ].notna().any():

        available.append(
            "enrichment_ratio"
        )

    if not available:

        ax.text(
            0.5,
            0.5,
            "No valid comparison metrics available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        ax.set_title(title)

        fig.tight_layout()

        return fig

    x = np.arange(
        len(grouped)
    )

    width = (
        0.8
        / len(available)
    )

    labels = {
        "mass_recovery": "Mass Recovery (%)",
        "grade_recovery": "Grade Recovery (%)",
        "enrichment_ratio": "Enrichment Ratio",
    }

    for index, metric in enumerate(
        available
    ):

        positions = (
            x
            - 0.4
            + width / 2
            + index * width
        )

        ax.bar(
            positions,
            grouped[metric],
            width=width,
            label=labels[metric],
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        grouped.index.astype(str)
    )

    ax.set_xlabel(
        "Sample ID"
    )

    ax.set_ylabel(
        "Calculated Value"
    )

    ax.set_title(
        title
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# TKINTER EMBEDDING
# ---------------------------------------------------------------------------

def embed_beneficiation_chart(
    parent,
    df: pd.DataFrame,
    sample_id: Optional[str] = None,
    replicate: Optional[str] = None,
    chart_type: str = "beneficiation",
    figure: Optional[Figure] = None,
):
    """
    Embed a beneficiation chart into a Tkinter widget.

    Supported chart types:

        beneficiation
        grade
        yield
        enrichment
        mass_balance
        balance_error
        comparison
    """

    if FigureCanvasTkAgg is None:
        raise RuntimeError(
            "Matplotlib Tkinter integration is unavailable."
        )

    if figure is None:

        if chart_type == "grade":

            figure = create_grade_figure(
                df
            )

        elif chart_type == "yield":

            figure = create_yield_figure(
                df
            )

        elif chart_type == "enrichment":

            figure = create_enrichment_figure(
                df
            )

        elif chart_type == "mass_balance":

            figure = create_mass_balance_figure(
                df
            )

        elif chart_type == "balance_error":

            figure = create_balance_error_figure(
                df
            )

        elif chart_type == "comparison":

            figure = create_beneficiation_comparison_figure(
                df
            )

        else:

            figure = create_beneficiation_figure(
                df,
                sample_id=sample_id,
                replicate=replicate,
            )

    canvas = FigureCanvasTkAgg(
        figure,
        master=parent,
    )

    canvas.draw()

    widget = canvas.get_tk_widget()

    widget.pack(
        fill="both",
        expand=True,
    )

    return canvas


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------

def get_beneficiation_summary(
    df: pd.DataFrame,
) -> dict:
    """
    Return a summary of beneficiation performance.

    The values are calculated from one record per sample/replicate,
    preventing repeated sieve rows from inflating the results.
    """

    data = calculate_beneficiation_metrics(
        df
    )

    if data.empty:

        return {
            "number_of_runs": 0,
            "average_mass_recovery": None,
            "maximum_mass_recovery": None,
            "minimum_mass_recovery": None,
            "average_grade_recovery": None,
            "average_yield": None,
            "average_enrichment_ratio": None,
            "average_balance_error": None,
        }

    def safe_mean(
        column: str,
    ):
        if column not in data.columns:
            return None

        values = _safe_numeric(
            data[column]
        ).dropna()

        if values.empty:
            return None

        return float(
            values.mean()
        )

    def safe_max(
        column: str,
    ):
        if column not in data.columns:
            return None

        values = _safe_numeric(
            data[column]
        ).dropna()

        if values.empty:
            return None

        return float(
            values.max()
        )

    def safe_min(
        column: str,
    ):
        if column not in data.columns:
            return None

        values = _safe_numeric(
            data[column]
        ).dropna()

        if values.empty:
            return None

        return float(
            values.min()
        )

    return {
        "number_of_runs": int(
            len(data)
        ),
        "average_mass_recovery": safe_mean(
            "mass_recovery"
        ),
        "maximum_mass_recovery": safe_max(
            "mass_recovery"
        ),
        "minimum_mass_recovery": safe_min(
            "mass_recovery"
        ),
        "average_grade_recovery": safe_mean(
            "grade_recovery"
        ),
        "average_yield": safe_mean(
            "yield"
        ),
        "average_enrichment_ratio": safe_mean(
            "enrichment_ratio"
        ),
        "average_balance_error": safe_mean(
            "balance_error_percent"
        ),
    }


# ---------------------------------------------------------------------------
# SAVE CHART
# ---------------------------------------------------------------------------

def save_beneficiation_chart(
    df: pd.DataFrame,
    filepath: str,
    chart_type: str = "beneficiation",
    dpi: int = 300,
) -> str:
    """
    Create and save a beneficiation chart.

    Parameters
    ----------
    df : pandas.DataFrame
        LEBAS dataset.

    filepath : str
        Output image path.

    chart_type : str
        beneficiation
        grade
        yield
        enrichment
        mass_balance
        balance_error
        comparison

    dpi : int
        Output resolution.

    Returns
    -------
    str
        Saved filepath.
    """

    if chart_type == "grade":

        figure = create_grade_figure(
            df
        )

    elif chart_type == "yield":

        figure = create_yield_figure(
            df
        )

    elif chart_type == "enrichment":

        figure = create_enrichment_figure(
            df
        )

    elif chart_type == "mass_balance":

        figure = create_mass_balance_figure(
            df
        )

    elif chart_type == "balance_error":

        figure = create_balance_error_figure(
            df
        )

    elif chart_type == "comparison":

        figure = create_beneficiation_comparison_figure(
            df
        )

    else:

        figure = create_beneficiation_figure(
            df
        )

    figure.savefig(
        filepath,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    logger.info(
        "Beneficiation chart saved to %s",
        filepath,
    )

    return filepath


# ---------------------------------------------------------------------------
# EXPORTS
# ---------------------------------------------------------------------------

__all__ = [
    "prepare_beneficiation_data",
    "calculate_beneficiation_metrics",
    "calculate_beneficiation",
    "create_beneficiation_figure",
    "create_grade_figure",
    "create_yield_figure",
    "create_enrichment_figure",
    "create_mass_balance_figure",
    "create_balance_error_figure",
    "create_beneficiation_comparison_figure",
    "embed_beneficiation_chart",
    "get_beneficiation_summary",
    "save_beneficiation_chart",
]