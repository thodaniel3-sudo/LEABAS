"""
LEBAS - Recovery Chart Module

File:
    charts/recovery.py

Purpose:
    Provides reusable functions for calculating and plotting beneficiation
    recovery results from the LEBAS dataset.

Important:
    - Fully dataset-driven.
    - Does not hard-code experimental results.
    - Does not double-count repeated sieve-analysis rows.
    - Groups process data by sample_id + replicate.
    - Supports mass recovery, grade recovery, yield, and enrichment ratio.
    - Can compare multiple samples/replicates.

Main functions:
    prepare_recovery_data()
    calculate_recovery()
    calculate_all_recoveries()
    create_recovery_figure()
    create_grade_recovery_figure()
    create_recovery_comparison_figure()
    embed_recovery_chart()
    get_recovery_summary()
    save_recovery_chart()
"""

from __future__ import annotations

import logging
from typing import Optional, List

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
# HELPERS
# ---------------------------------------------------------------------------

def _validate_dataframe(df: pd.DataFrame) -> None:
    """Validate the minimum columns required for recovery calculations."""

    if df is None:
        raise ValueError("Dataset cannot be None.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Dataset must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    missing = REQUIRED_COLUMNS.difference(df.columns)

    if missing:
        raise ValueError(
            "Dataset is missing required recovery columns: "
            + ", ".join(sorted(missing))
        )


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a pandas Series to numeric values safely."""
    return pd.to_numeric(series, errors="coerce")


def _normalise_replicate(value) -> str:
    """Return a consistent string representation of replicate values."""

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
    """Return the first non-null value in a Series."""

    valid = series.dropna()

    if valid.empty:
        return np.nan

    return valid.iloc[0]


def _aggregate_process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert repeated sieve-level records into one process-level record
    per sample and replicate.

    This prevents feed mass, concentrate mass, grades, etc. from being
    counted repeatedly when a sample contains multiple sieve rows.
    """

    data = df.copy()

    data["sample_id"] = (
        data["sample_id"]
        .astype(str)
        .str.strip()
    )

    data["_replicate"] = data["replicate"].apply(
        _normalise_replicate
    )

    # Numeric conversion.
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
        if column in data.columns:
            data[column] = _safe_numeric(
                data[column]
            )

    # Select the most complete row in each sample/replicate group.
    process_columns = [
        column
        for column in numeric_columns
        if column in data.columns
    ]

    grouping_columns = [
        "sample_id",
        "_replicate",
    ]

    data["_completeness"] = data[process_columns].notna().sum(
        axis=1
    )

    data = data.sort_values(
        by="_completeness",
        ascending=False,
    )

    process_data = (
        data.groupby(
            grouping_columns,
            as_index=False,
            sort=False,
        )
        .first()
    )

    process_data = process_data.rename(
        columns={
            "_replicate": "replicate"
        }
    )

    return process_data


def _filter_data(
    df: pd.DataFrame,
    sample_id: Optional[str] = None,
    replicate: Optional[str] = None,
) -> pd.DataFrame:
    """Filter process data by sample and replicate."""

    result = df.copy()

    if sample_id is not None and str(sample_id).strip():

        target = str(sample_id).strip()

        result = result[
            result["sample_id"]
            .astype(str)
            .str.strip()
            == target
        ]

    if replicate is not None and str(replicate).strip():

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
# RECOVERY CALCULATION
# ---------------------------------------------------------------------------

def prepare_recovery_data(
    df: pd.DataFrame,
    sample_id: Optional[str] = None,
    replicate: Optional[str] = None,
) -> pd.DataFrame:
    """
    Prepare beneficiation recovery data.

    Calculated parameters:

        Mass Recovery (%)
            = Concentrate Mass / Feed Mass × 100

        Yield (%)
            = Concentrate Mass / Feed Mass × 100

        Grade Recovery (%)
            = Concentrate Mass × Concentrate Grade
              ------------------------------------ × 100
                    Feed Mass × Feed Grade

        Enrichment Ratio
            = Concentrate Grade / Feed Grade

        Mass Balance
            = Feed Mass - (Concentrate + Tailings + Losses)

        Balance Error (%)
            = |Mass Balance| / Feed Mass × 100

    Returns one record per sample/replicate.

    Parameters
    ----------
    df : pandas.DataFrame
        Complete LEBAS dataset.

    sample_id : str, optional
        Sample identifier.

    replicate : str, optional
        Replicate identifier.

    Returns
    -------
    pandas.DataFrame
        Recovery results.
    """

    _validate_dataframe(df)

    process_data = _aggregate_process_data(df)

    process_data = _filter_data(
        process_data,
        sample_id=sample_id,
        replicate=replicate,
    )

    if process_data.empty:
        return process_data

    data = process_data.copy()

    feed_mass = _safe_numeric(
        data["feed_mass"]
    )

    concentrate_mass = _safe_numeric(
        data["concentrate_mass"]
    )

    data["mass_recovery"] = np.where(
        feed_mass > 0,
        concentrate_mass / feed_mass * 100,
        np.nan,
    )

    # Yield is kept as a separate field because it is useful for reporting,
    # even though under this mass-based formulation it has the same formula
    # as mass recovery.
    data["yield"] = np.where(
        feed_mass > 0,
        concentrate_mass / feed_mass * 100,
        np.nan,
    )

    # Grade recovery requires both grades.
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

        denominator = (
            feed_mass * feed_grade
        )

        numerator = (
            concentrate_mass * concentrate_grade
        )

        data["grade_recovery"] = np.where(
            denominator > 0,
            numerator / denominator * 100,
            np.nan,
        )

        data["enrichment_ratio"] = np.where(
            feed_grade > 0,
            concentrate_grade / feed_grade,
            np.nan,
        )

    else:

        data["grade_recovery"] = np.nan
        data["enrichment_ratio"] = np.nan

    # Material balance.
    tailings_mass = (
        _safe_numeric(data["tailings_mass"])
        if "tailings_mass" in data.columns
        else pd.Series(
            np.zeros(len(data)),
            index=data.index,
        )
    )

    losses_mass = (
        _safe_numeric(data["losses_mass"])
        if "losses_mass" in data.columns
        else pd.Series(
            np.zeros(len(data)),
            index=data.index,
        )
    )

    data["mass_balance"] = (
        feed_mass
        - concentrate_mass
        - tailings_mass
        - losses_mass
    )

    data["balance_error_percent"] = np.where(
        feed_mass > 0,
        data["mass_balance"].abs()
        / feed_mass
        * 100,
        np.nan,
    )

    # Keep values within sensible reporting limits.
    data["mass_recovery"] = data[
        "mass_recovery"
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data["grade_recovery"] = data[
        "grade_recovery"
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data["enrichment_ratio"] = data[
        "enrichment_ratio"
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return data.reset_index(drop=True)


def calculate_recovery(
    df: pd.DataFrame,
    sample_id: Optional[str] = None,
    replicate: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function for calculating recovery.

    Equivalent to prepare_recovery_data().
    """

    return prepare_recovery_data(
        df,
        sample_id=sample_id,
        replicate=replicate,
    )


def calculate_all_recoveries(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate recovery results for every sample and replicate.

    Returns
    -------
    pandas.DataFrame
        One row per sample/replicate.
    """

    return prepare_recovery_data(
        df
    )


# ---------------------------------------------------------------------------
# RECOVERY CHART
# ---------------------------------------------------------------------------

def create_recovery_figure(
    df: pd.DataFrame,
    sample_id: Optional[str] = None,
    replicate: Optional[str] = None,
    ax=None,
    title: Optional[str] = None,
) -> Figure:
    """
    Create a recovery chart.

    If a sample is selected, the chart displays its mass recovery.

    If no sample is selected, the chart displays the average mass recovery
    for each sample.

    Parameters
    ----------
    df : pandas.DataFrame
        LEBAS dataset.

    sample_id : str, optional
        Sample identifier.

    replicate : str, optional
        Replicate identifier.

    ax : matplotlib.axes.Axes, optional
        Existing axes.

    title : str, optional
        Custom title.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure.
    """

    recovery_data = calculate_all_recoveries(
        df
    )

    recovery_data = _filter_data(
        recovery_data,
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

    if recovery_data.empty:

        ax.text(
            0.5,
            0.5,
            "No recovery data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )

        ax.set_title(
            title or "Beneficiation Recovery"
        )

        ax.set_ylabel(
            "Mass Recovery (%)"
        )

        fig.tight_layout()

        return fig

    if sample_id is None:

        grouped = (
            recovery_data
            .groupby("sample_id")["mass_recovery"]
            .mean()
            .dropna()
        )

        labels = grouped.index.astype(str)
        values = grouped.values

        ax.bar(
            labels,
            values,
        )

        ax.set_xlabel(
            "Sample ID"
        )

    else:

        labels = recovery_data[
            "replicate"
        ].astype(str)

        values = recovery_data[
            "mass_recovery"
        ]

        ax.bar(
            labels,
            values,
        )

        ax.set_xlabel(
            "Replicate"
        )

    ax.set_ylabel(
        "Mass Recovery (%)"
    )

    ax.set_ylim(
        bottom=0
    )

    ax.set_title(
        title or "Beneficiation Mass Recovery"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# GRADE RECOVERY CHART
# ---------------------------------------------------------------------------

def create_grade_recovery_figure(
    df: pd.DataFrame,
    sample_id: Optional[str] = None,
    replicate: Optional[str] = None,
    ax=None,
    title: Optional[str] = None,
) -> Figure:
    """
    Create a grade-recovery chart.

    Displays calculated grade recovery for each sample or replicate.
    """

    recovery_data = calculate_all_recoveries(
        df
    )

    recovery_data = _filter_data(
        recovery_data,
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

    valid = recovery_data.dropna(
        subset=["grade_recovery"]
    )

    if valid.empty:

        ax.text(
            0.5,
            0.5,
            "No grade recovery data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )

        ax.set_title(
            title or "Grade Recovery"
        )

        ax.set_ylabel(
            "Grade Recovery (%)"
        )

        fig.tight_layout()

        return fig

    if sample_id is None:

        grouped = (
            valid
            .groupby("sample_id")[
                "grade_recovery"
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

    else:

        ax.bar(
            valid["replicate"].astype(str),
            valid["grade_recovery"],
        )

        ax.set_xlabel(
            "Replicate"
        )

    ax.set_ylabel(
        "Grade Recovery (%)"
    )

    ax.set_ylim(
        bottom=0
    )

    ax.set_title(
        title or "Beneficiation Grade Recovery"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# RECOVERY COMPARISON
# ---------------------------------------------------------------------------

def create_recovery_comparison_figure(
    df: pd.DataFrame,
    ax=None,
    title: str = "Recovery Comparison",
) -> Figure:
    """
    Compare mass recovery, grade recovery, and yield.

    Only metrics available in the dataset are displayed.
    """

    recovery_data = calculate_all_recoveries(
        df
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(10, 6)
        )
    else:
        fig = ax.figure

    ax.clear()

    if recovery_data.empty:

        ax.text(
            0.5,
            0.5,
            "No recovery data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )

        ax.set_title(title)

        fig.tight_layout()

        return fig

    grouped = (
        recovery_data
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

    metrics = [
        "mass_recovery",
        "grade_recovery",
        "yield_value",
    ]

    available_metrics = [
        metric
        for metric in metrics
        if grouped[metric].notna().any()
    ]

    if not available_metrics:

        ax.text(
            0.5,
            0.5,
            "No valid recovery metrics available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )

        ax.set_title(title)

        fig.tight_layout()

        return fig

    x = np.arange(
        len(grouped.index)
    )

    width = 0.8 / len(
        available_metrics
    )

    labels = {
        "mass_recovery": "Mass Recovery",
        "grade_recovery": "Grade Recovery",
        "yield_value": "Yield",
    }

    for index, metric in enumerate(
        available_metrics
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
            label=labels[metric],
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        grouped.index.astype(str)
    )

    ax.set_xlabel(
        "Sample ID"
    )

    ax.set_ylabel(
        "Recovery / Yield (%)"
    )

    ax.set_title(title)

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
# ENRICHMENT RATIO CHART
# ---------------------------------------------------------------------------

def create_enrichment_ratio_figure(
    df: pd.DataFrame,
    ax=None,
    title: str = "Enrichment Ratio",
) -> Figure:
    """
    Plot average enrichment ratio by sample.

    Enrichment ratio:

        Concentrate Grade
        -----------------
            Feed Grade
    """

    recovery_data = calculate_all_recoveries(
        df
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(9, 5.5)
        )
    else:
        fig = ax.figure

    ax.clear()

    valid = recovery_data.dropna(
        subset=["enrichment_ratio"]
    )

    if valid.empty:

        ax.text(
            0.5,
            0.5,
            "No enrichment data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
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

    ax.set_title(title)

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# TKINTER EMBEDDING
# ---------------------------------------------------------------------------

def embed_recovery_chart(
    parent,
    df: pd.DataFrame,
    sample_id: Optional[str] = None,
    replicate: Optional[str] = None,
    figure: Optional[Figure] = None,
    chart_type: str = "recovery",
):
    """
    Embed a recovery chart inside a Tkinter widget.

    Parameters
    ----------
    parent
        Tkinter parent widget.

    df : pandas.DataFrame
        LEBAS dataset.

    sample_id : str, optional
        Sample identifier.

    replicate : str, optional
        Replicate identifier.

    figure : matplotlib.figure.Figure, optional
        Existing figure.

    chart_type : str
        Available options:

            recovery
            grade_recovery
            comparison
            enrichment

    Returns
    -------
    FigureCanvasTkAgg
        Embedded canvas.
    """

    if FigureCanvasTkAgg is None:
        raise RuntimeError(
            "Matplotlib Tkinter integration is unavailable."
        )

    if figure is None:

        if chart_type == "grade_recovery":

            figure = create_grade_recovery_figure(
                df,
                sample_id=sample_id,
                replicate=replicate,
            )

        elif chart_type == "comparison":

            figure = create_recovery_comparison_figure(
                df
            )

        elif chart_type == "enrichment":

            figure = create_enrichment_ratio_figure(
                df
            )

        else:

            figure = create_recovery_figure(
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

def get_recovery_summary(
    df: pd.DataFrame,
) -> dict:
    """
    Return a summary of calculated recovery results.
    """

    recovery_data = calculate_all_recoveries(
        df
    )

    if recovery_data.empty:
        return {
            "number_of_runs": 0,
            "average_mass_recovery": None,
            "maximum_mass_recovery": None,
            "minimum_mass_recovery": None,
            "average_grade_recovery": None,
            "average_yield": None,
            "average_enrichment_ratio": None,
        }

    def safe_mean(column):
        if column not in recovery_data.columns:
            return None

        values = pd.to_numeric(
            recovery_data[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            return None

        return float(values.mean())

    def safe_max(column):
        if column not in recovery_data.columns:
            return None

        values = pd.to_numeric(
            recovery_data[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            return None

        return float(values.max())

    def safe_min(column):
        if column not in recovery_data.columns:
            return None

        values = pd.to_numeric(
            recovery_data[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            return None

        return float(values.min())

    return {
        "number_of_runs": int(
            len(recovery_data)
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
    }


# ---------------------------------------------------------------------------
# SAVE CHART
# ---------------------------------------------------------------------------

def save_recovery_chart(
    df: pd.DataFrame,
    filepath: str,
    chart_type: str = "recovery",
    dpi: int = 300,
) -> str:
    """
    Create and save a recovery chart.

    Parameters
    ----------
    df : pandas.DataFrame
        LEBAS dataset.

    filepath : str
        Output image path.

    chart_type : str
        recovery
        grade_recovery
        comparison
        enrichment

    dpi : int
        Image resolution.

    Returns
    -------
    str
        Saved filepath.
    """

    if chart_type == "grade_recovery":

        figure = create_grade_recovery_figure(
            df
        )

    elif chart_type == "comparison":

        figure = create_recovery_comparison_figure(
            df
        )

    elif chart_type == "enrichment":

        figure = create_enrichment_ratio_figure(
            df
        )

    else:

        figure = create_recovery_figure(
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
        "Recovery chart saved to %s",
        filepath,
    )

    return filepath


# ---------------------------------------------------------------------------
# EXPORTS
# ---------------------------------------------------------------------------

__all__ = [
    "prepare_recovery_data",
    "calculate_recovery",
    "calculate_all_recoveries",
    "create_recovery_figure",
    "create_grade_recovery_figure",
    "create_recovery_comparison_figure",
    "create_enrichment_ratio_figure",
    "embed_recovery_chart",
    "get_recovery_summary",
    "save_recovery_chart",
]