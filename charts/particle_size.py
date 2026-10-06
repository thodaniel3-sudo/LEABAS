"""
LEBAS - Particle Size Distribution Chart Module

File:
    charts/particle_size.py

Purpose:
    Provides reusable functions for preparing and plotting particle size
    distribution data from the LEBAS dataset.

Important:
    - Sieve sizes are read dynamically from dataset.csv.
    - No sieve size is hard-coded.
    - A real dataset with a minimum sieve size of 150 µm is fully supported.
    - Multiple sieve records belonging to the same sample/replicate are handled.
    - Duplicate sieve-size records are combined by summing mass retained.
    - Process-level fields are not double-counted.

Main functions:
    prepare_sieve_data()
    calculate_particle_size_distribution()
    create_particle_size_figure()
    create_retained_distribution_figure()
    embed_particle_size_chart()

Author:
    LEBAS Development Team
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:
    FigureCanvasTkAgg = None


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# REQUIRED DATASET COLUMNS
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "sample_id",
    "replicate",
    "sieve_size",
    "mass_retained",
}


# ---------------------------------------------------------------------------
# BASIC HELPERS
# ---------------------------------------------------------------------------

def _validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate that the supplied DataFrame contains the columns required
    for particle size analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to validate.

    Raises
    ------
    ValueError
        If df is missing required columns.
    """
    if df is None:
        raise ValueError("Dataset cannot be None.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Dataset must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    missing = REQUIRED_COLUMNS.difference(df.columns)

    if missing:
        raise ValueError(
            "Dataset is missing required sieve-analysis columns: "
            + ", ".join(sorted(missing))
        )


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a pandas Series to numeric values safely."""
    return pd.to_numeric(series, errors="coerce")


def _normalise_replicate(value) -> str:
    """
    Convert replicate values to a consistent string representation.

    This prevents mismatches such as:
        1
        1.0
        "1"
    """
    if pd.isna(value):
        return ""

    try:
        number = float(value)

        if number.is_integer():
            return str(int(number))

    except (ValueError, TypeError):
        pass

    return str(value).strip()


def _filter_sample(
    df: pd.DataFrame,
    sample_id: Optional[str] = None,
    replicate: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filter the dataset by sample and replicate.

    If no sample_id is supplied, all samples are retained.
    If replicate is supplied without sample_id, replicate filtering is applied.
    """
    result = df.copy()

    if sample_id is not None and str(sample_id).strip():
        target = str(sample_id).strip()

        result = result[
            result["sample_id"].astype(str).str.strip() == target
        ]

    if replicate is not None and str(replicate).strip():
        target_rep = _normalise_replicate(replicate)

        result = result[
            result["replicate"].apply(_normalise_replicate) == target_rep
        ]

    return result


# ---------------------------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------------------------

def prepare_sieve_data(
    df: pd.DataFrame,
    sample_id: Optional[str] = None,
    replicate: Optional[str] = None,
) -> pd.DataFrame:
    """
    Prepare sieve records for particle size analysis.

    The function:

    1. Validates the dataset.
    2. Filters by sample and/or replicate.
    3. Converts sieve size and mass retained to numeric values.
    4. Removes invalid sieve records.
    5. Groups duplicate sieve sizes.
    6. Calculates percentage retained.
    7. Calculates cumulative retained.
    8. Calculates cumulative passing.

    Parameters
    ----------
    df : pandas.DataFrame
        Complete LEBAS dataset.

    sample_id : str, optional
        Sample to analyse.

    replicate : str, optional
        Replicate to analyse.

    Returns
    -------
    pandas.DataFrame
        Prepared sieve-analysis DataFrame.

    Notes
    -----
    Sieve sizes are NOT hard-coded. Whatever numeric sieve sizes exist
    in the dataset are used.
    """
    _validate_dataframe(df)

    data = _filter_sample(
        df,
        sample_id=sample_id,
        replicate=replicate,
    )

    if data.empty:
        return pd.DataFrame(
            columns=[
                "sieve_size",
                "mass_retained",
                "percent_retained",
                "cumulative_retained",
                "cumulative_passing",
            ]
        )

    data = data.copy()

    data["sieve_size"] = _safe_numeric(data["sieve_size"])
    data["mass_retained"] = _safe_numeric(data["mass_retained"])

    # Remove invalid records.
    data = data.dropna(
        subset=["sieve_size", "mass_retained"]
    )

    # Sieve size must be positive for logarithmic plotting.
    data = data[data["sieve_size"] > 0]

    # Negative retained mass is not physically meaningful.
    data = data[data["mass_retained"] >= 0]

    if data.empty:
        return pd.DataFrame(
            columns=[
                "sieve_size",
                "mass_retained",
                "percent_retained",
                "cumulative_retained",
                "cumulative_passing",
            ]
        )

    # If duplicate rows exist for the same sieve size,
    # combine their retained masses.
    data = (
        data.groupby("sieve_size", as_index=False)["mass_retained"]
        .sum()
    )

    # Sort from largest sieve opening to smallest.
    data = data.sort_values(
        by="sieve_size",
        ascending=False
    ).reset_index(drop=True)

    total_mass = data["mass_retained"].sum()

    if total_mass <= 0:
        data["percent_retained"] = 0.0
        data["cumulative_retained"] = 0.0
        data["cumulative_passing"] = 100.0
        return data

    # Percentage retained on each sieve.
    data["percent_retained"] = (
        data["mass_retained"] / total_mass * 100.0
    )

    # Cumulative percentage retained.
    data["cumulative_retained"] = (
        data["percent_retained"].cumsum()
    )

    # Percentage passing.
    data["cumulative_passing"] = (
        100.0 - data["cumulative_retained"]
    )

    # Avoid tiny floating-point errors such as -1e-14.
    data["cumulative_passing"] = (
        data["cumulative_passing"]
        .clip(lower=0.0, upper=100.0)
    )

    return data


# ---------------------------------------------------------------------------
# PARTICLE SIZE DISTRIBUTION CALCULATION
# ---------------------------------------------------------------------------

def calculate_particle_size_distribution(
    df: pd.DataFrame,
    sample_id: Optional[str] = None,
    replicate: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate particle size distribution values.

    This is a convenience wrapper around prepare_sieve_data().

    Returns columns:
        sieve_size
        mass_retained
        percent_retained
        cumulative_retained
        cumulative_passing

    Parameters
    ----------
    df : pandas.DataFrame
        LEBAS dataset.

    sample_id : str, optional
        Sample identifier.

    replicate : str, optional
        Replicate identifier.

    Returns
    -------
    pandas.DataFrame
        Particle size distribution table.
    """
    return prepare_sieve_data(
        df,
        sample_id=sample_id,
        replicate=replicate,
    )


# ---------------------------------------------------------------------------
# ALL-RUN PARTICLE SIZE DISTRIBUTION
# ---------------------------------------------------------------------------

def calculate_all_runs_particle_size_distribution(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate particle size distributions for every sample/replicate.

    Each sample/replicate is normalized independently before being combined.

    This prevents a sample with a larger feed mass from automatically
    dominating the particle-size distribution.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing:

        sample_id
        replicate
        sieve_size
        mass_retained
        percent_retained
        cumulative_retained
        cumulative_passing
    """
    _validate_dataframe(df)

    records: List[pd.DataFrame] = []

    data = df.copy()

    data["sample_id"] = data["sample_id"].astype(str).str.strip()
    data["replicate_normalized"] = data["replicate"].apply(
        _normalise_replicate
    )

    groups = data.groupby(
        ["sample_id", "replicate_normalized"],
        dropna=False,
    )

    for (sample, rep), group in groups:

        try:
            prepared = prepare_sieve_data(
                group,
                sample_id=None,
                replicate=None,
            )

            if prepared.empty:
                continue

            prepared.insert(
                0,
                "replicate",
                rep,
            )

            prepared.insert(
                0,
                "sample_id",
                sample,
            )

            records.append(prepared)

        except Exception as exc:
            logger.warning(
                "Could not process sieve data for %s / Replicate %s: %s",
                sample,
                rep,
                exc,
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "replicate",
                "sieve_size",
                "mass_retained",
                "percent_retained",
                "cumulative_retained",
                "cumulative_passing",
            ]
        )

    return pd.concat(
        records,
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# FIGURE CREATION
# ---------------------------------------------------------------------------

def create_particle_size_figure(
    df: pd.DataFrame,
    sample_id: Optional[str] = None,
    replicate: Optional[str] = None,
    ax=None,
    title: Optional[str] = None,
    sieve_unit: str = "µm",
) -> Figure:
    """
    Create a cumulative particle size distribution chart.

    The chart displays:

        X-axis:
            Sieve opening

        Y-axis:
            Cumulative percentage passing

    The x-axis uses a logarithmic scale, which is appropriate for
    particle-size distributions covering multiple size ranges.

    Parameters
    ----------
    df : pandas.DataFrame
        LEBAS dataset.

    sample_id : str, optional
        Sample to plot.

    replicate : str, optional
        Replicate to plot.

    ax : matplotlib.axes.Axes, optional
        Existing axes.

    title : str, optional
        Custom chart title.

    sieve_unit : str
        Unit displayed on the x-axis.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the chart.
    """
    distribution = calculate_particle_size_distribution(
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

    if distribution.empty:
        ax.text(
            0.5,
            0.5,
            "No valid sieve data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )

        ax.set_title(
            title or "Particle Size Distribution"
        )

        ax.set_xlabel(
            f"Sieve Size ({sieve_unit})"
        )

        ax.set_ylabel(
            "Cumulative Passing (%)"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        fig.tight_layout()

        return fig

    x = distribution["sieve_size"].to_numpy()
    y = distribution["cumulative_passing"].to_numpy()

    ax.plot(
        x,
        y,
        marker="o",
        linewidth=2,
        markersize=5,
    )

    ax.set_xscale("log")

    # Conventional PSD presentation:
    # larger sieve openings on the left,
    # smaller openings on the right.
    ax.invert_xaxis()

    ax.set_ylim(
        0,
        100,
    )

    ax.set_xlabel(
        f"Sieve Size ({sieve_unit})"
    )

    ax.set_ylabel(
        "Cumulative Passing (%)"
    )

    if title:
        chart_title = title
    elif sample_id:
        if replicate:
            chart_title = (
                f"Particle Size Distribution - "
                f"{sample_id} - Replicate {replicate}"
            )
        else:
            chart_title = (
                f"Particle Size Distribution - {sample_id}"
            )
    else:
        chart_title = "Particle Size Distribution"

    ax.set_title(chart_title)

    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )

    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# RETAINED MASS DISTRIBUTION
# ---------------------------------------------------------------------------

def create_retained_distribution_figure(
    df: pd.DataFrame,
    sample_id: Optional[str] = None,
    replicate: Optional[str] = None,
    ax=None,
    title: Optional[str] = None,
    sieve_unit: str = "µm",
) -> Figure:
    """
    Create a percentage-retained particle size chart.

    Parameters
    ----------
    df : pandas.DataFrame
        LEBAS dataset.

    sample_id : str, optional
        Sample identifier.

    replicate : str, optional
        Replicate identifier.

    ax : matplotlib.axes.Axes, optional
        Existing matplotlib axes.

    title : str, optional
        Custom title.

    sieve_unit : str
        Unit displayed on x-axis.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure.
    """
    distribution = calculate_particle_size_distribution(
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

    if distribution.empty:
        ax.text(
            0.5,
            0.5,
            "No valid sieve data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )

        ax.set_title(
            title or "Sieve Mass Distribution"
        )

        ax.set_xlabel(
            f"Sieve Size ({sieve_unit})"
        )

        ax.set_ylabel(
            "Mass Retained (%)"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        fig.tight_layout()

        return fig

    x = distribution["sieve_size"].to_numpy()
    y = distribution["percent_retained"].to_numpy()

    ax.bar(
        x,
        y,
        width=np.maximum(
            x * 0.08,
            np.finfo(float).eps,
        ),
        align="center",
    )

    ax.set_xscale("log")
    ax.invert_xaxis()

    ax.set_xlabel(
        f"Sieve Size ({sieve_unit})"
    )

    ax.set_ylabel(
        "Mass Retained (%)"
    )

    if title:
        chart_title = title
    elif sample_id:
        if replicate:
            chart_title = (
                f"Sieve Mass Distribution - "
                f"{sample_id} - Replicate {replicate}"
            )
        else:
            chart_title = (
                f"Sieve Mass Distribution - {sample_id}"
            )
    else:
        chart_title = "Sieve Mass Distribution"

    ax.set_title(chart_title)

    ax.grid(
        True,
        which="both",
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# EMBEDDING INTO TKINTER
# ---------------------------------------------------------------------------

def embed_particle_size_chart(
    parent,
    df: pd.DataFrame,
    sample_id: Optional[str] = None,
    replicate: Optional[str] = None,
    figure: Optional[Figure] = None,
    **kwargs,
):
    """
    Embed a particle size chart inside a Tkinter widget.

    Parameters
    ----------
    parent
        Tkinter parent frame.

    df : pandas.DataFrame
        LEBAS dataset.

    sample_id : str, optional
        Sample identifier.

    replicate : str, optional
        Replicate identifier.

    figure : matplotlib.figure.Figure, optional
        Existing figure. If omitted, a new figure is created.

    **kwargs
        Additional arguments passed to create_particle_size_figure().

    Returns
    -------
    FigureCanvasTkAgg
        Embedded matplotlib canvas.

    Raises
    ------
    RuntimeError
        If FigureCanvasTkAgg is unavailable.
    """
    if FigureCanvasTkAgg is None:
        raise RuntimeError(
            "Tkinter Matplotlib integration is unavailable. "
            "Install a compatible matplotlib package."
        )

    if figure is None:
        figure = create_particle_size_figure(
            df,
            sample_id=sample_id,
            replicate=replicate,
            **kwargs,
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
# MULTI-SAMPLE CHART
# ---------------------------------------------------------------------------

def create_multi_sample_particle_size_figure(
    df: pd.DataFrame,
    ax=None,
    title: str = "Particle Size Distribution",
    sieve_unit: str = "µm",
) -> Figure:
    """
    Plot particle size distributions for multiple sample/replicate runs.

    Each sample/replicate is normalized independently.

    This function is useful for comparing different experimental runs.

    Parameters
    ----------
    df : pandas.DataFrame
        LEBAS dataset.

    ax : matplotlib.axes.Axes, optional
        Existing axes.

    title : str
        Chart title.

    sieve_unit : str
        Sieve-size unit.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure.
    """
    all_data = calculate_all_runs_particle_size_distribution(df)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(10, 6)
        )
    else:
        fig = ax.figure

    ax.clear()

    if all_data.empty:
        ax.text(
            0.5,
            0.5,
            "No valid sieve data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )

        ax.set_title(title)
        ax.set_xlabel(f"Sieve Size ({sieve_unit})")
        ax.set_ylabel("Cumulative Passing (%)")

        fig.tight_layout()

        return fig

    for (sample_id, replicate), group in all_data.groupby(
        ["sample_id", "replicate"],
        dropna=False,
    ):
        group = group.sort_values(
            "sieve_size",
            ascending=False,
        )

        label = str(sample_id)

        if str(replicate).strip():
            label += f" - R{replicate}"

        ax.plot(
            group["sieve_size"],
            group["cumulative_passing"],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=label,
        )

    ax.set_xscale("log")
    ax.invert_xaxis()

    ax.set_ylim(
        0,
        100,
    )

    ax.set_xlabel(
        f"Sieve Size ({sieve_unit})"
    )

    ax.set_ylabel(
        "Cumulative Passing (%)"
    )

    ax.set_title(title)

    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )

    # Only display a legend if there is a manageable
    # number of experimental runs.
    run_count = all_data[
        ["sample_id", "replicate"]
    ].drop_duplicates().shape[0]

    if run_count <= 15:
        ax.legend(
            loc="best",
            fontsize=8,
        )

    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# SIMPLE DATA SUMMARY
# ---------------------------------------------------------------------------

def get_particle_size_summary(
    df: pd.DataFrame,
    sample_id: Optional[str] = None,
    replicate: Optional[str] = None,
) -> dict:
    """
    Return a small summary of the particle-size distribution.

    Returns
    -------
    dict
        Summary information including:

        - number_of_sieves
        - minimum_sieve_size
        - maximum_sieve_size
        - total_sieve_mass
        - maximum_retained_percent
        - minimum_cumulative_passing
        - maximum_cumulative_passing
    """
    distribution = calculate_particle_size_distribution(
        df,
        sample_id=sample_id,
        replicate=replicate,
    )

    if distribution.empty:
        return {
            "number_of_sieves": 0,
            "minimum_sieve_size": None,
            "maximum_sieve_size": None,
            "total_sieve_mass": 0.0,
            "maximum_retained_percent": None,
            "minimum_cumulative_passing": None,
            "maximum_cumulative_passing": None,
        }

    return {
        "number_of_sieves": int(
            len(distribution)
        ),
        "minimum_sieve_size": float(
            distribution["sieve_size"].min()
        ),
        "maximum_sieve_size": float(
            distribution["sieve_size"].max()
        ),
        "total_sieve_mass": float(
            distribution["mass_retained"].sum()
        ),
        "maximum_retained_percent": float(
            distribution["percent_retained"].max()
        ),
        "minimum_cumulative_passing": float(
            distribution["cumulative_passing"].min()
        ),
        "maximum_cumulative_passing": float(
            distribution["cumulative_passing"].max()
        ),
    }


# ---------------------------------------------------------------------------
# SAVE CHART
# ---------------------------------------------------------------------------

def save_particle_size_chart(
    df: pd.DataFrame,
    filepath: str,
    sample_id: Optional[str] = None,
    replicate: Optional[str] = None,
    dpi: int = 300,
    sieve_unit: str = "µm",
) -> str:
    """
    Create and save a particle size distribution chart.

    Parameters
    ----------
    df : pandas.DataFrame
        LEBAS dataset.

    filepath : str
        Destination image path.

    sample_id : str, optional
        Sample identifier.

    replicate : str, optional
        Replicate identifier.

    dpi : int
        Image resolution.

    sieve_unit : str
        Sieve-size unit.

    Returns
    -------
    str
        Saved filepath.
    """
    figure = create_particle_size_figure(
        df,
        sample_id=sample_id,
        replicate=replicate,
        sieve_unit=sieve_unit,
    )

    figure.savefig(
        filepath,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(figure)

    logger.info(
        "Particle size chart saved to %s",
        filepath,
    )

    return filepath


# ---------------------------------------------------------------------------
# MODULE EXPORTS
# ---------------------------------------------------------------------------

__all__ = [
    "prepare_sieve_data",
    "calculate_particle_size_distribution",
    "calculate_all_runs_particle_size_distribution",
    "create_particle_size_figure",
    "create_retained_distribution_figure",
    "create_multi_sample_particle_size_figure",
    "embed_particle_size_chart",
    "get_particle_size_summary",
    "save_particle_size_chart",
]