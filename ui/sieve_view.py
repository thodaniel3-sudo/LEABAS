"""
LEBAS - Sieve Analysis View
File: ui/sieve_view.py

Purpose:
    Provides a complete sieve-analysis interface for LEBAS.

Important design principles:
    - The application is fully dataset-driven.
    - Sieve sizes are read directly from dataset.csv.
    - No sieve size is hard-coded.
    - The real laboratory may have 150 µm as its smallest sieve.
    - Any valid sieve-size distribution in the CSV is supported.
    - Repeated process information is not double-counted.
    - Sieve rows are preserved because each row represents a sieve fraction.
    - Missing and invalid values are handled safely.

Calculations:
    Percentage Retained
        = Mass Retained / Total Sieve Mass × 100

    Cumulative Percentage Retained
        = Running sum of Percentage Retained

    Percentage Passing
        = 100 - Cumulative Percentage Retained
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
from typing import Optional

import pandas as pd
import numpy as np

from config import Config
from services.data_loader import load_dataset


logger = logging.getLogger(__name__)


class SieveView:
    """Sieve analysis interface."""

    def __init__(self, parent, controller):
        self.controller = controller
        self.frame = ttk.Frame(parent)

        self.df = pd.DataFrame()
        self.sieve_df = pd.DataFrame()

        self.selected_sample = tk.StringVar(
            value="All Samples"
        )

        self.selected_replicate = tk.StringVar(
            value="All Replicates"
        )

        self.status_var = tk.StringVar(
            value="Ready"
        )

        # --------------------------------------------------------------
        # KPI VARIABLES
        # --------------------------------------------------------------

        self.kpi_vars = {
            "samples": tk.StringVar(value="0"),
            "fractions": tk.StringVar(value="0"),
            "total_mass": tk.StringVar(value="0"),
            "largest_sieve": tk.StringVar(value="N/A"),
            "smallest_sieve": tk.StringVar(value="N/A"),
            "max_retained": tk.StringVar(value="0"),
            "max_passing": tk.StringVar(value="0"),
            "balance": tk.StringVar(value="0"),
        }

        # --------------------------------------------------------------
        # DETAIL VARIABLES
        # --------------------------------------------------------------

        self.detail_vars = {
            "sample_id": tk.StringVar(value="N/A"),
            "material": tk.StringVar(value="N/A"),
            "date": tk.StringVar(value="N/A"),
            "operator": tk.StringVar(value="N/A"),
            "replicate": tk.StringVar(value="N/A"),
            "sieve_size": tk.StringVar(value="N/A"),
            "mass_retained": tk.StringVar(value="N/A"),
            "shaking_time": tk.StringVar(value="N/A"),
            "total_mass": tk.StringVar(value="N/A"),
            "percent_retained": tk.StringVar(value="N/A"),
            "cumulative_retained": tk.StringVar(value="N/A"),
            "percent_passing": tk.StringVar(value="N/A"),
        }

        self._configure_style()
        self._build_ui()
        self.refresh()

    # ==================================================================
    # PUBLIC METHODS
    # ==================================================================

    def get_frame(self):
        """Return the frame used by main.py."""
        return self.frame

    def refresh(self):
        """Reload dataset and refresh the complete sieve interface."""

        try:
            self.df = load_dataset()

            if self.df is None:
                self.df = pd.DataFrame()

            self.sieve_df = self._prepare_sieve_dataframe(
                self.df
            )

            self._update_filter_values()
            self._update_kpis()
            self._update_table()
            self._clear_details()

            if self.sieve_df.empty:
                self.status_var.set(
                    "No sieve-analysis data available"
                )
            else:
                self.status_var.set(
                    f"{len(self.sieve_df):,} sieve fraction(s) loaded"
                )

        except Exception as exc:
            logger.exception(
                "Failed to refresh sieve analysis view"
            )

            self.status_var.set(
                "Error loading sieve-analysis data"
            )

            try:
                messagebox.showerror(
                    "Sieve Analysis Error",
                    f"Unable to load sieve-analysis data.\n\n{exc}",
                )
            except Exception:
                pass

    # ==================================================================
    # STYLE
    # ==================================================================

    def _configure_style(self):
        """Configure ttk styles used by the sieve view."""

        style = ttk.Style()

        try:
            style.configure(
                "SieveTitle.TLabel",
                font=("Segoe UI", 20, "bold"),
            )

            style.configure(
                "SieveSubtitle.TLabel",
                font=("Segoe UI", 10),
            )

            style.configure(
                "SieveCardValue.TLabel",
                font=("Segoe UI", 17, "bold"),
            )

            style.configure(
                "SieveCardTitle.TLabel",
                font=("Segoe UI", 9),
            )

            style.configure(
                "Treeview",
                rowheight=30,
                font=("Segoe UI", 9),
            )

            style.configure(
                "Treeview.Heading",
                font=("Segoe UI", 9, "bold"),
            )

        except Exception:
            logger.exception(
                "Could not configure sieve styles"
            )

    # ==================================================================
    # BUILD USER INTERFACE
    # ==================================================================

    def _build_ui(self):
        """Build the complete sieve-analysis interface."""

        self.frame.grid_rowconfigure(
            3,
            weight=1,
        )

        self.frame.grid_columnconfigure(
            0,
            weight=1,
        )

        # --------------------------------------------------------------
        # HEADER
        # --------------------------------------------------------------

        header = ttk.Frame(self.frame)

        header.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=20,
            pady=(18, 10),
        )

        header.grid_columnconfigure(
            0,
            weight=1,
        )

        ttk.Label(
            header,
            text="Sieve Analysis",
            style="SieveTitle.TLabel",
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        version = getattr(
            Config,
            "VERSION",
            "",
        )

        subtitle = (
            "Particle-size distribution, retained mass "
            "and percentage passing"
        )

        if version:
            subtitle += f"  •  {version}"

        ttk.Label(
            header,
            text=subtitle,
            style="SieveSubtitle.TLabel",
        ).grid(
            row=1,
            column=0,
            sticky="w",
            pady=(3, 0),
        )

        # --------------------------------------------------------------
        # FILTERS
        # --------------------------------------------------------------

        filter_frame = ttk.LabelFrame(
            self.frame,
            text="Filters and Actions",
            padding=10,
        )

        filter_frame.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=20,
            pady=(0, 12),
        )

        filter_frame.grid_columnconfigure(
            1,
            weight=1,
        )

        filter_frame.grid_columnconfigure(
            3,
            weight=1,
        )

        ttk.Label(
            filter_frame,
            text="Sample:",
        ).grid(
            row=0,
            column=0,
            padx=(0, 6),
            pady=4,
            sticky="w",
        )

        self.sample_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.selected_sample,
            state="readonly",
            width=24,
        )

        self.sample_combo.grid(
            row=0,
            column=1,
            padx=(0, 15),
            pady=4,
            sticky="ew",
        )

        self.sample_combo.bind(
            "<<ComboboxSelected>>",
            self._on_filter_changed,
        )

        ttk.Label(
            filter_frame,
            text="Replicate:",
        ).grid(
            row=0,
            column=2,
            padx=(0, 6),
            pady=4,
            sticky="w",
        )

        self.replicate_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.selected_replicate,
            state="readonly",
            width=18,
        )

        self.replicate_combo.grid(
            row=0,
            column=3,
            padx=(0, 15),
            pady=4,
            sticky="ew",
        )

        self.replicate_combo.bind(
            "<<ComboboxSelected>>",
            self._on_filter_changed,
        )

        ttk.Button(
            filter_frame,
            text="↻ Refresh",
            command=self.refresh,
        ).grid(
            row=0,
            column=4,
            padx=5,
            pady=4,
        )

        ttk.Button(
            filter_frame,
            text="Reset Filters",
            command=self._reset_filters,
        ).grid(
            row=0,
            column=5,
            padx=(5, 0),
            pady=4,
        )

        # --------------------------------------------------------------
        # KPI CARDS
        # --------------------------------------------------------------

        kpi_container = ttk.Frame(
            self.frame
        )

        kpi_container.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=20,
            pady=(0, 12),
        )

        for column in range(8):
            kpi_container.grid_columnconfigure(
                column,
                weight=1,
                uniform="sieve_kpi",
            )

        self._create_kpi_card(
            kpi_container,
            0,
            "Samples",
            self.kpi_vars["samples"],
        )

        self._create_kpi_card(
            kpi_container,
            1,
            "Sieve Fractions",
            self.kpi_vars["fractions"],
        )

        self._create_kpi_card(
            kpi_container,
            2,
            "Total Mass",
            self.kpi_vars["total_mass"],
        )

        self._create_kpi_card(
            kpi_container,
            3,
            "Largest Sieve",
            self.kpi_vars["largest_sieve"],
        )

        self._create_kpi_card(
            kpi_container,
            4,
            "Smallest Sieve",
            self.kpi_vars["smallest_sieve"],
        )

        self._create_kpi_card(
            kpi_container,
            5,
            "Max % Retained",
            self.kpi_vars["max_retained"],
        )

        self._create_kpi_card(
            kpi_container,
            6,
            "Max % Passing",
            self.kpi_vars["max_passing"],
        )

        self._create_kpi_card(
            kpi_container,
            7,
            "Mass Balance",
            self.kpi_vars["balance"],
        )

        # --------------------------------------------------------------
        # MAIN CONTENT
        # --------------------------------------------------------------

        content = ttk.Frame(
            self.frame
        )

        content.grid(
            row=3,
            column=0,
            sticky="nsew",
            padx=20,
            pady=(0, 10),
        )

        content.grid_rowconfigure(
            0,
            weight=1,
        )

        content.grid_columnconfigure(
            0,
            weight=3,
        )

        content.grid_columnconfigure(
            1,
            weight=1,
        )

        # --------------------------------------------------------------
        # TABLE
        # --------------------------------------------------------------

        table_frame = ttk.LabelFrame(
            content,
            text="Sieve Fractions",
            padding=8,
        )

        table_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 10),
        )

        table_frame.grid_rowconfigure(
            0,
            weight=1,
        )

        table_frame.grid_columnconfigure(
            0,
            weight=1,
        )

        columns = (
            "sample_id",
            "material",
            "replicate",
            "sieve_size",
            "mass_retained",
            "total_mass",
            "percent_retained",
            "cumulative_retained",
            "percent_passing",
            "shaking_time",
        )

        self.tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            style="Treeview",
            selectmode="browse",
        )

        headings = {
            "sample_id": "Sample ID",
            "material": "Material",
            "replicate": "Replicate",
            "sieve_size": "Sieve Size (µm)",
            "mass_retained": "Mass Retained",
            "total_mass": "Total Sieve Mass",
            "percent_retained": "% Retained",
            "cumulative_retained": "Cumulative % Retained",
            "percent_passing": "% Passing",
            "shaking_time": "Shaking Time",
        }

        widths = {
            "sample_id": 120,
            "material": 120,
            "replicate": 80,
            "sieve_size": 115,
            "mass_retained": 110,
            "total_mass": 120,
            "percent_retained": 105,
            "cumulative_retained": 140,
            "percent_passing": 105,
            "shaking_time": 110,
        }

        for column in columns:
            self.tree.heading(
                column,
                text=headings[column],
            )

            self.tree.column(
                column,
                width=widths[column],
                minwidth=70,
                anchor="center",
                stretch=True,
            )

        y_scroll = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.tree.yview,
        )

        x_scroll = ttk.Scrollbar(
            table_frame,
            orient="horizontal",
            command=self.tree.xview,
        )

        self.tree.configure(
            yscrollcommand=y_scroll.set,
            xscrollcommand=x_scroll.set,
        )

        self.tree.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        y_scroll.grid(
            row=0,
            column=1,
            sticky="ns",
        )

        x_scroll.grid(
            row=1,
            column=0,
            sticky="ew",
        )

        self.tree.bind(
            "<<TreeviewSelect>>",
            self._on_row_selected,
        )

        # --------------------------------------------------------------
        # DETAILS
        # --------------------------------------------------------------

        details_frame = ttk.LabelFrame(
            content,
            text="Sieve Fraction Details",
            padding=12,
        )

        details_frame.grid(
            row=0,
            column=1,
            sticky="nsew",
        )

        details_frame.grid_columnconfigure(
            1,
            weight=1,
        )

        details = [
            ("Sample ID", "sample_id"),
            ("Material", "material"),
            ("Experiment Date", "date"),
            ("Operator", "operator"),
            ("Replicate", "replicate"),
            ("Sieve Size", "sieve_size"),
            ("Mass Retained", "mass_retained"),
            ("Shaking Time", "shaking_time"),
            ("Total Sieve Mass", "total_mass"),
            ("% Retained", "percent_retained"),
            ("Cumulative % Retained", "cumulative_retained"),
            ("% Passing", "percent_passing"),
        ]

        for row, (label, key) in enumerate(details):

            ttk.Label(
                details_frame,
                text=f"{label}:",
                font=("Segoe UI", 9, "bold"),
            ).grid(
                row=row,
                column=0,
                sticky="nw",
                padx=(0, 8),
                pady=5,
            )

            ttk.Label(
                details_frame,
                textvariable=self.detail_vars[key],
                wraplength=220,
            ).grid(
                row=row,
                column=1,
                sticky="nw",
                pady=5,
            )

        # --------------------------------------------------------------
        # STATUS
        # --------------------------------------------------------------

        status_frame = ttk.Frame(
            self.frame
        )

        status_frame.grid(
            row=4,
            column=0,
            sticky="ew",
            padx=20,
            pady=(0, 10),
        )

        ttk.Separator(
            status_frame,
            orient="horizontal",
        ).pack(
            fill="x",
            pady=(0, 5),
        )

        ttk.Label(
            status_frame,
            textvariable=self.status_var,
        ).pack(
            side="left"
        )

    # ==================================================================
    # KPI CARD
    # ==================================================================

    def _create_kpi_card(
        self,
        parent,
        column,
        title,
        variable,
    ):
        """Create one KPI card."""

        card = ttk.Frame(
            parent,
            relief="solid",
            borderwidth=1,
            padding=8,
        )

        card.grid(
            row=0,
            column=column,
            sticky="nsew",
            padx=3,
        )

        ttk.Label(
            card,
            text=title,
            style="SieveCardTitle.TLabel",
            anchor="center",
        ).pack(
            fill="x"
        )

        ttk.Label(
            card,
            textvariable=variable,
            style="SieveCardValue.TLabel",
            anchor="center",
        ).pack(
            fill="x",
            pady=(4, 0),
        )

    # ==================================================================
    # DATA PREPARATION
    # ==================================================================

    def _prepare_sieve_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prepare sieve-analysis data.

        IMPORTANT:
            Unlike crushing and grinding, sieve rows must NOT be
            deduplicated because every row represents a separate
            particle-size fraction.

        Example:

            SAMPLE-001 / 2000 µm
            SAMPLE-001 / 1400 µm
            SAMPLE-001 / 1000 µm
            SAMPLE-001 / 710 µm
            ...
            SAMPLE-001 / 150 µm

        All these rows must remain.
        """

        if df is None or df.empty:
            return pd.DataFrame()

        working = df.copy()

        # --------------------------------------------------------------
        # NORMALIZE COLUMN NAMES
        # --------------------------------------------------------------

        working.columns = [
            str(column).strip().lower()
            for column in working.columns
        ]

        if "sieve_size" not in working.columns:
            logger.warning(
                "Dataset does not contain 'sieve_size'."
            )
            return pd.DataFrame()

        if "mass_retained" not in working.columns:
            logger.warning(
                "Dataset does not contain 'mass_retained'."
            )
            return pd.DataFrame()

        # --------------------------------------------------------------
        # REQUIRED GROUPING COLUMNS
        # --------------------------------------------------------------

        if "sample_id" not in working.columns:
            working["sample_id"] = "Unknown"

        if "replicate" not in working.columns:
            working["replicate"] = "1"

        # --------------------------------------------------------------
        # NUMERIC CONVERSION
        # --------------------------------------------------------------

        numeric_columns = [
            "sieve_size",
            "mass_retained",
            "sieve_shaking_time",
            "feed_mass",
        ]

        for column in numeric_columns:
            if column in working.columns:
                working[column] = pd.to_numeric(
                    working[column],
                    errors="coerce",
                )

        # --------------------------------------------------------------
        # REMOVE INVALID SIEVE ROWS
        # --------------------------------------------------------------

        working = working[
            working["sieve_size"].notna()
            & working["mass_retained"].notna()
        ].copy()

        if working.empty:
            return pd.DataFrame()

        # Sieve opening cannot logically be zero or negative.
        working = working[
            working["sieve_size"] > 0
        ].copy()

        if working.empty:
            return pd.DataFrame()

        # --------------------------------------------------------------
        # NORMALIZE IDENTIFIERS
        # --------------------------------------------------------------

        working["sample_id"] = (
            working["sample_id"]
            .astype(str)
            .str.strip()
        )

        working["replicate"] = (
            working["replicate"]
            .astype(str)
            .str.strip()
        )

        # --------------------------------------------------------------
        # SORT SIEVE DATA
        # --------------------------------------------------------------
        #
        # Larger openings first.
        #
        # Example:
        #
        # 2000
        # 1400
        # 1000
        # ...
        # 150
        #
        # This makes cumulative retained and passing values meaningful.
        # --------------------------------------------------------------

        working = working.sort_values(
            by=[
                "sample_id",
                "replicate",
                "sieve_size",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        ).reset_index(
            drop=True
        )

        # --------------------------------------------------------------
        # CALCULATE RESULTS PER SAMPLE + REPLICATE
        # --------------------------------------------------------------

        working["total_sieve_mass"] = np.nan
        working["percent_retained"] = np.nan
        working["cumulative_retained"] = np.nan
        working["percent_passing"] = np.nan

        group_columns = [
            "sample_id",
            "replicate",
        ]

        for _, group_indices in working.groupby(
            group_columns,
            sort=False,
        ).groups.items():

            indices = list(group_indices)

            masses = pd.to_numeric(
                working.loc[
                    indices,
                    "mass_retained",
                ],
                errors="coerce",
            )

            valid_masses = masses[
                masses.notna()
            ]

            if valid_masses.empty:
                continue

            # ----------------------------------------------------------
            # TOTAL MASS RETAINED
            # ----------------------------------------------------------

            total_mass = valid_masses.sum()

            working.loc[
                indices,
                "total_sieve_mass"
            ] = total_mass

            if total_mass <= 0:
                continue

            # ----------------------------------------------------------
            # PERCENTAGE RETAINED
            # ----------------------------------------------------------

            percent_retained = (
                masses / total_mass
            ) * 100.0

            working.loc[
                indices,
                "percent_retained"
            ] = percent_retained

            # ----------------------------------------------------------
            # CUMULATIVE RETAINED
            # ----------------------------------------------------------

            cumulative = (
                percent_retained
                .fillna(0)
                .cumsum()
            )

            working.loc[
                indices,
                "cumulative_retained"
            ] = cumulative

            # ----------------------------------------------------------
            # PERCENTAGE PASSING
            # ----------------------------------------------------------

            passing = (
                100.0 - cumulative
            )

            # Avoid tiny floating-point errors such as
            # 99.99999999999999.
            passing = passing.clip(
                lower=0,
                upper=100,
            )

            working.loc[
                indices,
                "percent_passing"
            ] = passing

        return working.reset_index(
            drop=True
        )

    # ==================================================================
    # FILTER VALUES
    # ==================================================================

    def _update_filter_values(self):
        """Update sample and replicate filter values."""

        if self.sieve_df.empty:

            self.sample_combo["values"] = [
                "All Samples"
            ]

            self.replicate_combo["values"] = [
                "All Replicates"
            ]

            self.selected_sample.set(
                "All Samples"
            )

            self.selected_replicate.set(
                "All Replicates"
            )

            return

        samples = sorted(
            self.sieve_df["sample_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        self.sample_combo["values"] = (
            ["All Samples"] + samples
        )

        if (
            self.selected_sample.get()
            not in self.sample_combo["values"]
        ):
            self.selected_sample.set(
                "All Samples"
            )

        self._update_replicate_values()

    def _update_replicate_values(self):
        """Update replicate filter according to selected sample."""

        if self.sieve_df.empty:

            self.replicate_combo["values"] = [
                "All Replicates"
            ]

            self.selected_replicate.set(
                "All Replicates"
            )

            return

        filtered = self.sieve_df.copy()

        sample = self.selected_sample.get()

        if (
            sample
            and sample != "All Samples"
        ):
            filtered = filtered[
                filtered["sample_id"].astype(str)
                == str(sample)
            ]

        if filtered.empty:

            values = [
                "All Replicates"
            ]

        else:

            values = sorted(
                filtered["replicate"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            values = (
                ["All Replicates"]
                + values
            )

        self.replicate_combo["values"] = values

        if (
            self.selected_replicate.get()
            not in values
        ):
            self.selected_replicate.set(
                "All Replicates"
            )

    def _on_filter_changed(
        self,
        event=None,
    ):
        """Handle filter changes."""

        try:

            if (
                event is not None
                and event.widget == self.sample_combo
            ):
                self._update_replicate_values()

            self._update_kpis()
            self._update_table()
            self._clear_details()

        except Exception:
            logger.exception(
                "Failed to apply sieve filter"
            )

    def _reset_filters(self):
        """Reset sieve filters."""

        self.selected_sample.set(
            "All Samples"
        )

        self.selected_replicate.set(
            "All Replicates"
        )

        self._update_replicate_values()
        self._update_kpis()
        self._update_table()
        self._clear_details()

        if self.sieve_df.empty:
            self.status_var.set(
                "No sieve-analysis data available"
            )
        else:
            self.status_var.set(
                f"{len(self.sieve_df):,} sieve fraction(s) loaded"
            )

    def _get_filtered_dataframe(self) -> pd.DataFrame:
        """Return sieve data according to active filters."""

        if self.sieve_df.empty:
            return pd.DataFrame()

        filtered = self.sieve_df.copy()

        sample = self.selected_sample.get()
        replicate = self.selected_replicate.get()

        if (
            sample
            and sample != "All Samples"
        ):
            filtered = filtered[
                filtered["sample_id"].astype(str)
                == str(sample)
            ]

        if (
            replicate
            and replicate != "All Replicates"
        ):
            filtered = filtered[
                filtered["replicate"].astype(str)
                == str(replicate)
            ]

        return filtered.reset_index(
            drop=True
        )

    # ==================================================================
    # KPI CALCULATIONS
    # ==================================================================

    def _update_kpis(self):
        """Calculate and display sieve KPIs."""

        filtered = self._get_filtered_dataframe()

        if filtered.empty:

            for variable in self.kpi_vars.values():
                variable.set("0")

            self.kpi_vars[
                "largest_sieve"
            ].set("N/A")

            self.kpi_vars[
                "smallest_sieve"
            ].set("N/A")

            return

        # --------------------------------------------------------------
        # NUMBER OF SAMPLES
        # --------------------------------------------------------------

        self.kpi_vars["samples"].set(
            f"{filtered['sample_id'].nunique():,}"
        )

        # --------------------------------------------------------------
        # NUMBER OF SIEVE FRACTIONS
        # --------------------------------------------------------------

        self.kpi_vars["fractions"].set(
            f"{len(filtered):,}"
        )

        # --------------------------------------------------------------
        # TOTAL MASS
        # --------------------------------------------------------------
        #
        # Do not simply sum total_sieve_mass because that value is
        # repeated on every sieve row.
        #
        # Instead, calculate the average unique total per experimental
        # run.
        # --------------------------------------------------------------

        if "total_sieve_mass" in filtered.columns:

            total_by_run = (
                filtered
                .groupby(
                    [
                        "sample_id",
                        "replicate",
                    ],
                    dropna=False,
                )[
                    "total_sieve_mass"
                ]
                .first()
            )

            total_mass = pd.to_numeric(
                total_by_run,
                errors="coerce",
            ).dropna()

            if not total_mass.empty:
                self.kpi_vars[
                    "total_mass"
                ].set(
                    f"{total_mass.mean():,.3f}"
                )
            else:
                self.kpi_vars[
                    "total_mass"
                ].set("N/A")

        # --------------------------------------------------------------
        # LARGEST SIEVE
        # --------------------------------------------------------------

        sieve_values = pd.to_numeric(
            filtered["sieve_size"],
            errors="coerce",
        ).dropna()

        if not sieve_values.empty:

            self.kpi_vars[
                "largest_sieve"
            ].set(
                self._format_number(
                    sieve_values.max()
                )
            )

            self.kpi_vars[
                "smallest_sieve"
            ].set(
                self._format_number(
                    sieve_values.min()
                )
            )

        # --------------------------------------------------------------
        # MAXIMUM PERCENT RETAINED
        # --------------------------------------------------------------

        percent_retained = pd.to_numeric(
            filtered["percent_retained"],
            errors="coerce",
        ).dropna()

        if not percent_retained.empty:

            self.kpi_vars[
                "max_retained"
            ].set(
                self._format_number(
                    percent_retained.max(),
                    suffix="%",
                )
            )
        else:
            self.kpi_vars[
                "max_retained"
            ].set("N/A")

        # --------------------------------------------------------------
        # MAXIMUM PERCENT PASSING
        # --------------------------------------------------------------

        percent_passing = pd.to_numeric(
            filtered["percent_passing"],
            errors="coerce",
        ).dropna()

        if not percent_passing.empty:

            self.kpi_vars[
                "max_passing"
            ].set(
                self._format_number(
                    percent_passing.max(),
                    suffix="%",
                )
            )
        else:
            self.kpi_vars[
                "max_passing"
            ].set("N/A")

        # --------------------------------------------------------------
        # MASS BALANCE
        # --------------------------------------------------------------

        balances = []

        for _, group in filtered.groupby(
            [
                "sample_id",
                "replicate",
            ],
            dropna=False,
        ):

            masses = pd.to_numeric(
                group["mass_retained"],
                errors="coerce",
            ).dropna()

            if masses.empty:
                continue

            total = masses.sum()

            if total <= 0:
                continue

            balances.append(
                total
            )

        if balances:

            balance_value = float(
                np.mean(balances)
            )

            self.kpi_vars[
                "balance"
            ].set(
                f"{balance_value:,.3f}"
            )

        else:

            self.kpi_vars[
                "balance"
            ].set("N/A")

    # ==================================================================
    # TABLE
    # ==================================================================

    def _update_table(self):
        """Refresh the sieve-analysis table."""

        for item in self.tree.get_children():
            self.tree.delete(item)

        filtered = self._get_filtered_dataframe()

        if filtered.empty:
            return

        for index, row in filtered.iterrows():

            values = (
                self._display_value(
                    row.get("sample_id")
                ),

                self._display_value(
                    row.get("material_type")
                ),

                self._display_value(
                    row.get("replicate")
                ),

                self._format_number(
                    row.get("sieve_size")
                ),

                self._format_number(
                    row.get("mass_retained")
                ),

                self._format_number(
                    row.get("total_sieve_mass")
                ),

                self._format_number(
                    row.get("percent_retained"),
                    suffix="%",
                ),

                self._format_number(
                    row.get("cumulative_retained"),
                    suffix="%",
                ),

                self._format_number(
                    row.get("percent_passing"),
                    suffix="%",
                ),

                self._format_number(
                    row.get("sieve_shaking_time")
                ),
            )

            self.tree.insert(
                "",
                "end",
                iid=str(index),
                values=values,
            )

    # ==================================================================
    # SELECTED ROW DETAILS
    # ==================================================================

    def _on_row_selected(
        self,
        event=None,
    ):
        """Display details of the selected sieve fraction."""

        selection = self.tree.selection()

        if not selection:
            return

        try:

            index = int(
                selection[0]
            )

            filtered = (
                self._get_filtered_dataframe()
            )

            if (
                index < 0
                or index >= len(filtered)
            ):
                return

            row = filtered.iloc[index]

            self.detail_vars[
                "sample_id"
            ].set(
                self._display_value(
                    row.get("sample_id")
                )
            )

            self.detail_vars[
                "material"
            ].set(
                self._display_value(
                    row.get("material_type")
                )
            )

            self.detail_vars[
                "date"
            ].set(
                self._format_date(
                    row.get("experiment_date")
                )
            )

            self.detail_vars[
                "operator"
            ].set(
                self._display_value(
                    row.get("operator")
                )
            )

            self.detail_vars[
                "replicate"
            ].set(
                self._display_value(
                    row.get("replicate")
                )
            )

            self.detail_vars[
                "sieve_size"
            ].set(
                self._format_number(
                    row.get("sieve_size"),
                    suffix=" µm",
                )
            )

            self.detail_vars[
                "mass_retained"
            ].set(
                self._format_number(
                    row.get("mass_retained")
                )
            )

            self.detail_vars[
                "shaking_time"
            ].set(
                self._format_number(
                    row.get("sieve_shaking_time")
                )
            )

            self.detail_vars[
                "total_mass"
            ].set(
                self._format_number(
                    row.get("total_sieve_mass")
                )
            )

            self.detail_vars[
                "percent_retained"
            ].set(
                self._format_number(
                    row.get("percent_retained"),
                    suffix="%",
                )
            )

            self.detail_vars[
                "cumulative_retained"
            ].set(
                self._format_number(
                    row.get("cumulative_retained"),
                    suffix="%",
                )
            )

            self.detail_vars[
                "percent_passing"
            ].set(
                self._format_number(
                    row.get("percent_passing"),
                    suffix="%",
                )
            )

        except Exception:
            logger.exception(
                "Failed to display selected sieve record"
            )

    def _clear_details(self):
        """Clear the sieve details panel."""

        for variable in self.detail_vars.values():
            variable.set("N/A")

    # ==================================================================
    # FORMATTING HELPERS
    # ==================================================================

    @staticmethod
    def _display_value(
        value,
    ) -> str:
        """Convert a value to a safe display string."""

        if value is None:
            return "N/A"

        try:
            if pd.isna(value):
                return "N/A"
        except Exception:
            pass

        text = str(value).strip()

        if not text:
            return "N/A"

        if text.lower() in {
            "nan",
            "none",
            "nat",
        }:
            return "N/A"

        return text

    @staticmethod
    def _format_number(
        value,
        suffix="",
    ) -> str:
        """Safely format a numeric value."""

        if value is None:
            return "N/A"

        try:

            numeric = float(value)

            if not np.isfinite(numeric):
                return "N/A"

            return (
                f"{numeric:,.3f}"
                f"{suffix}"
            )

        except (
            TypeError,
            ValueError,
        ):
            return "N/A"

    @staticmethod
    def _format_date(
        value,
    ) -> str:
        """Safely format an experiment date."""

        if value is None:
            return "N/A"

        try:
            if pd.isna(value):
                return "N/A"
        except Exception:
            pass

        try:

            parsed = pd.to_datetime(
                value,
                errors="coerce",
            )

            if pd.isna(parsed):
                return str(value)

            return parsed.strftime(
                "%Y-%m-%d"
            )

        except Exception:
            return str(value)

    # ==================================================================
    # PROGRAMMATIC ACCESS
    # ==================================================================

    def get_current_data(
        self,
    ) -> pd.DataFrame:
        """
        Return the currently filtered sieve dataset.

        Useful for charts, reports and other application modules.
        """

        return (
            self._get_filtered_dataframe()
            .copy()
        )

    def get_selected_record(
        self,
    ) -> Optional[pd.Series]:
        """
        Return the currently selected sieve record.

        Returns:
            pandas.Series or None
        """

        selection = self.tree.selection()

        if not selection:
            return None

        try:

            index = int(
                selection[0]
            )

            filtered = (
                self._get_filtered_dataframe()
            )

            if (
                0 <= index < len(filtered)
            ):
                return filtered.iloc[index]

        except Exception:
            logger.exception(
                "Failed to retrieve selected sieve record"
            )

        return None


__all__ = ["SieveView"]