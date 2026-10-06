"""
LEBAS - Results View
File: ui/results_view.py

Purpose:
    Central results and analysis interface for LEBAS.

    This view combines:
        - Crushing results
        - Grinding results
        - Sieve analysis
        - Beneficiation results
        - Material balance
        - Basic statistics

Important:
    - All laboratory data comes from dataset.csv.
    - No experimental results are hard-coded.
    - Repeated sieve rows are reduced to one process-level record
      where appropriate.
    - The sieve analysis itself is not reduced because each sieve
      fraction is an independent observation.
    - Missing values are handled safely.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
from typing import Optional

import pandas as pd
import numpy as np

from config import Config
from services.data_loader import load_dataset

from calculations.material_balance import (
    calculate_material_balance,
)

logger = logging.getLogger(__name__)


class ResultsView:
    """Central results and analysis interface."""

    def __init__(self, parent, controller):
        self.controller = controller
        self.frame = ttk.Frame(parent)

        self.df = pd.DataFrame()
        self.process_df = pd.DataFrame()

        self.selected_sample = tk.StringVar(
            value="All Samples"
        )

        self.selected_replicate = tk.StringVar(
            value="All Replicates"
        )

        self.selected_result = tk.StringVar(
            value="Summary"
        )

        self.status_var = tk.StringVar(
            value="Ready"
        )

        self.kpi_vars = {
            "samples": tk.StringVar(value="0"),
            "feed_mass": tk.StringVar(value="0"),
            "concentrate_mass": tk.StringVar(value="0"),
            "recovery": tk.StringVar(value="0"),
            "yield": tk.StringVar(value="0"),
            "balance": tk.StringVar(value="0"),
            "sieve_records": tk.StringVar(value="0"),
            "avg_feed_grade": tk.StringVar(value="0"),
        }

        self._configure_style()
        self._build_ui()
        self.refresh()

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def get_frame(self):
        """Return the frame used by main.py."""
        return self.frame

    def refresh(self):
        """Reload data and refresh the results page."""

        try:
            self.df = load_dataset()

            if self.df is None:
                self.df = pd.DataFrame()

            self.process_df = (
                self._prepare_process_dataframe(
                    self.df
                )
            )

            self._update_filters()
            self._update_kpis()
            self._update_result_table()
            self._clear_details()

            if self.df.empty:
                self.status_var.set(
                    "No dataset loaded"
                )
            else:
                self.status_var.set(
                    f"{len(self.df):,} dataset row(s) loaded"
                )

        except Exception as exc:
            logger.exception(
                "Failed to refresh results view"
            )

            self.status_var.set(
                "Error loading results"
            )

            try:
                messagebox.showerror(
                    "Results Error",
                    f"Unable to load results.\n\n{exc}",
                )
            except Exception:
                pass

    # ==================================================================
    # STYLE
    # ==================================================================

    def _configure_style(self):
        """Configure styles used by this view."""

        style = ttk.Style()

        try:
            style.configure(
                "ResultsTitle.TLabel",
                font=("Segoe UI", 20, "bold"),
            )

            style.configure(
                "ResultsSubtitle.TLabel",
                font=("Segoe UI", 10),
            )

            style.configure(
                "ResultsCardTitle.TLabel",
                font=("Segoe UI", 9),
            )

            style.configure(
                "ResultsCardValue.TLabel",
                font=("Segoe UI", 17, "bold"),
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
                "Could not configure ResultsView styles"
            )

    # ==================================================================
    # UI
    # ==================================================================

    def _build_ui(self):
        """Build the complete results interface."""

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

        header = ttk.Frame(
            self.frame
        )

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
            text="Results & Analysis",
            style="ResultsTitle.TLabel",
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
            "Integrated laboratory results, process performance "
            "and material-balance analysis"
        )

        if version:
            subtitle += f"  •  {version}"

        ttk.Label(
            header,
            text=subtitle,
            style="ResultsSubtitle.TLabel",
        ).grid(
            row=1,
            column=0,
            sticky="w",
            pady=(3, 0),
        )

        # --------------------------------------------------------------
        # FILTER BAR
        # --------------------------------------------------------------

        filters = ttk.LabelFrame(
            self.frame,
            text="Filters",
            padding=10,
        )

        filters.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=20,
            pady=(0, 12),
        )

        filters.grid_columnconfigure(
            1,
            weight=1,
        )

        filters.grid_columnconfigure(
            3,
            weight=1,
        )

        ttk.Label(
            filters,
            text="Sample:",
        ).grid(
            row=0,
            column=0,
            padx=(0, 6),
            pady=4,
            sticky="w",
        )

        self.sample_combo = ttk.Combobox(
            filters,
            textvariable=self.selected_sample,
            state="readonly",
            width=22,
        )

        self.sample_combo.grid(
            row=0,
            column=1,
            padx=(0, 15),
            sticky="ew",
        )

        self.sample_combo.bind(
            "<<ComboboxSelected>>",
            self._on_filter_changed,
        )

        ttk.Label(
            filters,
            text="Replicate:",
        ).grid(
            row=0,
            column=2,
            padx=(0, 6),
            sticky="w",
        )

        self.replicate_combo = ttk.Combobox(
            filters,
            textvariable=self.selected_replicate,
            state="readonly",
            width=18,
        )

        self.replicate_combo.grid(
            row=0,
            column=3,
            padx=(0, 15),
            sticky="ew",
        )

        self.replicate_combo.bind(
            "<<ComboboxSelected>>",
            self._on_filter_changed,
        )

        ttk.Button(
            filters,
            text="↻ Refresh",
            command=self.refresh,
        ).grid(
            row=0,
            column=4,
            padx=5,
        )

        ttk.Button(
            filters,
            text="Reset",
            command=self._reset_filters,
        ).grid(
            row=0,
            column=5,
            padx=(5, 0),
        )

        # --------------------------------------------------------------
        # KPI CARDS
        # --------------------------------------------------------------

        kpi_frame = ttk.Frame(
            self.frame
        )

        kpi_frame.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=20,
            pady=(0, 12),
        )

        for column in range(8):
            kpi_frame.grid_columnconfigure(
                column,
                weight=1,
                uniform="results_kpi",
            )

        cards = [
            ("Samples", "samples"),
            ("Feed Mass", "feed_mass"),
            ("Concentrate Mass", "concentrate_mass"),
            ("Recovery", "recovery"),
            ("Yield", "yield"),
            ("Balance", "balance"),
            ("Sieve Records", "sieve_records"),
            ("Avg Feed Grade", "avg_feed_grade"),
        ]

        for index, (title, key) in enumerate(cards):
            self._create_kpi_card(
                kpi_frame,
                index,
                title,
                self.kpi_vars[key],
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
        # RESULTS TABLE
        # --------------------------------------------------------------

        table_frame = ttk.LabelFrame(
            content,
            text="Integrated Results",
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
            "feed_mass",
            "feed_grade",
            "concentrate_mass",
            "concentrate_grade",
            "tailings_mass",
            "tailings_grade",
            "recovery",
            "yield",
            "grade_recovery",
            "enrichment",
            "balance",
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
            "feed_mass": "Feed Mass",
            "feed_grade": "Feed Grade",
            "concentrate_mass": "Concentrate Mass",
            "concentrate_grade": "Conc. Grade",
            "tailings_mass": "Tailings Mass",
            "tailings_grade": "Tailings Grade",
            "recovery": "Recovery (%)",
            "yield": "Yield (%)",
            "grade_recovery": "Grade Recovery (%)",
            "enrichment": "Enrichment",
            "balance": "Balance",
        }

        widths = {
            "sample_id": 110,
            "material": 105,
            "replicate": 70,
            "feed_mass": 90,
            "feed_grade": 90,
            "concentrate_mass": 120,
            "concentrate_grade": 105,
            "tailings_mass": 105,
            "tailings_grade": 105,
            "recovery": 95,
            "yield": 90,
            "grade_recovery": 120,
            "enrichment": 95,
            "balance": 95,
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

        details = ttk.LabelFrame(
            content,
            text="Selected Result",
            padding=12,
        )

        details.grid(
            row=0,
            column=1,
            sticky="nsew",
        )

        details.grid_columnconfigure(
            1,
            weight=1,
        )

        self.detail_vars = {
            "sample_id": tk.StringVar(value="N/A"),
            "material": tk.StringVar(value="N/A"),
            "replicate": tk.StringVar(value="N/A"),
            "date": tk.StringVar(value="N/A"),
            "operator": tk.StringVar(value="N/A"),
            "feed_mass": tk.StringVar(value="N/A"),
            "feed_grade": tk.StringVar(value="N/A"),
            "concentrate_mass": tk.StringVar(value="N/A"),
            "concentrate_grade": tk.StringVar(value="N/A"),
            "tailings_mass": tk.StringVar(value="N/A"),
            "tailings_grade": tk.StringVar(value="N/A"),
            "recovery": tk.StringVar(value="N/A"),
            "yield": tk.StringVar(value="N/A"),
            "grade_recovery": tk.StringVar(value="N/A"),
            "enrichment": tk.StringVar(value="N/A"),
            "balance": tk.StringVar(value="N/A"),
            "balance_error": tk.StringVar(value="N/A"),
        }

        detail_items = [
            ("Sample ID", "sample_id"),
            ("Material", "material"),
            ("Replicate", "replicate"),
            ("Experiment Date", "date"),
            ("Operator", "operator"),
            ("Feed Mass", "feed_mass"),
            ("Feed Grade", "feed_grade"),
            ("Concentrate Mass", "concentrate_mass"),
            ("Concentrate Grade", "concentrate_grade"),
            ("Tailings Mass", "tailings_mass"),
            ("Tailings Grade", "tailings_grade"),
            ("Mass Recovery", "recovery"),
            ("Yield", "yield"),
            ("Grade Recovery", "grade_recovery"),
            ("Enrichment Ratio", "enrichment"),
            ("Mass Balance", "balance"),
            ("Balance Error", "balance_error"),
        ]

        for row, (label, key) in enumerate(detail_items):

            ttk.Label(
                details,
                text=f"{label}:",
                font=("Segoe UI", 9, "bold"),
            ).grid(
                row=row,
                column=0,
                sticky="nw",
                padx=(0, 8),
                pady=4,
            )

            ttk.Label(
                details,
                textvariable=self.detail_vars[key],
                wraplength=220,
            ).grid(
                row=row,
                column=1,
                sticky="nw",
                pady=4,
            )

        # --------------------------------------------------------------
        # STATUS BAR
        # --------------------------------------------------------------

        status = ttk.Frame(
            self.frame
        )

        status.grid(
            row=4,
            column=0,
            sticky="ew",
            padx=20,
            pady=(0, 10),
        )

        ttk.Separator(
            status,
            orient="horizontal",
        ).pack(
            fill="x",
            pady=(0, 5),
        )

        ttk.Label(
            status,
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
        """Create a KPI card."""

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
            style="ResultsCardTitle.TLabel",
            anchor="center",
        ).pack(
            fill="x"
        )

        ttk.Label(
            card,
            textvariable=variable,
            style="ResultsCardValue.TLabel",
            anchor="center",
        ).pack(
            fill="x",
            pady=(4, 0),
        )

    # ==================================================================
    # PROCESS DATA
    # ==================================================================

    def _prepare_process_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Reduce repeated sieve rows to one process-level row
        per sample and replicate.

        This prevents feed/concentrate/tailings masses from being
        counted repeatedly simply because the same experiment has
        multiple sieve-size records.
        """

        if df is None or df.empty:
            return pd.DataFrame()

        working = df.copy()

        working.columns = [
            str(column).strip().lower()
            for column in working.columns
        ]

        if "sample_id" not in working.columns:
            working["sample_id"] = "Unknown"

        if "replicate" not in working.columns:
            working["replicate"] = "1"

        process_columns = [
            "feed_mass",
            "feed_grade",
            "concentrate_mass",
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
            "material_type",
            "experiment_date",
            "operator",
        ]

        existing = [
            column
            for column in process_columns
            if column in working.columns
        ]

        if not existing:
            return pd.DataFrame()

        working["_complete"] = (
            working[existing]
            .notna()
            .sum(axis=1)
        )

        working = working.sort_values(
            by=[
                "sample_id",
                "replicate",
                "_complete",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )

        working = (
            working
            .drop_duplicates(
                subset=[
                    "sample_id",
                    "replicate",
                ],
                keep="first",
            )
            .drop(
                columns=["_complete"],
                errors="ignore",
            )
            .reset_index(drop=True)
        )

        return self._calculate_metrics(
            working
        )

    # ==================================================================
    # CALCULATIONS
    # ==================================================================

    def _calculate_metrics(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate integrated beneficiation metrics."""

        df = dataframe.copy()

        numeric_columns = [
            "feed_mass",
            "feed_grade",
            "concentrate_mass",
            "concentrate_grade",
            "tailings_mass",
            "tailings_grade",
            "losses_mass",
        ]

        for column in numeric_columns:

            if column in df.columns:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

        df["mass_recovery"] = np.nan
        df["yield"] = np.nan
        df["grade_recovery"] = np.nan
        df["enrichment_ratio"] = np.nan
        df["mass_balance"] = np.nan
        df["balance_error_percent"] = np.nan

        # --------------------------------------------------------------
        # MASS RECOVERY
        # --------------------------------------------------------------

        if (
            "feed_mass" in df.columns
            and "concentrate_mass" in df.columns
        ):

            feed = df["feed_mass"]
            concentrate = df["concentrate_mass"]

            valid = (
                feed.notna()
                & concentrate.notna()
                & (feed > 0)
            )

            df.loc[
                valid,
                "mass_recovery"
            ] = (
                concentrate[valid]
                / feed[valid]
            ) * 100.0

            df.loc[
                valid,
                "yield"
            ] = (
                concentrate[valid]
                / feed[valid]
            ) * 100.0

        # --------------------------------------------------------------
        # GRADE RECOVERY
        # --------------------------------------------------------------

        if all(
            column in df.columns
            for column in [
                "feed_mass",
                "feed_grade",
                "concentrate_mass",
                "concentrate_grade",
            ]
        ):

            feed_mass = df["feed_mass"]
            feed_grade = df["feed_grade"]
            concentrate_mass = df["concentrate_mass"]
            concentrate_grade = df["concentrate_grade"]

            denominator = (
                feed_mass
                * feed_grade
            )

            valid = (
                denominator.notna()
                & (denominator > 0)
                & concentrate_mass.notna()
                & concentrate_grade.notna()
            )

            df.loc[
                valid,
                "grade_recovery"
            ] = (
                (
                    concentrate_mass[valid]
                    * concentrate_grade[valid]
                )
                / denominator[valid]
            ) * 100.0

        # --------------------------------------------------------------
        # ENRICHMENT
        # --------------------------------------------------------------

        if (
            "feed_grade" in df.columns
            and "concentrate_grade" in df.columns
        ):

            feed_grade = df["feed_grade"]
            concentrate_grade = df[
                "concentrate_grade"
            ]

            valid = (
                feed_grade.notna()
                & concentrate_grade.notna()
                & (feed_grade > 0)
            )

            df.loc[
                valid,
                "enrichment_ratio"
            ] = (
                concentrate_grade[valid]
                / feed_grade[valid]
            )

        # --------------------------------------------------------------
        # MATERIAL BALANCE
        # --------------------------------------------------------------

        if all(
            column in df.columns
            for column in [
                "feed_mass",
                "concentrate_mass",
                "tailings_mass",
            ]
        ):

            feed = df["feed_mass"]
            concentrate = df[
                "concentrate_mass"
            ]
            tailings = df[
                "tailings_mass"
            ]

            if "losses_mass" in df.columns:

                losses = (
                    df["losses_mass"]
                    .fillna(0)
                )

            else:
                losses = pd.Series(
                    0.0,
                    index=df.index,
                )

            total_output = (
                concentrate
                + tailings
                + losses
            )

            valid = (
                feed.notna()
                & concentrate.notna()
                & tailings.notna()
                & (feed > 0)
            )

            df.loc[
                valid,
                "mass_balance"
            ] = (
                feed[valid]
                - total_output[valid]
            )

            df.loc[
                valid,
                "balance_error_percent"
            ] = (
                (
                    feed[valid]
                    - total_output[valid]
                ).abs()
                / feed[valid]
            ) * 100.0

        return df

    # ==================================================================
    # FILTERS
    # ==================================================================

    def _update_filters(self):
        """Populate sample and replicate filters."""

        if self.process_df.empty:

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
            self.process_df["sample_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        sample_values = [
            "All Samples"
        ] + samples

        self.sample_combo["values"] = sample_values

        if (
            self.selected_sample.get()
            not in sample_values
        ):

            self.selected_sample.set(
                "All Samples"
            )

        self._update_replicates()

    def _update_replicates(self):
        """Update replicate choices based on selected sample."""

        if self.process_df.empty:

            self.replicate_combo["values"] = [
                "All Replicates"
            ]

            self.selected_replicate.set(
                "All Replicates"
            )

            return

        filtered = (
            self.process_df.copy()
        )

        sample = self.selected_sample.get()

        if (
            sample
            and sample != "All Samples"
        ):

            filtered = filtered[
                filtered["sample_id"].astype(str)
                == str(sample)
            ]

        replicates = sorted(
            filtered["replicate"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        values = [
            "All Replicates"
        ] + replicates

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
        """Respond to filter changes."""

        try:

            if (
                event is not None
                and event.widget == self.sample_combo
            ):
                self._update_replicates()

            self._update_kpis()
            self._update_result_table()
            self._clear_details()

        except Exception:
            logger.exception(
                "Failed to update results filters"
            )

    def _reset_filters(self):
        """Reset result filters."""

        self.selected_sample.set(
            "All Samples"
        )

        self.selected_replicate.set(
            "All Replicates"
        )

        self._update_replicates()
        self._update_kpis()
        self._update_result_table()
        self._clear_details()

        self.status_var.set(
            "Filters reset"
        )

    def _get_filtered_data(
        self,
    ) -> pd.DataFrame:
        """Return process data after filters."""

        if self.process_df.empty:
            return pd.DataFrame()

        filtered = (
            self.process_df.copy()
        )

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
    # KPIs
    # ==================================================================

    def _update_kpis(self):
        """Update summary KPI cards."""

        filtered = self._get_filtered_data()

        if filtered.empty:

            for variable in self.kpi_vars.values():
                variable.set("0")

            return

        self.kpi_vars["samples"].set(
            f"{filtered['sample_id'].nunique():,}"
        )

        self.kpi_vars["feed_mass"].set(
            self._average(
                filtered,
                "feed_mass",
            )
        )

        self.kpi_vars["concentrate_mass"].set(
            self._average(
                filtered,
                "concentrate_mass",
            )
        )

        self.kpi_vars["recovery"].set(
            self._average(
                filtered,
                "mass_recovery",
                "%",
            )
        )

        self.kpi_vars["yield"].set(
            self._average(
                filtered,
                "yield",
                "%",
            )
        )

        self.kpi_vars["balance"].set(
            self._average(
                filtered,
                "mass_balance",
            )
        )

        self.kpi_vars["avg_feed_grade"].set(
            self._average(
                filtered,
                "feed_grade",
            )
        )

        # Sieve records must come from the original dataframe.
        sieve_filtered = self._get_filtered_sieve_data()

        self.kpi_vars["sieve_records"].set(
            f"{len(sieve_filtered):,}"
        )

    def _get_filtered_sieve_data(
        self,
    ) -> pd.DataFrame:
        """Return original sieve rows matching current filters."""

        if self.df.empty:
            return pd.DataFrame()

        filtered = self.df.copy()

        if "sample_id" not in filtered.columns:
            return pd.DataFrame()

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
            and "replicate" in filtered.columns
        ):

            filtered = filtered[
                filtered["replicate"].astype(str)
                == str(replicate)
            ]

        if "sieve_size" in filtered.columns:

            filtered = filtered[
                filtered["sieve_size"].notna()
            ]

        return filtered

    @staticmethod
    def _average(
        dataframe,
        column,
        suffix="",
    ):
        """Safely calculate an average."""

        if column not in dataframe.columns:
            return "N/A"

        values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            return "N/A"

        value = values.mean()

        if not np.isfinite(value):
            return "N/A"

        return f"{value:,.3f}{suffix}"

    # ==================================================================
    # TABLE
    # ==================================================================

    def _update_result_table(self):
        """Populate the integrated results table."""

        for item in self.tree.get_children():
            self.tree.delete(item)

        filtered = self._get_filtered_data()

        if filtered.empty:
            return

        for index, row in filtered.iterrows():

            values = (
                self._display(row.get("sample_id")),
                self._display(row.get("material_type")),
                self._display(row.get("replicate")),
                self._number(row.get("feed_mass")),
                self._number(row.get("feed_grade")),
                self._number(
                    row.get("concentrate_mass")
                ),
                self._number(
                    row.get("concentrate_grade")
                ),
                self._number(
                    row.get("tailings_mass")
                ),
                self._number(
                    row.get("tailings_grade")
                ),
                self._number(
                    row.get("mass_recovery"),
                    "%",
                ),
                self._number(
                    row.get("yield"),
                    "%",
                ),
                self._number(
                    row.get("grade_recovery"),
                    "%",
                ),
                self._number(
                    row.get("enrichment_ratio")
                ),
                self._number(
                    row.get("mass_balance")
                ),
            )

            self.tree.insert(
                "",
                "end",
                iid=str(index),
                values=values,
            )

    # ==================================================================
    # DETAILS
    # ==================================================================

    def _on_row_selected(
        self,
        event=None,
    ):
        """Display details for selected result."""

        selection = self.tree.selection()

        if not selection:
            return

        try:

            index = int(
                selection[0]
            )

            filtered = (
                self._get_filtered_data()
            )

            if not (
                0 <= index < len(filtered)
            ):
                return

            row = filtered.iloc[index]

            self.detail_vars[
                "sample_id"
            ].set(
                self._display(
                    row.get("sample_id")
                )
            )

            self.detail_vars[
                "material"
            ].set(
                self._display(
                    row.get("material_type")
                )
            )

            self.detail_vars[
                "replicate"
            ].set(
                self._display(
                    row.get("replicate")
                )
            )

            self.detail_vars[
                "date"
            ].set(
                self._date(
                    row.get("experiment_date")
                )
            )

            self.detail_vars[
                "operator"
            ].set(
                self._display(
                    row.get("operator")
                )
            )

            self.detail_vars[
                "feed_mass"
            ].set(
                self._number(
                    row.get("feed_mass")
                )
            )

            self.detail_vars[
                "feed_grade"
            ].set(
                self._number(
                    row.get("feed_grade")
                )
            )

            self.detail_vars[
                "concentrate_mass"
            ].set(
                self._number(
                    row.get("concentrate_mass")
                )
            )

            self.detail_vars[
                "concentrate_grade"
            ].set(
                self._number(
                    row.get("concentrate_grade")
                )
            )

            self.detail_vars[
                "tailings_mass"
            ].set(
                self._number(
                    row.get("tailings_mass")
                )
            )

            self.detail_vars[
                "tailings_grade"
            ].set(
                self._number(
                    row.get("tailings_grade")
                )
            )

            self.detail_vars[
                "recovery"
            ].set(
                self._number(
                    row.get("mass_recovery"),
                    "%",
                )
            )

            self.detail_vars[
                "yield"
            ].set(
                self._number(
                    row.get("yield"),
                    "%",
                )
            )

            self.detail_vars[
                "grade_recovery"
            ].set(
                self._number(
                    row.get("grade_recovery"),
                    "%",
                )
            )

            self.detail_vars[
                "enrichment"
            ].set(
                self._number(
                    row.get("enrichment_ratio")
                )
            )

            self.detail_vars[
                "balance"
            ].set(
                self._number(
                    row.get("mass_balance")
                )
            )

            self.detail_vars[
                "balance_error"
            ].set(
                self._number(
                    row.get(
                        "balance_error_percent"
                    ),
                    "%",
                )
            )

        except Exception:
            logger.exception(
                "Failed to display selected result"
            )

    def _clear_details(self):
        """Clear the selected-result panel."""

        for variable in self.detail_vars.values():
            variable.set("N/A")

    # ==================================================================
    # FORMATTING
    # ==================================================================

    @staticmethod
    def _display(value):
        """Safely convert a value to display text."""

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
    def _number(
        value,
        suffix="",
    ):
        """Safely format a number."""

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
    def _date(value):
        """Safely format a date."""

        if value is None:
            return "N/A"

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

    def get_current_data(self):
        """Return currently filtered process-level results."""

        return (
            self._get_filtered_data()
            .copy()
        )

    def get_selected_record(
        self,
    ) -> Optional[pd.Series]:
        """Return the currently selected result."""

        selection = self.tree.selection()

        if not selection:
            return None

        try:

            index = int(
                selection[0]
            )

            filtered = (
                self._get_filtered_data()
            )

            if 0 <= index < len(filtered):
                return filtered.iloc[index]

        except Exception:
            logger.exception(
                "Failed to retrieve selected result"
            )

        return None


__all__ = ["ResultsView"]