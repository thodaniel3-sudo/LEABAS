"""
LEBAS - Beneficiation View
File: ui/beneficiation_view.py

Purpose:
    Displays and analyzes beneficiation-test data.

Important:
    - All laboratory values come from dataset.csv.
    - No experimental results are hard-coded.
    - Repeated sieve rows are reduced to one beneficiation record
      per sample + replicate.
    - Missing values are handled safely.
    - The view is compatible with the LEBAS main.py navigation system.

Main calculations:
    Mass Recovery (%) =
        Concentrate Mass / Feed Mass × 100

    Grade Recovery (%) =
        Concentrate Mass × Concentrate Grade
        ----------------------------------- × 100
              Feed Mass × Feed Grade

    Yield (%) =
        Concentrate Mass / Feed Mass × 100

    Enrichment Ratio =
        Concentrate Grade / Feed Grade
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


class BeneficiationView:
    """Beneficiation analysis interface."""

    def __init__(self, parent, controller):
        self.controller = controller
        self.frame = ttk.Frame(parent)

        self.df = pd.DataFrame()
        self.beneficiation_df = pd.DataFrame()

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
            "feed_mass": tk.StringVar(value="0"),
            "concentrate_mass": tk.StringVar(value="0"),
            "tailings_mass": tk.StringVar(value="0"),
            "recovery": tk.StringVar(value="0"),
            "yield": tk.StringVar(value="0"),
            "feed_grade": tk.StringVar(value="0"),
            "concentrate_grade": tk.StringVar(value="0"),
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
            "method": tk.StringVar(value="N/A"),
            "time": tk.StringVar(value="N/A"),
            "reagent": tk.StringVar(value="N/A"),
            "dosage": tk.StringVar(value="N/A"),
            "ph": tk.StringVar(value="N/A"),
            "pulp_density": tk.StringVar(value="N/A"),
            "temperature": tk.StringVar(value="N/A"),
            "feed_mass": tk.StringVar(value="N/A"),
            "feed_grade": tk.StringVar(value="N/A"),
            "concentrate_mass": tk.StringVar(value="N/A"),
            "concentrate_grade": tk.StringVar(value="N/A"),
            "tailings_mass": tk.StringVar(value="N/A"),
            "tailings_grade": tk.StringVar(value="N/A"),
            "losses_mass": tk.StringVar(value="N/A"),
            "recovery": tk.StringVar(value="N/A"),
            "yield": tk.StringVar(value="N/A"),
            "enrichment": tk.StringVar(value="N/A"),
            "mass_balance": tk.StringVar(value="N/A"),
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
        """Reload the dataset and refresh the beneficiation interface."""

        try:
            self.df = load_dataset()

            if self.df is None:
                self.df = pd.DataFrame()

            self.beneficiation_df = (
                self._prepare_beneficiation_dataframe(
                    self.df
                )
            )

            self._update_filter_values()
            self._update_kpis()
            self._update_table()
            self._clear_details()

            if self.beneficiation_df.empty:
                self.status_var.set(
                    "No beneficiation data available"
                )
            else:
                self.status_var.set(
                    f"{len(self.beneficiation_df):,} beneficiation "
                    "record(s) loaded"
                )

        except Exception as exc:
            logger.exception(
                "Failed to refresh beneficiation view"
            )

            self.status_var.set(
                "Error loading beneficiation data"
            )

            try:
                messagebox.showerror(
                    "Beneficiation Data Error",
                    f"Unable to load beneficiation data.\n\n{exc}",
                )
            except Exception:
                pass

    # ==================================================================
    # STYLE
    # ==================================================================

    def _configure_style(self):
        """Configure ttk styles."""

        style = ttk.Style()

        try:
            style.configure(
                "BeneficiationTitle.TLabel",
                font=("Segoe UI", 20, "bold"),
            )

            style.configure(
                "BeneficiationSubtitle.TLabel",
                font=("Segoe UI", 10),
            )

            style.configure(
                "BeneficiationCardValue.TLabel",
                font=("Segoe UI", 17, "bold"),
            )

            style.configure(
                "BeneficiationCardTitle.TLabel",
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
                "Could not configure beneficiation styles"
            )

    # ==================================================================
    # USER INTERFACE
    # ==================================================================

    def _build_ui(self):
        """Build the complete beneficiation interface."""

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
            text="Beneficiation Analysis",
            style="BeneficiationTitle.TLabel",
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
            "Beneficiation performance, grades, recovery "
            "and concentrate production"
        )

        if version:
            subtitle += f"  •  {version}"

        ttk.Label(
            header,
            text=subtitle,
            style="BeneficiationSubtitle.TLabel",
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
                uniform="beneficiation_kpi",
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
            "Feed Mass",
            self.kpi_vars["feed_mass"],
        )

        self._create_kpi_card(
            kpi_container,
            2,
            "Concentrate Mass",
            self.kpi_vars["concentrate_mass"],
        )

        self._create_kpi_card(
            kpi_container,
            3,
            "Tailings Mass",
            self.kpi_vars["tailings_mass"],
        )

        self._create_kpi_card(
            kpi_container,
            4,
            "Recovery",
            self.kpi_vars["recovery"],
        )

        self._create_kpi_card(
            kpi_container,
            5,
            "Yield",
            self.kpi_vars["yield"],
        )

        self._create_kpi_card(
            kpi_container,
            6,
            "Feed Grade",
            self.kpi_vars["feed_grade"],
        )

        self._create_kpi_card(
            kpi_container,
            7,
            "Concentrate Grade",
            self.kpi_vars["concentrate_grade"],
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
            text="Beneficiation Results",
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
            "method",
            "feed_mass",
            "feed_grade",
            "concentrate_mass",
            "concentrate_grade",
            "tailings_mass",
            "tailings_grade",
            "recovery",
            "yield",
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
            "method": "Method",
            "feed_mass": "Feed Mass",
            "feed_grade": "Feed Grade",
            "concentrate_mass": "Concentrate Mass",
            "concentrate_grade": "Concentrate Grade",
            "tailings_mass": "Tailings Mass",
            "tailings_grade": "Tailings Grade",
            "recovery": "Recovery (%)",
            "yield": "Yield (%)",
            "enrichment": "Enrichment",
            "balance": "Balance",
        }

        widths = {
            "sample_id": 115,
            "material": 110,
            "replicate": 75,
            "method": 115,
            "feed_mass": 95,
            "feed_grade": 95,
            "concentrate_mass": 125,
            "concentrate_grade": 130,
            "tailings_mass": 105,
            "tailings_grade": 105,
            "recovery": 100,
            "yield": 90,
            "enrichment": 100,
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
        # DETAILS PANEL
        # --------------------------------------------------------------

        details_frame = ttk.LabelFrame(
            content,
            text="Beneficiation Details",
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
            ("Method", "method"),
            ("Beneficiation Time", "time"),
            ("Reagent", "reagent"),
            ("Reagent Dosage", "dosage"),
            ("pH", "ph"),
            ("Pulp Density", "pulp_density"),
            ("Temperature", "temperature"),
            ("Feed Mass", "feed_mass"),
            ("Feed Grade", "feed_grade"),
            ("Concentrate Mass", "concentrate_mass"),
            ("Concentrate Grade", "concentrate_grade"),
            ("Tailings Mass", "tailings_mass"),
            ("Tailings Grade", "tailings_grade"),
            ("Losses Mass", "losses_mass"),
            ("Recovery", "recovery"),
            ("Yield", "yield"),
            ("Enrichment Ratio", "enrichment"),
            ("Mass Balance", "mass_balance"),
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
                pady=4,
            )

            ttk.Label(
                details_frame,
                textvariable=self.detail_vars[key],
                wraplength=220,
            ).grid(
                row=row,
                column=1,
                sticky="nw",
                pady=4,
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
            style="BeneficiationCardTitle.TLabel",
            anchor="center",
        ).pack(
            fill="x"
        )

        ttk.Label(
            card,
            textvariable=variable,
            style="BeneficiationCardValue.TLabel",
            anchor="center",
        ).pack(
            fill="x",
            pady=(4, 0),
        )

    # ==================================================================
    # DATA PREPARATION
    # ==================================================================

    def _prepare_beneficiation_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prepare one beneficiation record per sample + replicate.

        The CSV may contain several rows per sample because each sieve
        fraction occupies a separate row.

        Beneficiation data must not be counted repeatedly.

        Therefore, this method selects the most complete process-level
        row for each sample + replicate.
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

        # --------------------------------------------------------------
        # REQUIRED IDENTIFIERS
        # --------------------------------------------------------------

        if "sample_id" not in working.columns:
            working["sample_id"] = "Unknown"

        if "replicate" not in working.columns:
            working["replicate"] = "1"

        # --------------------------------------------------------------
        # CHECK FOR BENEFICIATION DATA
        # --------------------------------------------------------------

        beneficiation_columns = [
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
        ]

        existing_columns = [
            column
            for column in beneficiation_columns
            if column in working.columns
        ]

        if not existing_columns:
            logger.warning(
                "No beneficiation columns were found."
            )
            return pd.DataFrame()

        # --------------------------------------------------------------
        # NUMERIC CONVERSION
        # --------------------------------------------------------------

        numeric_columns = [
            "feed_mass",
            "feed_grade",
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
        ]

        for column in numeric_columns:

            if column in working.columns:

                working[column] = pd.to_numeric(
                    working[column],
                    errors="coerce",
                )

        # --------------------------------------------------------------
        # KEEP ONLY ROWS CONTAINING BENEFICIATION INFORMATION
        # --------------------------------------------------------------

        has_beneficiation_data = (
            working[existing_columns]
            .notna()
            .any(axis=1)
        )

        working = working.loc[
            has_beneficiation_data
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
        # COMPLETENESS SCORE
        # --------------------------------------------------------------

        process_columns = [
            "feed_mass",
            "feed_grade",
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
            "material_type",
            "experiment_date",
            "operator",
        ]

        existing_process_columns = [
            column
            for column in process_columns
            if column in working.columns
        ]

        if existing_process_columns:

            working["_completeness"] = (
                working[
                    existing_process_columns
                ]
                .notna()
                .sum(axis=1)
            )

        else:

            working["_completeness"] = 0

        # --------------------------------------------------------------
        # SELECT MOST COMPLETE PROCESS ROW
        # --------------------------------------------------------------

        working = working.sort_values(
            by=[
                "sample_id",
                "replicate",
                "_completeness",
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
            .copy()
        )

        working.drop(
            columns=["_completeness"],
            inplace=True,
            errors="ignore",
        )

        # --------------------------------------------------------------
        # CALCULATE BENEFICIATION METRICS
        # --------------------------------------------------------------

        working["mass_recovery"] = np.nan
        working["yield"] = np.nan
        working["grade_recovery"] = np.nan
        working["enrichment_ratio"] = np.nan
        working["mass_balance"] = np.nan
        working["balance_error_percent"] = np.nan

        # --------------------------------------------------------------
        # MASS RECOVERY / YIELD
        # --------------------------------------------------------------

        if (
            "feed_mass" in working.columns
            and "concentrate_mass" in working.columns
        ):

            feed_mass = pd.to_numeric(
                working["feed_mass"],
                errors="coerce",
            )

            concentrate_mass = pd.to_numeric(
                working["concentrate_mass"],
                errors="coerce",
            )

            valid = (
                feed_mass.notna()
                & concentrate_mass.notna()
                & (feed_mass > 0)
            )

            working.loc[
                valid,
                "mass_recovery"
            ] = (
                concentrate_mass[valid]
                / feed_mass[valid]
            ) * 100.0

            working.loc[
                valid,
                "yield"
            ] = (
                concentrate_mass[valid]
                / feed_mass[valid]
            ) * 100.0

        # --------------------------------------------------------------
        # GRADE RECOVERY
        # --------------------------------------------------------------

        if all(
            column in working.columns
            for column in [
                "feed_mass",
                "feed_grade",
                "concentrate_mass",
                "concentrate_grade",
            ]
        ):

            feed_mass = pd.to_numeric(
                working["feed_mass"],
                errors="coerce",
            )

            feed_grade = pd.to_numeric(
                working["feed_grade"],
                errors="coerce",
            )

            concentrate_mass = pd.to_numeric(
                working["concentrate_mass"],
                errors="coerce",
            )

            concentrate_grade = pd.to_numeric(
                working["concentrate_grade"],
                errors="coerce",
            )

            denominator = (
                feed_mass * feed_grade
            )

            valid = (
                denominator.notna()
                & (denominator > 0)
                & concentrate_mass.notna()
                & concentrate_grade.notna()
            )

            working.loc[
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
        # ENRICHMENT RATIO
        # --------------------------------------------------------------

        if (
            "feed_grade" in working.columns
            and "concentrate_grade" in working.columns
        ):

            feed_grade = pd.to_numeric(
                working["feed_grade"],
                errors="coerce",
            )

            concentrate_grade = pd.to_numeric(
                working["concentrate_grade"],
                errors="coerce",
            )

            valid = (
                feed_grade.notna()
                & concentrate_grade.notna()
                & (feed_grade > 0)
            )

            working.loc[
                valid,
                "enrichment_ratio"
            ] = (
                concentrate_grade[valid]
                / feed_grade[valid]
            )

        # --------------------------------------------------------------
        # MASS BALANCE
        # --------------------------------------------------------------

        if (
            "feed_mass" in working.columns
            and "concentrate_mass" in working.columns
            and "tailings_mass" in working.columns
        ):

            feed_mass = pd.to_numeric(
                working["feed_mass"],
                errors="coerce",
            )

            concentrate_mass = pd.to_numeric(
                working["concentrate_mass"],
                errors="coerce",
            )

            tailings_mass = pd.to_numeric(
                working["tailings_mass"],
                errors="coerce",
            )

            losses_mass = pd.Series(
                0.0,
                index=working.index,
            )

            if "losses_mass" in working.columns:

                losses_mass = pd.to_numeric(
                    working["losses_mass"],
                    errors="coerce",
                ).fillna(0)

            output_mass = (
                concentrate_mass
                + tailings_mass
                + losses_mass
            )

            valid = (
                feed_mass.notna()
                & concentrate_mass.notna()
                & tailings_mass.notna()
                & (feed_mass > 0)
            )

            working.loc[
                valid,
                "mass_balance"
            ] = (
                feed_mass[valid]
                - output_mass[valid]
            )

            working.loc[
                valid,
                "balance_error_percent"
            ] = (
                (
                    feed_mass[valid]
                    - output_mass[valid]
                ).abs()
                / feed_mass[valid]
            ) * 100.0

        return working.reset_index(
            drop=True
        )

    # ==================================================================
    # FILTERS
    # ==================================================================

    def _update_filter_values(self):
        """Update sample and replicate filters."""

        if self.beneficiation_df.empty:

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
            self.beneficiation_df["sample_id"]
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
        """Update replicate choices according to selected sample."""

        if self.beneficiation_df.empty:

            self.replicate_combo["values"] = [
                "All Replicates"
            ]

            self.selected_replicate.set(
                "All Replicates"
            )

            return

        filtered = (
            self.beneficiation_df.copy()
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

            values = [
                "All Replicates"
            ] + values

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
                "Failed to apply beneficiation filter"
            )

    def _reset_filters(self):
        """Reset all filters."""

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

        if self.beneficiation_df.empty:

            self.status_var.set(
                "No beneficiation data available"
            )

        else:

            self.status_var.set(
                f"{len(self.beneficiation_df):,} beneficiation "
                "record(s) loaded"
            )

    def _get_filtered_dataframe(
        self,
    ) -> pd.DataFrame:
        """Return data according to current filters."""

        if self.beneficiation_df.empty:
            return pd.DataFrame()

        filtered = (
            self.beneficiation_df.copy()
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
    # KPI CALCULATIONS
    # ==================================================================

    def _update_kpis(self):
        """Calculate and display beneficiation KPIs."""

        filtered = self._get_filtered_dataframe()

        if filtered.empty:

            for variable in self.kpi_vars.values():
                variable.set("0")

            return

        # --------------------------------------------------------------
        # SAMPLE COUNT
        # --------------------------------------------------------------

        self.kpi_vars["samples"].set(
            f"{filtered['sample_id'].nunique():,}"
        )

        # --------------------------------------------------------------
        # AVERAGE VALUES
        # --------------------------------------------------------------

        self.kpi_vars["feed_mass"].set(
            self._format_average(
                filtered,
                "feed_mass",
            )
        )

        self.kpi_vars["concentrate_mass"].set(
            self._format_average(
                filtered,
                "concentrate_mass",
            )
        )

        self.kpi_vars["tailings_mass"].set(
            self._format_average(
                filtered,
                "tailings_mass",
            )
        )

        self.kpi_vars["recovery"].set(
            self._format_average(
                filtered,
                "mass_recovery",
                suffix="%",
            )
        )

        self.kpi_vars["yield"].set(
            self._format_average(
                filtered,
                "yield",
                suffix="%",
            )
        )

        self.kpi_vars["feed_grade"].set(
            self._format_average(
                filtered,
                "feed_grade",
            )
        )

        self.kpi_vars["concentrate_grade"].set(
            self._format_average(
                filtered,
                "concentrate_grade",
            )
        )

    def _format_average(
        self,
        dataframe,
        column,
        suffix="",
    ):
        """Safely calculate a column average."""

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

    def _update_table(self):
        """Refresh the beneficiation table."""

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

                self._display_value(
                    row.get("beneficiation_method")
                ),

                self._format_number(
                    row.get("feed_mass")
                ),

                self._format_number(
                    row.get("feed_grade")
                ),

                self._format_number(
                    row.get("concentrate_mass")
                ),

                self._format_number(
                    row.get("concentrate_grade")
                ),

                self._format_number(
                    row.get("tailings_mass")
                ),

                self._format_number(
                    row.get("tailings_grade")
                ),

                self._format_number(
                    row.get("mass_recovery"),
                    suffix="%",
                ),

                self._format_number(
                    row.get("yield"),
                    suffix="%",
                ),

                self._format_number(
                    row.get("enrichment_ratio")
                ),

                self._format_number(
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
    # SELECTED RECORD
    # ==================================================================

    def _on_row_selected(
        self,
        event=None,
    ):
        """Display the selected beneficiation record."""

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
                "method"
            ].set(
                self._display_value(
                    row.get("beneficiation_method")
                )
            )

            self.detail_vars[
                "time"
            ].set(
                self._format_number(
                    row.get("beneficiation_time")
                )
            )

            self.detail_vars[
                "reagent"
            ].set(
                self._display_value(
                    row.get("reagent_type")
                )
            )

            self.detail_vars[
                "dosage"
            ].set(
                self._format_number(
                    row.get("reagent_dosage")
                )
            )

            self.detail_vars[
                "ph"
            ].set(
                self._format_number(
                    row.get("ph_level")
                )
            )

            self.detail_vars[
                "pulp_density"
            ].set(
                self._format_number(
                    row.get("pulp_density")
                )
            )

            self.detail_vars[
                "temperature"
            ].set(
                self._format_number(
                    row.get("temperature")
                )
            )

            self.detail_vars[
                "feed_mass"
            ].set(
                self._format_number(
                    row.get("feed_mass")
                )
            )

            self.detail_vars[
                "feed_grade"
            ].set(
                self._format_number(
                    row.get("feed_grade")
                )
            )

            self.detail_vars[
                "concentrate_mass"
            ].set(
                self._format_number(
                    row.get("concentrate_mass")
                )
            )

            self.detail_vars[
                "concentrate_grade"
            ].set(
                self._format_number(
                    row.get("concentrate_grade")
                )
            )

            self.detail_vars[
                "tailings_mass"
            ].set(
                self._format_number(
                    row.get("tailings_mass")
                )
            )

            self.detail_vars[
                "tailings_grade"
            ].set(
                self._format_number(
                    row.get("tailings_grade")
                )
            )

            self.detail_vars[
                "losses_mass"
            ].set(
                self._format_number(
                    row.get("losses_mass")
                )
            )

            self.detail_vars[
                "recovery"
            ].set(
                self._format_number(
                    row.get("mass_recovery"),
                    suffix="%",
                )
            )

            self.detail_vars[
                "yield"
            ].set(
                self._format_number(
                    row.get("yield"),
                    suffix="%",
                )
            )

            self.detail_vars[
                "enrichment"
            ].set(
                self._format_number(
                    row.get("enrichment_ratio")
                )
            )

            self.detail_vars[
                "mass_balance"
            ].set(
                self._format_number(
                    row.get("mass_balance")
                )
            )

        except Exception:
            logger.exception(
                "Failed to display selected beneficiation record"
            )

    def _clear_details(self):
        """Clear the details panel."""

        for variable in self.detail_vars.values():
            variable.set("N/A")

    # ==================================================================
    # FORMATTING HELPERS
    # ==================================================================

    @staticmethod
    def _display_value(
        value,
    ) -> str:
        """Convert a value safely to display text."""

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
        """Safely format numeric data."""

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
        """Safely format experiment date."""

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
        """Return currently filtered beneficiation data."""

        return (
            self._get_filtered_dataframe()
            .copy()
        )

    def get_selected_record(
        self,
    ) -> Optional[pd.Series]:
        """Return the selected beneficiation record."""

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
                "Failed to retrieve selected beneficiation record"
            )

        return None


__all__ = ["BeneficiationView"]