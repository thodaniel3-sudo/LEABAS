"""
LEBAS Crushing Analysis View

Laboratory Extraction and Beneficiation Analysis System

This module provides the Crushing Analysis interface.

The view is completely dataset-driven. It reads the current dataset through
the existing LEBAS data-loader and calculates crushing performance from the
available laboratory data.

Expected dataset columns include:

    sample_id
    material_type
    experiment_date
    operator
    replicate
    feed_mass
    feed_grade
    crusher_type
    crush_time
    crush_power
    crush_feed_size
    crush_product_size

The class is compatible with the main LEBAS controller and provides:

    CrushingView(parent, controller)
    get_frame()
    refresh()
"""

from __future__ import annotations

import logging
import math
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any, Optional

import numpy as np
import pandas as pd

from services.data_loader import load_dataset


logger = logging.getLogger(__name__)


class CrushingView:
    """
    Crushing Analysis page for LEBAS.
    """

    def __init__(self, parent, controller):
        """
        Initialize the Crushing Analysis view.

        Parameters
        ----------
        parent:
            Parent Tkinter widget.

        controller:
            Main LEBAS application controller.
        """

        self.parent = parent
        self.controller = controller

        self.df: Optional[pd.DataFrame] = None
        self.crushing_df: Optional[pd.DataFrame] = None

        self.selected_sample = tk.StringVar(value="All Samples")
        self.selected_replicate = tk.StringVar(value="All Replicates")

        self.kpi_vars = {
            "samples": tk.StringVar(value="0"),
            "feed_mass": tk.StringVar(value="N/A"),
            "crush_time": tk.StringVar(value="N/A"),
            "crush_power": tk.StringVar(value="N/A"),
            "feed_size": tk.StringVar(value="N/A"),
            "product_size": tk.StringVar(value="N/A"),
            "reduction_ratio": tk.StringVar(value="N/A"),
            "size_reduction": tk.StringVar(value="N/A"),
        }

        self.status_var = tk.StringVar(
            value="Ready"
        )

        # Main page frame
        self.frame = ttk.Frame(parent)

        self.frame.grid_rowconfigure(1, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

        self._build_ui()

        # Initial dataset load
        self.refresh()

    # ======================================================================
    # PUBLIC API
    # ======================================================================

    def get_frame(self):
        """
        Return the main view frame.

        Required by main.py.
        """
        return self.frame

    def refresh(self):
        """
        Reload the current dataset and refresh all crushing information.

        This method is called by main.py whenever the Crushing page is
        displayed.
        """

        try:
            self._set_status("Loading crushing data...")

            self.df = load_dataset()

            if self.df is None:
                self.df = pd.DataFrame()

            if not isinstance(self.df, pd.DataFrame):
                self.df = pd.DataFrame()

            self.df = self._prepare_dataframe(self.df)

            self._update_sample_selectors()
            self._update_crushing_data()
            self._update_kpis()
            self._update_table()

            self._set_status(
                f"Crushing analysis updated successfully "
                f"from the current dataset."
            )

        except Exception as exc:
            logger.exception(
                "Error refreshing CrushingView: %s",
                exc,
            )

            self._set_status(
                f"Error loading crushing data: {exc}"
            )

            self._clear_table()

    # ======================================================================
    # USER INTERFACE
    # ======================================================================

    def _build_ui(self):
        """
        Build the complete Crushing Analysis interface.
        """

        self._create_header()
        self._create_controls()
        self._create_kpis()
        self._create_results_section()
        self._create_status_bar()

    def _create_header(self):
        """
        Create page header.
        """

        header = ttk.Frame(self.frame)
        header.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=20,
            pady=(15, 5),
        )

        header.grid_columnconfigure(0, weight=1)

        title = ttk.Label(
            header,
            text="Crushing Analysis",
            font=("Helvetica", 20, "bold"),
        )
        title.grid(
            row=0,
            column=0,
            sticky="w",
        )

        subtitle = ttk.Label(
            header,
            text=(
                "Analysis of feed size reduction, crushing time, "
                "crusher power and crushing performance."
            ),
            font=("Helvetica", 10),
        )
        subtitle.grid(
            row=1,
            column=0,
            sticky="w",
            pady=(3, 0),
        )

        separator = ttk.Separator(
            self.frame,
            orient="horizontal",
        )
        separator.grid(
            row=0,
            column=0,
            sticky="se",
            padx=20,
            pady=(65, 0),
        )

    def _create_controls(self):
        """
        Create sample and replicate selection controls.
        """

        controls = ttk.LabelFrame(
            self.frame,
            text="Analysis Controls",
            padding=10,
        )

        controls.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=20,
            pady=(10, 8),
        )

        controls.grid_columnconfigure(5, weight=1)

        # Sample
        ttk.Label(
            controls,
            text="Sample:",
            font=("Helvetica", 9, "bold"),
        ).grid(
            row=0,
            column=0,
            padx=(0, 5),
            pady=5,
            sticky="w",
        )

        self.sample_combo = ttk.Combobox(
            controls,
            textvariable=self.selected_sample,
            state="readonly",
            width=22,
        )

        self.sample_combo.grid(
            row=0,
            column=1,
            padx=(0, 15),
            pady=5,
            sticky="w",
        )

        self.sample_combo.bind(
            "<<ComboboxSelected>>",
            self._on_sample_changed,
        )

        # Replicate
        ttk.Label(
            controls,
            text="Replicate:",
            font=("Helvetica", 9, "bold"),
        ).grid(
            row=0,
            column=2,
            padx=(0, 5),
            pady=5,
            sticky="w",
        )

        self.replicate_combo = ttk.Combobox(
            controls,
            textvariable=self.selected_replicate,
            state="readonly",
            width=18,
        )

        self.replicate_combo.grid(
            row=0,
            column=3,
            padx=(0, 15),
            pady=5,
            sticky="w",
        )

        self.replicate_combo.bind(
            "<<ComboboxSelected>>",
            self._on_replicate_changed,
        )

        # Refresh button
        refresh_button = ttk.Button(
            controls,
            text="↻ Refresh Data",
            command=self.refresh,
        )

        refresh_button.grid(
            row=0,
            column=4,
            padx=5,
            pady=5,
        )

        # Reset selection
        reset_button = ttk.Button(
            controls,
            text="Reset Filters",
            command=self._reset_filters,
        )

        reset_button.grid(
            row=0,
            column=5,
            padx=(5, 0),
            pady=5,
            sticky="e",
        )

    def _create_kpis(self):
        """
        Create crushing KPI cards.
        """

        kpi_frame = ttk.LabelFrame(
            self.frame,
            text="Crushing Performance Indicators",
            padding=10,
        )

        kpi_frame.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=20,
            pady=(0, 8),
        )

        for column in range(8):
            kpi_frame.grid_columnconfigure(
                column,
                weight=1,
            )

        cards = [
            (
                "Samples",
                self.kpi_vars["samples"],
            ),
            (
                "Feed Mass",
                self.kpi_vars["feed_mass"],
            ),
            (
                "Avg. Crush Time",
                self.kpi_vars["crush_time"],
            ),
            (
                "Avg. Power",
                self.kpi_vars["crush_power"],
            ),
            (
                "Avg. Feed Size",
                self.kpi_vars["feed_size"],
            ),
            (
                "Avg. Product Size",
                self.kpi_vars["product_size"],
            ),
            (
                "Reduction Ratio",
                self.kpi_vars["reduction_ratio"],
            ),
            (
                "Size Reduction",
                self.kpi_vars["size_reduction"],
            ),
        ]

        for index, (label, variable) in enumerate(cards):

            card = ttk.Frame(
                kpi_frame,
                relief="ridge",
                borderwidth=1,
            )

            card.grid(
                row=0,
                column=index,
                padx=3,
                pady=3,
                sticky="nsew",
            )

            card.grid_columnconfigure(
                0,
                weight=1,
            )

            value = ttk.Label(
                card,
                textvariable=variable,
                font=("Helvetica", 13, "bold"),
            )

            value.grid(
                row=0,
                column=0,
                padx=6,
                pady=(9, 2),
            )

            label_widget = ttk.Label(
                card,
                text=label,
                font=("Helvetica", 8),
            )

            label_widget.grid(
                row=1,
                column=0,
                padx=6,
                pady=(0, 9),
            )

    def _create_results_section(self):
        """
        Create the main crushing-results table and analysis information.
        """

        results_frame = ttk.LabelFrame(
            self.frame,
            text="Crushing Results",
            padding=8,
        )

        results_frame.grid(
            row=3,
            column=0,
            sticky="nsew",
            padx=20,
            pady=(0, 8),
        )

        self.frame.grid_rowconfigure(
            3,
            weight=1,
        )

        results_frame.grid_rowconfigure(
            0,
            weight=1,
        )

        results_frame.grid_columnconfigure(
            0,
            weight=1,
        )

        # Table container
        table_container = ttk.Frame(
            results_frame
        )

        table_container.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        table_container.grid_rowconfigure(
            0,
            weight=1,
        )

        table_container.grid_columnconfigure(
            0,
            weight=1,
        )

        columns = [
            "sample_id",
            "material",
            "replicate",
            "crusher",
            "feed_mass",
            "feed_size",
            "product_size",
            "crush_time",
            "crush_power",
            "reduction_ratio",
            "size_reduction",
        ]

        self.tree = ttk.Treeview(
            table_container,
            columns=columns,
            show="headings",
            selectmode="browse",
        )

        headings = {
            "sample_id": "Sample ID",
            "material": "Material",
            "replicate": "Replicate",
            "crusher": "Crusher Type",
            "feed_mass": "Feed Mass (g)",
            "feed_size": "Feed Size",
            "product_size": "Product Size",
            "crush_time": "Crush Time",
            "crush_power": "Power",
            "reduction_ratio": "Reduction Ratio",
            "size_reduction": "Size Reduction (%)",
        }

        widths = {
            "sample_id": 115,
            "material": 120,
            "replicate": 75,
            "crusher": 120,
            "feed_mass": 100,
            "feed_size": 105,
            "product_size": 110,
            "crush_time": 100,
            "crush_power": 95,
            "reduction_ratio": 110,
            "size_reduction": 125,
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
                stretch=False,
            )

        self.tree.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        vertical_scrollbar = ttk.Scrollbar(
            table_container,
            orient="vertical",
            command=self.tree.yview,
        )

        vertical_scrollbar.grid(
            row=0,
            column=1,
            sticky="ns",
        )

        horizontal_scrollbar = ttk.Scrollbar(
            table_container,
            orient="horizontal",
            command=self.tree.xview,
        )

        horizontal_scrollbar.grid(
            row=1,
            column=0,
            sticky="ew",
        )

        self.tree.configure(
            yscrollcommand=vertical_scrollbar.set,
            xscrollcommand=horizontal_scrollbar.set,
        )

        self.tree.bind(
            "<<TreeviewSelect>>",
            self._on_table_selection,
        )

        # Details panel
        details_frame = ttk.LabelFrame(
            results_frame,
            text="Selected Crushing Record",
            padding=8,
        )

        details_frame.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )

        details_frame.grid_columnconfigure(
            1,
            weight=1,
        )

        self.details_text = tk.Text(
            details_frame,
            height=5,
            wrap=tk.WORD,
            font=("Helvetica", 9),
            relief="sunken",
            bd=1,
            padx=6,
            pady=5,
        )

        self.details_text.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="ew",
        )

        self.details_text.insert(
            "1.0",
            "Select a crushing record from the table to view details.",
        )

        self.details_text.configure(
            state="disabled"
        )

    def _create_status_bar(self):
        """
        Create bottom status bar.
        """

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

        status_frame.grid_columnconfigure(
            0,
            weight=1,
        )

        status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            relief="sunken",
            padding=5,
            font=("Helvetica", 9),
        )

        status_label.grid(
            row=0,
            column=0,
            sticky="ew",
        )

    # ======================================================================
    # DATA PREPARATION
    # ======================================================================

    def _prepare_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prepare a clean copy of the dataset for crushing analysis.

        Process-level values can be repeated because the dataset also
        contains sieve rows. We therefore remove exact duplicate process
        records before displaying the crushing table.
        """

        if df is None or df.empty:
            return pd.DataFrame()

        working = df.copy()

        # Normalize column names.
        working.columns = [
            str(column).strip()
            for column in working.columns
        ]

        numeric_columns = [
            "feed_mass",
            "feed_grade",
            "crush_time",
            "crush_power",
            "crush_feed_size",
            "crush_product_size",
        ]

        for column in numeric_columns:
            if column in working.columns:
                working[column] = pd.to_numeric(
                    working[column],
                    errors="coerce",
                )

        # Create the crushing subset based on available columns.
        crushing_columns = [
            "sample_id",
            "material_type",
            "experiment_date",
            "operator",
            "replicate",
            "feed_mass",
            "feed_grade",
            "crusher_type",
            "crush_time",
            "crush_power",
            "crush_feed_size",
            "crush_product_size",
        ]

        available_columns = [
            column
            for column in crushing_columns
            if column in working.columns
        ]

        if not available_columns:
            return pd.DataFrame()

        crushing = working[
            available_columns
        ].copy()

        # --------------------------------------------------------------
        # Prevent repeated sieve rows from creating duplicate crushing
        # records.
        # --------------------------------------------------------------

        identity_columns = []

        if "sample_id" in crushing.columns:
            identity_columns.append("sample_id")

        if "replicate" in crushing.columns:
            identity_columns.append("replicate")

        if identity_columns:
            # Sort so that records containing the greatest amount of
            # information are preferred.
            process_columns = [
                column
                for column in [
                    "feed_mass",
                    "crusher_type",
                    "crush_time",
                    "crush_power",
                    "crush_feed_size",
                    "crush_product_size",
                ]
                if column in crushing.columns
            ]

            if process_columns:
                crushing["_valid_count"] = (
                    crushing[process_columns]
                    .notna()
                    .sum(axis=1)
                )

                crushing = (
                    crushing
                    .sort_values(
                        identity_columns + ["_valid_count"],
                        ascending=[True] * len(identity_columns) + [False],
                        kind="stable",
                    )
                    .drop_duplicates(
                        subset=identity_columns,
                        keep="first",
                    )
                    .drop(
                        columns=["_valid_count"],
                        errors="ignore",
                    )
                )
            else:
                crushing = crushing.drop_duplicates(
                    subset=identity_columns,
                    keep="first",
                )
        else:
            crushing = crushing.drop_duplicates()

        return crushing.reset_index(
            drop=True
        )

    # ======================================================================
    # FILTERING
    # ======================================================================

    def _update_sample_selectors(self):
        """
        Populate sample and replicate selectors from the dataset.
        """

        if (
            self.df is None
            or self.df.empty
        ):
            self.sample_combo["values"] = [
                "No Samples"
            ]

            self.selected_sample.set(
                "No Samples"
            )

            self.replicate_combo["values"] = [
                "No Replicates"
            ]

            self.selected_replicate.set(
                "No Replicates"
            )

            return

        # Samples
        if "sample_id" in self.df.columns:

            samples = (
                self.df["sample_id"]
                .dropna()
                .astype(str)
                .str.strip()
            )

            samples = sorted(
                value
                for value in samples.unique()
                if value
            )

            sample_values = [
                "All Samples"
            ] + samples

            self.sample_combo["values"] = sample_values

            current = self.selected_sample.get()

            if current not in sample_values:
                self.selected_sample.set(
                    "All Samples"
                )

        else:
            self.sample_combo["values"] = [
                "All Samples"
            ]

            self.selected_sample.set(
                "All Samples"
            )

        self._update_replicate_selector()

    def _update_replicate_selector(self):
        """
        Populate replicate selector according to the selected sample.
        """

        if (
            self.df is None
            or self.df.empty
            or "replicate" not in self.df.columns
        ):
            self.replicate_combo["values"] = [
                "All Replicates"
            ]

            self.selected_replicate.set(
                "All Replicates"
            )

            return

        working = self.df

        sample = self.selected_sample.get()

        if (
            sample
            and sample != "All Samples"
            and "sample_id" in working.columns
        ):
            working = working[
                working["sample_id"]
                .astype(str)
                .str.strip()
                == str(sample).strip()
            ]

        replicates = (
            working["replicate"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        replicates = sorted(
            value
            for value in replicates.unique()
            if value
        )

        values = [
            "All Replicates"
        ] + replicates

        self.replicate_combo["values"] = values

        current = self.selected_replicate.get()

        if current not in values:
            self.selected_replicate.set(
                "All Replicates"
            )

    def _get_filtered_dataframe(self) -> pd.DataFrame:
        """
        Return crushing data according to the current filters.
        """

        if (
            self.crushing_df is None
            or self.crushing_df.empty
        ):
            return pd.DataFrame()

        filtered = self.crushing_df.copy()

        sample = self.selected_sample.get()
        replicate = self.selected_replicate.get()

        if (
            sample
            and sample != "All Samples"
            and "sample_id" in filtered.columns
        ):
            filtered = filtered[
                filtered["sample_id"]
                .astype(str)
                .str.strip()
                == str(sample).strip()
            ]

        if (
            replicate
            and replicate != "All Replicates"
            and "replicate" in filtered.columns
        ):
            filtered = filtered[
                filtered["replicate"]
                .astype(str)
                .str.strip()
                == str(replicate).strip()
            ]

        return filtered.reset_index(
            drop=True
        )

    def _update_crushing_data(self):
        """
        Prepare crushing data and apply current filters.
        """

        self.crushing_df = self._prepare_dataframe(
            self.df
            if self.df is not None
            else pd.DataFrame()
        )

    # ======================================================================
    # KPI CALCULATIONS
    # ======================================================================

    def _update_kpis(self):
        """
        Calculate and display KPI values.
        """

        # Reset first.
        for variable in self.kpi_vars.values():
            variable.set("N/A")

        filtered = self._get_filtered_dataframe()

        if filtered.empty:
            self.kpi_vars["samples"].set("0")
            return

        # Number of experimental runs.
        if "sample_id" in filtered.columns:
            self.kpi_vars["samples"].set(
                str(
                    filtered["sample_id"]
                    .nunique()
                )
            )
        else:
            self.kpi_vars["samples"].set(
                str(len(filtered))
            )

        # Feed mass.
        if "feed_mass" in filtered.columns:
            value = self._mean_value(
                filtered["feed_mass"]
            )

            if value is not None:
                self.kpi_vars["feed_mass"].set(
                    f"{value:.2f} g"
                )

        # Crushing time.
        if "crush_time" in filtered.columns:
            value = self._mean_value(
                filtered["crush_time"]
            )

            if value is not None:
                self.kpi_vars["crush_time"].set(
                    f"{value:.2f}"
                )

        # Crushing power.
        if "crush_power" in filtered.columns:
            value = self._mean_value(
                filtered["crush_power"]
            )

            if value is not None:
                self.kpi_vars["crush_power"].set(
                    f"{value:.2f}"
                )

        # Feed size.
        if "crush_feed_size" in filtered.columns:
            value = self._mean_value(
                filtered["crush_feed_size"]
            )

            if value is not None:
                self.kpi_vars["feed_size"].set(
                    f"{value:.2f}"
                )

        # Product size.
        if "crush_product_size" in filtered.columns:
            value = self._mean_value(
                filtered["crush_product_size"]
            )

            if value is not None:
                self.kpi_vars["product_size"].set(
                    f"{value:.2f}"
                )

        # Calculate reduction metrics from each record.
        ratios = []
        reductions = []

        for _, row in filtered.iterrows():

            ratio = self._calculate_reduction_ratio(
                row.get("crush_feed_size"),
                row.get("crush_product_size"),
            )

            reduction = self._calculate_size_reduction(
                row.get("crush_feed_size"),
                row.get("crush_product_size"),
            )

            if ratio is not None:
                ratios.append(ratio)

            if reduction is not None:
                reductions.append(reduction)

        if ratios:
            self.kpi_vars["reduction_ratio"].set(
                f"{np.mean(ratios):.2f}"
            )

        if reductions:
            self.kpi_vars["size_reduction"].set(
                f"{np.mean(reductions):.2f}%"
            )

    # ======================================================================
    # TABLE
    # ======================================================================

    def _update_table(self):
        """
        Populate the crushing results Treeview.
        """

        self._clear_table()

        filtered = self._get_filtered_dataframe()

        if filtered.empty:
            return

        for index, row in filtered.iterrows():

            sample_id = self._display_value(
                row.get("sample_id")
            )

            material = self._display_value(
                row.get("material_type")
            )

            replicate = self._display_value(
                row.get("replicate")
            )

            crusher = self._display_value(
                row.get("crusher_type")
            )

            feed_mass = self._format_number(
                row.get("feed_mass"),
                suffix=" g",
            )

            feed_size = self._format_number(
                row.get("crush_feed_size")
            )

            product_size = self._format_number(
                row.get("crush_product_size")
            )

            crush_time = self._format_number(
                row.get("crush_time")
            )

            crush_power = self._format_number(
                row.get("crush_power")
            )

            ratio = self._calculate_reduction_ratio(
                row.get("crush_feed_size"),
                row.get("crush_product_size"),
            )

            size_reduction = self._calculate_size_reduction(
                row.get("crush_feed_size"),
                row.get("crush_product_size"),
            )

            ratio_text = (
                f"{ratio:.2f}"
                if ratio is not None
                else "N/A"
            )

            reduction_text = (
                f"{size_reduction:.2f}%"
                if size_reduction is not None
                else "N/A"
            )

            values = (
                sample_id,
                material,
                replicate,
                crusher,
                feed_mass,
                feed_size,
                product_size,
                crush_time,
                crush_power,
                ratio_text,
                reduction_text,
            )

            self.tree.insert(
                "",
                "end",
                iid=str(index),
                values=values,
            )

    def _clear_table(self):
        """
        Remove all rows from the results table.
        """

        if not hasattr(self, "tree"):
            return

        for item in self.tree.get_children():
            self.tree.delete(item)

    # ======================================================================
    # SELECTION EVENTS
    # ======================================================================

    def _on_sample_changed(self, event=None):
        """
        Respond to sample selection.
        """

        self._update_replicate_selector()
        self._update_kpis()
        self._update_table()

        self._set_status(
            f"Displaying crushing results for "
            f"{self.selected_sample.get()}."
        )

    def _on_replicate_changed(self, event=None):
        """
        Respond to replicate selection.
        """

        self._update_kpis()
        self._update_table()

        self._set_status(
            f"Displaying replicate "
            f"{self.selected_replicate.get()}."
        )

    def _reset_filters(self):
        """
        Reset sample and replicate filters.
        """

        self.selected_sample.set(
            "All Samples"
        )

        self._update_replicate_selector()

        self.selected_replicate.set(
            "All Replicates"
        )

        self._update_kpis()
        self._update_table()

        self._set_status(
            "Crushing filters reset."
        )

    def _on_table_selection(self, event=None):
        """
        Display detailed information for the selected row.
        """

        selection = self.tree.selection()

        if not selection:
            return

        item_id = selection[0]

        try:
            index = int(item_id)
        except (TypeError, ValueError):
            return

        filtered = self._get_filtered_dataframe()

        if (
            filtered.empty
            or index < 0
            or index >= len(filtered)
        ):
            return

        row = filtered.iloc[index]

        self._show_record_details(row)

    # ======================================================================
    # DETAILS
    # ======================================================================

    def _show_record_details(
        self,
        row: pd.Series,
    ):
        """
        Show detailed information about one crushing record.
        """

        details = []

        sample_id = self._display_value(
            row.get("sample_id")
        )

        material = self._display_value(
            row.get("material_type")
        )

        replicate = self._display_value(
            row.get("replicate")
        )

        crusher = self._display_value(
            row.get("crusher_type")
        )

        feed_mass = self._format_number(
            row.get("feed_mass"),
            suffix=" g",
        )

        feed_size = self._format_number(
            row.get("crush_feed_size")
        )

        product_size = self._format_number(
            row.get("crush_product_size")
        )

        crush_time = self._format_number(
            row.get("crush_time")
        )

        power = self._format_number(
            row.get("crush_power")
        )

        ratio = self._calculate_reduction_ratio(
            row.get("crush_feed_size"),
            row.get("crush_product_size"),
        )

        reduction = self._calculate_size_reduction(
            row.get("crush_feed_size"),
            row.get("crush_product_size"),
        )

        details.append(
            f"Sample: {sample_id}"
        )

        details.append(
            f"Material: {material}"
        )

        details.append(
            f"Replicate: {replicate}"
        )

        details.append(
            f"Crusher Type: {crusher}"
        )

        details.append(
            f"Feed Mass: {feed_mass}"
        )

        details.append(
            f"Feed Size: {feed_size}"
        )

        details.append(
            f"Product Size: {product_size}"
        )

        details.append(
            f"Crushing Time: {crush_time}"
        )

        details.append(
            f"Crusher Power: {power}"
        )

        if ratio is not None:
            details.append(
                f"Size Reduction Ratio: {ratio:.3f}"
            )
        else:
            details.append(
                "Size Reduction Ratio: N/A"
            )

        if reduction is not None:
            details.append(
                f"Percentage Size Reduction: "
                f"{reduction:.2f}%"
            )
        else:
            details.append(
                "Percentage Size Reduction: N/A"
            )

        self.details_text.configure(
            state="normal"
        )

        self.details_text.delete(
            "1.0",
            tk.END,
        )

        self.details_text.insert(
            "1.0",
            " | ".join(details),
        )

        self.details_text.configure(
            state="disabled"
        )

    # ======================================================================
    # CALCULATIONS
    # ======================================================================

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        """
        Safely convert a value to float.
        """

        if value is None:
            return None

        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(number):
            return None

        return number

    @classmethod
    def _calculate_reduction_ratio(
        cls,
        feed_size: Any,
        product_size: Any,
    ) -> Optional[float]:
        """
        Calculate crushing size reduction ratio.

        Formula:

            Reduction Ratio = Feed Size / Product Size
        """

        feed = cls._to_float(feed_size)
        product = cls._to_float(product_size)

        if feed is None or product is None:
            return None

        if feed <= 0 or product <= 0:
            return None

        return feed / product

    @classmethod
    def _calculate_size_reduction(
        cls,
        feed_size: Any,
        product_size: Any,
    ) -> Optional[float]:
        """
        Calculate percentage size reduction.

        Formula:

            ((Feed Size - Product Size) / Feed Size) × 100
        """

        feed = cls._to_float(feed_size)
        product = cls._to_float(product_size)

        if feed is None or product is None:
            return None

        if feed <= 0:
            return None

        return (
            (feed - product)
            / feed
        ) * 100.0

    @staticmethod
    def _mean_value(
        series: pd.Series,
    ) -> Optional[float]:
        """
        Calculate the mean of valid numeric values.
        """

        if series is None:
            return None

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        ).dropna()

        if numeric.empty:
            return None

        value = numeric.mean()

        if not np.isfinite(value):
            return None

        return float(value)

    # ======================================================================
    # DISPLAY HELPERS
    # ======================================================================

    @staticmethod
    def _display_value(
        value: Any,
    ) -> str:
        """
        Convert a dataframe value into a user-friendly string.
        """

        if value is None:
            return "N/A"

        try:
            if pd.isna(value):
                return "N/A"
        except (TypeError, ValueError):
            pass

        text = str(value).strip()

        if not text:
            return "N/A"

        return text

    @classmethod
    def _format_number(
        cls,
        value: Any,
        suffix: str = "",
    ) -> str:
        """
        Format numeric values consistently.
        """

        number = cls._to_float(value)

        if number is None:
            return "N/A"

        return f"{number:.2f}{suffix}"

    def _set_status(
        self,
        message: str,
    ):
        """
        Update status bar.
        """

        self.status_var.set(
            str(message)
        )


# ==========================================================================
# CONVENIENCE FUNCTION
# ==========================================================================

def create_crushing_view(parent, controller):
    """
    Convenience function for creating a CrushingView.

    Parameters
    ----------
    parent:
        Parent Tkinter widget.

    controller:
        Main application controller.

    Returns
    -------
    CrushingView
        Newly created CrushingView instance.
    """

    return CrushingView(
        parent,
        controller,
    )


# ==========================================================================
# PUBLIC API
# ==========================================================================

__all__ = [
    "CrushingView",
    "create_crushing_view",
]