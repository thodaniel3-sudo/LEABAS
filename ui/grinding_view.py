"""
LEBAS - Grinding Analysis View
File: ui/grinding_view.py

Purpose
-------
This module provides the graphical user interface for the Grinding Analysis
section of LEBAS (Laboratory Extraction and Beneficiation Analysis System).

Important design principles
---------------------------
1. All experimental data comes from dataset.csv.
2. No experimental values are hard-coded.
3. Multiple sieve-analysis rows belonging to the same experiment are NOT
   counted as multiple grinding experiments.
4. One grinding record is created for each sample_id + replicate.
5. Missing or invalid numeric values are handled safely.
6. The Treeview uses the standard ttk "Treeview" style to avoid the
   "Layout GrindingTreeview not found" TclError.
7. Sieve sizes are treated dynamically. The application does not assume
   that only a fixed list of sieve sizes exists.
8. The view can be refreshed after dataset.csv is changed.
"""

import logging
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Any, Dict, List

import numpy as np
import pandas as pd

from config import Config
from services.data_loader import load_dataset


# -----------------------------------------------------------------------------
# LOGGER
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# =============================================================================
# GRINDING VIEW
# =============================================================================

class GrindingView:
    """
    Main graphical interface for grinding analysis.

    The class receives the main application's content frame and controller,
    prepares grinding records from the dataset, calculates KPIs, displays
    the records in a Treeview, and displays detailed information for a
    selected record.
    """

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------

    def __init__(self, parent, controller):
        """
        Initialize the Grinding View.

        Parameters
        ----------
        parent:
            Tkinter parent widget.

        controller:
            Reference to the main LEBAS application controller.
        """

        self.controller = controller

        # Main frame used by main.py.
        self.frame = ttk.Frame(parent)

        # Raw dataset.
        self.df = pd.DataFrame()

        # Prepared one-record-per-sample/replicate grinding dataset.
        self.grinding_df = pd.DataFrame()

        # Filter variables.
        self.selected_sample = tk.StringVar(value="All Samples")
        self.selected_replicate = tk.StringVar(value="All Replicates")

        # Status bar variable.
        self.status_var = tk.StringVar(value="Ready")

        # KPI variables.
        self.kpi_vars = {
            "samples": tk.StringVar(value="0"),
            "feed_mass": tk.StringVar(value="0"),
            "grinding_time": tk.StringVar(value="0"),
            "speed": tk.StringVar(value="0"),
            "media_mass": tk.StringVar(value="0"),
            "feed_size": tk.StringVar(value="0"),
            "product_size": tk.StringVar(value="0"),
            "reduction_ratio": tk.StringVar(value="0"),
        }

        # Detail variables.
        self.detail_vars = {
            "sample_id": tk.StringVar(value="N/A"),
            "material": tk.StringVar(value="N/A"),
            "date": tk.StringVar(value="N/A"),
            "operator": tk.StringVar(value="N/A"),
            "replicate": tk.StringVar(value="N/A"),
            "mill": tk.StringVar(value="N/A"),
            "feed_mass": tk.StringVar(value="N/A"),
            "time": tk.StringVar(value="N/A"),
            "media": tk.StringVar(value="N/A"),
            "speed": tk.StringVar(value="N/A"),
            "feed_size": tk.StringVar(value="N/A"),
            "product_size": tk.StringVar(value="N/A"),
            "ratio": tk.StringVar(value="N/A"),
            "reduction": tk.StringVar(value="N/A"),
            "yield": tk.StringVar(value="N/A"),
            "mass_loss": tk.StringVar(value="N/A"),
            "media_ratio": tk.StringVar(value="N/A"),
            "specific_energy": tk.StringVar(value="N/A"),
            "p80": tk.StringVar(value="N/A"),
        }

        # Configure styles before creating widgets.
        self._configure_style()

        # Build interface.
        self._build_ui()

        # Load data.
        self.refresh()

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def get_frame(self):
        """
        Return the main frame.

        main.py uses this method to place the view in the application.
        """
        return self.frame

    # -------------------------------------------------------------------------

    def refresh(self):
        """
        Reload dataset.csv and refresh the entire grinding interface.

        This method is intentionally defensive. If the dataset is missing or
        invalid, the application does not crash completely. The error is
        logged and displayed to the user.
        """

        try:
            logger.info("Refreshing Grinding View...")

            # Load the current dataset through the central data loader.
            self.df = load_dataset()

            if self.df is None:
                self.df = pd.DataFrame()

            # Prepare one grinding record per sample + replicate.
            self.grinding_df = self._prepare_dataframe(self.df)

            # Update filter dropdowns.
            self._update_filter_values()

            # Update KPI cards.
            self._update_kpis()

            # Update table.
            self._update_table()

            # Clear previously selected details.
            self._clear_details()

            # Update status.
            if self.grinding_df.empty:
                self.status_var.set("No grinding data available")
            else:
                self.status_var.set(
                    f"{len(self.grinding_df):,} grinding record(s) loaded"
                )

            logger.info(
                "Grinding View refreshed successfully. Records: %s",
                len(self.grinding_df),
            )

        except Exception as exc:
            logger.exception("Failed to refresh Grinding View")

            self.grinding_df = pd.DataFrame()

            self.status_var.set("Error loading grinding data")

            try:
                messagebox.showerror(
                    "Grinding Data Error",
                    "Unable to load grinding data.\n\n"
                    f"{exc}",
                )
            except Exception:
                pass

    # =========================================================================
    # STYLE CONFIGURATION
    # =========================================================================

    def _configure_style(self):
        """
        Configure ttk styles used by the Grinding View.

        IMPORTANT
        ---------
        We deliberately use the standard ttk Treeview style.

        The previous version used:

            style="GrindingTreeview"

        without registering a custom Treeview layout. That caused:

            TclError: Layout GrindingTreeview not found

        Using the standard "Treeview" style completely avoids that problem.
        """

        style = ttk.Style()

        try:
            # General title.
            style.configure(
                "GrindingTitle.TLabel",
                font=("Segoe UI", 20, "bold"),
            )

            # Subtitle.
            style.configure(
                "GrindingSubtitle.TLabel",
                font=("Segoe UI", 10),
            )

            # KPI title.
            style.configure(
                "GrindingCardTitle.TLabel",
                font=("Segoe UI", 9),
            )

            # KPI value.
            style.configure(
                "GrindingCardValue.TLabel",
                font=("Segoe UI", 16, "bold"),
            )

            # Section headings.
            style.configure(
                "GrindingSection.TLabel",
                font=("Segoe UI", 12, "bold"),
            )

            # IMPORTANT:
            # Use the standard Treeview style.
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
            logger.exception("Could not configure Grinding View styles")

    # =========================================================================
    # USER INTERFACE
    # =========================================================================

    def _build_ui(self):
        """
        Build the complete Grinding Analysis interface.
        """

        # Main frame layout.
        self.frame.grid_rowconfigure(3, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

        # =====================================================================
        # HEADER
        # =====================================================================

        header = ttk.Frame(self.frame)

        header.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=20,
            pady=(18, 10),
        )

        header.grid_columnconfigure(0, weight=1)

        ttk.Label(
            header,
            text="Grinding Analysis",
            style="GrindingTitle.TLabel",
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        subtitle = (
            "Grinding process performance and particle-size reduction"
        )

        # Config.py contains APP_VERSION, not VERSION.
        version_text = getattr(Config, "APP_VERSION", "")

        if version_text:
            subtitle += f"  •  Version {version_text}"

        ttk.Label(
            header,
            text=subtitle,
            style="GrindingSubtitle.TLabel",
        ).grid(
            row=1,
            column=0,
            sticky="w",
            pady=(3, 0),
        )

        # =====================================================================
        # FILTERS AND ACTIONS
        # =====================================================================

        control_frame = ttk.LabelFrame(
            self.frame,
            text="Filters and Actions",
            padding=10,
        )

        control_frame.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=20,
            pady=(0, 12),
        )

        control_frame.grid_columnconfigure(1, weight=1)
        control_frame.grid_columnconfigure(3, weight=1)

        # Sample label.
        ttk.Label(
            control_frame,
            text="Sample:",
        ).grid(
            row=0,
            column=0,
            padx=(0, 6),
            pady=4,
            sticky="w",
        )

        # Sample filter.
        self.sample_combo = ttk.Combobox(
            control_frame,
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

        # Replicate label.
        ttk.Label(
            control_frame,
            text="Replicate:",
        ).grid(
            row=0,
            column=2,
            padx=(0, 6),
            pady=4,
            sticky="w",
        )

        # Replicate filter.
        self.replicate_combo = ttk.Combobox(
            control_frame,
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

        # Refresh button.
        ttk.Button(
            control_frame,
            text="↻ Refresh",
            command=self.refresh,
        ).grid(
            row=0,
            column=4,
            padx=5,
            pady=4,
        )

        # Reset button.
        ttk.Button(
            control_frame,
            text="Reset Filters",
            command=self._reset_filters,
        ).grid(
            row=0,
            column=5,
            padx=(5, 0),
            pady=4,
        )

        # =====================================================================
        # KPI CARDS
        # =====================================================================

        kpi_container = ttk.Frame(self.frame)

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
                uniform="grinding_kpi",
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
            "Avg Grinding Time",
            self.kpi_vars["grinding_time"],
        )

        self._create_kpi_card(
            kpi_container,
            3,
            "Avg Grinding Speed",
            self.kpi_vars["speed"],
        )

        self._create_kpi_card(
            kpi_container,
            4,
            "Avg Media Mass",
            self.kpi_vars["media_mass"],
        )

        self._create_kpi_card(
            kpi_container,
            5,
            "Avg Feed Size",
            self.kpi_vars["feed_size"],
        )

        self._create_kpi_card(
            kpi_container,
            6,
            "Avg Product Size",
            self.kpi_vars["product_size"],
        )

        self._create_kpi_card(
            kpi_container,
            7,
            "Reduction Ratio",
            self.kpi_vars["reduction_ratio"],
        )

        # =====================================================================
        # MAIN CONTENT
        # =====================================================================

        content = ttk.Frame(self.frame)

        content.grid(
            row=3,
            column=0,
            sticky="nsew",
            padx=20,
            pady=(0, 10),
        )

        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=1)

        # =====================================================================
        # TABLE
        # =====================================================================

        table_frame = ttk.LabelFrame(
            content,
            text="Grinding Records",
            padding=8,
        )

        table_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 10),
        )

        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        columns = (
            "sample_id",
            "material",
            "replicate",
            "mill_type",
            "feed_mass",
            "feed_size",
            "product_size",
            "grinding_time",
            "media_mass",
            "speed",
            "reduction_ratio",
            "size_reduction",
        )

        # ---------------------------------------------------------------------
        # IMPORTANT FIX
        # ---------------------------------------------------------------------
        # Use the built-in "Treeview" style.
        #
        # DO NOT use:
        #
        #     style="GrindingTreeview"
        #
        # because that custom layout was not registered.
        # ---------------------------------------------------------------------

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
            "mill_type": "Mill Type",
            "feed_mass": "Feed Mass",
            "feed_size": "Feed Size",
            "product_size": "Product Size",
            "grinding_time": "Grinding Time",
            "media_mass": "Media Mass",
            "speed": "Grinding Speed",
            "reduction_ratio": "Reduction Ratio",
            "size_reduction": "Size Reduction (%)",
        }

        widths = {
            "sample_id": 120,
            "material": 120,
            "replicate": 80,
            "mill_type": 120,
            "feed_mass": 100,
            "feed_size": 100,
            "product_size": 105,
            "grinding_time": 110,
            "media_mass": 105,
            "speed": 105,
            "reduction_ratio": 115,
            "size_reduction": 125,
        }

        # Configure headings and columns.
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

        # Vertical scrollbar.
        y_scroll = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.tree.yview,
        )

        # Horizontal scrollbar.
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

        # Selecting a table row updates the details panel.
        self.tree.bind(
            "<<TreeviewSelect>>",
            self._on_row_selected,
        )

        # =====================================================================
        # DETAILS PANEL
        # =====================================================================

        details_frame = ttk.LabelFrame(
            content,
            text="Grinding Details",
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
            ("Mill Type", "mill"),
            ("Feed Mass", "feed_mass"),
            ("Grinding Time", "time"),
            ("Media Mass", "media"),
            ("Grinding Speed", "speed"),
            ("Feed Size", "feed_size"),
            ("Product Size", "product_size"),
            ("Reduction Ratio", "ratio"),
            ("Size Reduction", "reduction"),
            ("Grinding Yield", "yield"),
            ("Mass Loss", "mass_loss"),
            ("Media/Feed Ratio", "media_ratio"),
            ("Specific Energy", "specific_energy"),
            ("P80", "p80"),
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

        # =====================================================================
        # STATUS BAR
        # =====================================================================

        status_frame = ttk.Frame(self.frame)

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
            side="left",
        )

    # =========================================================================
    # KPI CARD
    # =========================================================================

    def _create_kpi_card(
        self,
        parent,
        column: int,
        title: str,
        variable: tk.StringVar,
    ):
        """
        Create a single KPI card.
        """

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
            style="GrindingCardTitle.TLabel",
            anchor="center",
        ).pack(
            fill="x",
        )

        ttk.Label(
            card,
            textvariable=variable,
            style="GrindingCardValue.TLabel",
            anchor="center",
        ).pack(
            fill="x",
            pady=(4, 0),
        )

    # =========================================================================
    # DATA PREPARATION
    # =========================================================================

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare one grinding record per sample_id + replicate.

        Why this is important
        ----------------------
        A single grinding experiment may have several rows in dataset.csv
        because sieve analysis records one row for each sieve size.

        Example:

            S001 | Rep 1 | 1000 um
            S001 | Rep 1 | 710 um
            S001 | Rep 1 | 500 um
            S001 | Rep 1 | 355 um
            S001 | Rep 1 | 250 um

        These are NOT five grinding experiments.

        They represent ONE grinding experiment with five sieve fractions.

        Therefore, the Grinding View creates exactly one record for:

            sample_id + replicate

        and retains the most complete process row.
        """

        if df is None or df.empty:
            return pd.DataFrame()

        working = df.copy()

        # Normalize column names.
        working.columns = [
            str(column).strip().lower()
            for column in working.columns
        ]

        # -------------------------------------------------------------
        # Ensure grouping columns exist.
        # -------------------------------------------------------------

        if "sample_id" not in working.columns:
            working["sample_id"] = "Unknown"

        if "replicate" not in working.columns:
            working["replicate"] = "1"

        # -------------------------------------------------------------
        # Convert numeric fields safely.
        # -------------------------------------------------------------

        numeric_columns = [
            "feed_mass",
            "grinding_time",
            "grinding_media_mass",
            "grinding_speed",
            "grinding_feed_size",
            "grinding_product_size",
            "losses_mass",
            "crush_power",
            "sieve_size",
            "mass_retained",
        ]

        for column in numeric_columns:

            if column in working.columns:

                working[column] = pd.to_numeric(
                    working[column],
                    errors="coerce",
                )

        # -------------------------------------------------------------
        # Remove rows with absolutely no grinding information.
        # -------------------------------------------------------------

        grinding_columns = [
            column
            for column in [
                "grinding_time",
                "grinding_mill_type",
                "grinding_media_mass",
                "grinding_speed",
                "grinding_feed_size",
                "grinding_product_size",
            ]
            if column in working.columns
        ]

        if grinding_columns:

            has_grinding_data = (
                working[grinding_columns]
                .notna()
                .any(axis=1)
            )

            working = working.loc[
                has_grinding_data
            ].copy()

        if working.empty:
            return pd.DataFrame()

        # -------------------------------------------------------------
        # Normalize identifiers.
        # -------------------------------------------------------------

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

        # -------------------------------------------------------------
        # Calculate row completeness.
        #
        # If several sieve rows contain the same process data, this score
        # helps us select the most complete process record.
        # -------------------------------------------------------------

        process_columns = [
            "feed_mass",
            "grinding_time",
            "grinding_mill_type",
            "grinding_media_mass",
            "grinding_speed",
            "grinding_feed_size",
            "grinding_product_size",
            "material_type",
            "experiment_date",
            "operator",
            "losses_mass",
            "crush_power",
        ]

        existing_process_columns = [
            column
            for column in process_columns
            if column in working.columns
        ]

        if existing_process_columns:

            working["_completeness"] = (
                working[existing_process_columns]
                .notna()
                .sum(axis=1)
            )

        else:

            working["_completeness"] = 0

        # -------------------------------------------------------------
        # Sort by sample, replicate and completeness.
        # Most complete row appears first.
        # -------------------------------------------------------------

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

        # -------------------------------------------------------------
        # Remove duplicate process records.
        # -------------------------------------------------------------

        process_df = (
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

        # -------------------------------------------------------------
        # Calculate derived grinding metrics.
        # -------------------------------------------------------------

        process_df["reduction_ratio"] = np.nan
        process_df["size_reduction_percent"] = np.nan
        process_df["ground_product_mass"] = np.nan
        process_df["mass_loss_percent"] = np.nan
        process_df["grinding_yield"] = np.nan
        process_df["media_to_feed_ratio"] = np.nan
        process_df["specific_energy"] = np.nan
        process_df["p80"] = np.nan

        # -------------------------------------------------------------
        # Size reduction.
        #
        # Reduction Ratio = Feed Size / Product Size
        #
        # Size Reduction (%) =
        #     ((Feed Size - Product Size) / Feed Size) * 100
        # -------------------------------------------------------------

        if {
            "grinding_feed_size",
            "grinding_product_size",
        }.issubset(process_df.columns):

            feed_size = pd.to_numeric(
                process_df["grinding_feed_size"],
                errors="coerce",
            )

            product_size = pd.to_numeric(
                process_df["grinding_product_size"],
                errors="coerce",
            )

            valid = (
                feed_size.gt(0)
                & product_size.gt(0)
            )

            process_df.loc[
                valid,
                "reduction_ratio",
            ] = (
                feed_size[valid]
                / product_size[valid]
            )

            process_df.loc[
                valid,
                "size_reduction_percent",
            ] = (
                (
                    feed_size[valid]
                    - product_size[valid]
                )
                / feed_size[valid]
            ) * 100.0

        # -------------------------------------------------------------
        # Ground product mass and yield.
        #
        # If losses_mass exists:
        #
        # Product Mass = Feed Mass - Losses
        #
        # Yield = Product Mass / Feed Mass * 100
        #
        # If losses_mass is unavailable, yield remains based on available
        # information instead of inventing an experimental value.
        # -------------------------------------------------------------

        if "feed_mass" in process_df.columns:

            feed_mass = pd.to_numeric(
                process_df["feed_mass"],
                errors="coerce",
            )

            if "losses_mass" in process_df.columns:

                losses = pd.to_numeric(
                    process_df["losses_mass"],
                    errors="coerce",
                )

                valid_loss = (
                    feed_mass.gt(0)
                    & losses.notna()
                )

                process_df.loc[
                    valid_loss,
                    "ground_product_mass",
                ] = (
                    feed_mass[valid_loss]
                    - losses[valid_loss]
                )

                process_df.loc[
                    valid_loss,
                    "mass_loss_percent",
                ] = (
                    losses[valid_loss]
                    / feed_mass[valid_loss]
                ) * 100.0

                process_df.loc[
                    valid_loss,
                    "grinding_yield",
                ] = (
                    process_df.loc[
                        valid_loss,
                        "ground_product_mass",
                    ]
                    / feed_mass[valid_loss]
                ) * 100.0

        # -------------------------------------------------------------
        # Media-to-feed ratio.
        # -------------------------------------------------------------

        if {
            "feed_mass",
            "grinding_media_mass",
        }.issubset(process_df.columns):

            feed_mass = pd.to_numeric(
                process_df["feed_mass"],
                errors="coerce",
            )

            media_mass = pd.to_numeric(
                process_df["grinding_media_mass"],
                errors="coerce",
            )

            valid = (
                feed_mass.gt(0)
                & media_mass.notna()
            )

            process_df.loc[
                valid,
                "media_to_feed_ratio",
            ] = (
                media_mass[valid]
                / feed_mass[valid]
            )

        # -------------------------------------------------------------
        # Specific energy.
        #
        # IMPORTANT:
        #
        # The dataset has crush_power but does not have a dedicated
        # grinding_power column.
        #
        # Therefore, we DO NOT falsely present crush_power as actual
        # grinding power.
        #
        # Specific energy is calculated only if a genuine
        # grinding_power column exists.
        # -------------------------------------------------------------

        if {
            "feed_mass",
            "grinding_time",
            "grinding_power",
        }.issubset(process_df.columns):

            feed_mass = pd.to_numeric(
                process_df["feed_mass"],
                errors="coerce",
            )

            grinding_time = pd.to_numeric(
                process_df["grinding_time"],
                errors="coerce",
            )

            grinding_power = pd.to_numeric(
                process_df["grinding_power"],
                errors="coerce",
            )

            valid = (
                feed_mass.gt(0)
                & grinding_time.notna()
                & grinding_power.notna()
            )

            energy_kwh = (
                grinding_power[valid]
                * (grinding_time[valid] / 60.0)
            )

            process_df.loc[
                valid,
                "specific_energy",
            ] = (
                energy_kwh
                / feed_mass[valid]
            ) * 1000.0

        # -------------------------------------------------------------
        # Remove helper column.
        # -------------------------------------------------------------

        if "_completeness" in process_df.columns:
            process_df.drop(
                columns=["_completeness"],
                inplace=True,
            )

        # Reset index.
        process_df.reset_index(
            drop=True,
            inplace=True,
        )

        return process_df

    # =========================================================================
    # FILTER VALUES
    # =========================================================================

    def _update_filter_values(self):
        """
        Populate Sample and Replicate dropdowns from the current dataset.
        """

        # -------------------------------------------------------------
        # Sample values.
        # -------------------------------------------------------------

        samples = ["All Samples"]

        if (
            not self.grinding_df.empty
            and "sample_id" in self.grinding_df.columns
        ):

            unique_samples = (
                self.grinding_df["sample_id"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )

            unique_samples = sorted(
                unique_samples,
                key=lambda value: value.lower(),
            )

            samples.extend(unique_samples)

        self.sample_combo["values"] = samples

        # Preserve selected sample if it still exists.
        if self.selected_sample.get() not in samples:
            self.selected_sample.set("All Samples")

        # -------------------------------------------------------------
        # Replicate values.
        # -------------------------------------------------------------

        self._update_replicate_values()

    # -------------------------------------------------------------------------

    def _update_replicate_values(self):
        """
        Update replicate dropdown based on the selected sample.
        """

        replicates = ["All Replicates"]

        if (
            not self.grinding_df.empty
            and "replicate" in self.grinding_df.columns
        ):

            filtered = self.grinding_df.copy()

            selected_sample = self.selected_sample.get()

            if (
                selected_sample
                and selected_sample != "All Samples"
                and "sample_id" in filtered.columns
            ):

                filtered = filtered[
                    filtered["sample_id"].astype(str)
                    == str(selected_sample)
                ]

            values = (
                filtered["replicate"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )

            # Sort naturally where possible.
            def sort_replicate(value):
                try:
                    return (0, float(value))
                except Exception:
                    return (1, str(value).lower())

            values = sorted(
                values,
                key=sort_replicate,
            )

            replicates.extend(values)

        self.replicate_combo["values"] = replicates

        if self.selected_replicate.get() not in replicates:
            self.selected_replicate.set("All Replicates")

    # =========================================================================
    # FILTERING
    # =========================================================================

    def _get_filtered_dataframe(self) -> pd.DataFrame:
        """
        Return the grinding dataset after applying current filters.
        """

        if self.grinding_df.empty:
            return pd.DataFrame()

        filtered = self.grinding_df.copy()

        selected_sample = self.selected_sample.get()
        selected_replicate = self.selected_replicate.get()

        # Filter sample.
        if (
            selected_sample
            and selected_sample != "All Samples"
            and "sample_id" in filtered.columns
        ):

            filtered = filtered[
                filtered["sample_id"].astype(str)
                == str(selected_sample)
            ]

        # Filter replicate.
        if (
            selected_replicate
            and selected_replicate != "All Replicates"
            and "replicate" in filtered.columns
        ):

            filtered = filtered[
                filtered["replicate"].astype(str)
                == str(selected_replicate)
            ]

        return filtered

    # -------------------------------------------------------------------------

    def _on_filter_changed(self, event=None):
        """
        Respond to a change in Sample or Replicate filter.
        """

        try:
            # If sample changes, update available replicates.
            if event is not None:

                widget = event.widget

                if widget == self.sample_combo:
                    self._update_replicate_values()

            self._update_kpis()
            self._update_table()
            self._clear_details()

            filtered = self._get_filtered_dataframe()

            self.status_var.set(
                f"{len(filtered):,} grinding record(s) displayed"
            )

        except Exception:
            logger.exception("Error applying grinding filters")

    # -------------------------------------------------------------------------

    def _reset_filters(self):
        """
        Reset all filters.
        """

        self.selected_sample.set("All Samples")
        self.selected_replicate.set("All Replicates")

        self._update_replicate_values()
        self._update_kpis()
        self._update_table()
        self._clear_details()

        self.status_var.set(
            f"{len(self.grinding_df):,} grinding record(s) loaded"
        )

    # =========================================================================
    # KPI CALCULATIONS
    # =========================================================================

    def _update_kpis(self):
        """
        Calculate and display KPI values for the currently filtered data.
        """

        filtered = self._get_filtered_dataframe()

        # No data.
        if filtered.empty:

            for variable in self.kpi_vars.values():
                variable.set("0")

            return

        # -------------------------------------------------------------
        # Number of records.
        # -------------------------------------------------------------

        self.kpi_vars["samples"].set(
            f"{len(filtered):,}"
        )

        # -------------------------------------------------------------
        # Feed mass.
        # -------------------------------------------------------------

        feed_mass = self._numeric_mean_or_sum(
            filtered,
            "feed_mass",
            operation="sum",
        )

        if feed_mass is None:
            self.kpi_vars["feed_mass"].set("N/A")
        else:
            self.kpi_vars["feed_mass"].set(
                f"{feed_mass:,.2f} g"
            )

        # -------------------------------------------------------------
        # Average grinding time.
        # -------------------------------------------------------------

        grinding_time = self._numeric_mean(
            filtered,
            "grinding_time",
        )

        if grinding_time is None:
            self.kpi_vars["grinding_time"].set("N/A")
        else:
            self.kpi_vars["grinding_time"].set(
                f"{grinding_time:,.2f} min"
            )

        # -------------------------------------------------------------
        # Average grinding speed.
        # -------------------------------------------------------------

        speed = self._numeric_mean(
            filtered,
            "grinding_speed",
        )

        if speed is None:
            self.kpi_vars["speed"].set("N/A")
        else:
            self.kpi_vars["speed"].set(
                f"{speed:,.2f}"
            )

        # -------------------------------------------------------------
        # Average media mass.
        # -------------------------------------------------------------

        media_mass = self._numeric_mean(
            filtered,
            "grinding_media_mass",
        )

        if media_mass is None:
            self.kpi_vars["media_mass"].set("N/A")
        else:
            self.kpi_vars["media_mass"].set(
                f"{media_mass:,.2f} g"
            )

        # -------------------------------------------------------------
        # Average feed size.
        # -------------------------------------------------------------

        feed_size = self._numeric_mean(
            filtered,
            "grinding_feed_size",
        )

        if feed_size is None:
            self.kpi_vars["feed_size"].set("N/A")
        else:
            self.kpi_vars["feed_size"].set(
                f"{feed_size:,.2f} µm"
            )

        # -------------------------------------------------------------
        # Average product size.
        # -------------------------------------------------------------

        product_size = self._numeric_mean(
            filtered,
            "grinding_product_size",
        )

        if product_size is None:
            self.kpi_vars["product_size"].set("N/A")
        else:
            self.kpi_vars["product_size"].set(
                f"{product_size:,.2f} µm"
            )

        # -------------------------------------------------------------
        # Average reduction ratio.
        # -------------------------------------------------------------

        reduction_ratio = self._numeric_mean(
            filtered,
            "reduction_ratio",
        )

        if reduction_ratio is None:
            self.kpi_vars["reduction_ratio"].set("N/A")
        else:
            self.kpi_vars["reduction_ratio"].set(
                f"{reduction_ratio:,.2f}:1"
            )

    # =========================================================================
    # TABLE UPDATE
    # =========================================================================

    def _update_table(self):
        """
        Refresh the grinding records Treeview.
        """

        # Delete all existing rows.
        for item in self.tree.get_children():
            self.tree.delete(item)

        filtered = self._get_filtered_dataframe()

        if filtered.empty:
            return

        # -------------------------------------------------------------
        # Add records to Treeview.
        # -------------------------------------------------------------

        for index, row in filtered.iterrows():

            values = (
                self._display_text(row.get("sample_id")),
                self._display_text(row.get("material_type")),
                self._display_text(row.get("replicate")),
                self._display_text(row.get("grinding_mill_type")),
                self._format_number(
                    row.get("feed_mass"),
                    2,
                ),
                self._format_number(
                    row.get("grinding_feed_size"),
                    2,
                ),
                self._format_number(
                    row.get("grinding_product_size"),
                    2,
                ),
                self._format_number(
                    row.get("grinding_time"),
                    2,
                ),
                self._format_number(
                    row.get("grinding_media_mass"),
                    2,
                ),
                self._format_number(
                    row.get("grinding_speed"),
                    2,
                ),
                self._format_number(
                    row.get("reduction_ratio"),
                    2,
                ),
                self._format_number(
                    row.get("size_reduction_percent"),
                    2,
                ),
            )

            # Store the DataFrame index as the Treeview item identifier.
            self.tree.insert(
                "",
                "end",
                iid=str(index),
                values=values,
            )

    # =========================================================================
    # TABLE SELECTION
    # =========================================================================

    def _on_row_selected(self, event=None):
        """
        Display detailed information for the selected grinding record.
        """

        try:

            selection = self.tree.selection()

            if not selection:
                return

            item_id = selection[0]

            try:
                index = int(item_id)
            except ValueError:
                return

            filtered = self._get_filtered_dataframe()

            if index not in filtered.index:
                return

            row = filtered.loc[index]

            self._show_details(row)

        except Exception:
            logger.exception(
                "Error displaying selected grinding record"
            )

    # =========================================================================
    # DETAILS
    # =========================================================================

    def _show_details(self, row: pd.Series):
        """
        Display all available details for a selected grinding record.
        """

        self.detail_vars["sample_id"].set(
            self._display_text(
                row.get("sample_id")
            )
        )

        self.detail_vars["material"].set(
            self._display_text(
                row.get("material_type")
            )
        )

        self.detail_vars["date"].set(
            self._display_text(
                row.get("experiment_date")
            )
        )

        self.detail_vars["operator"].set(
            self._display_text(
                row.get("operator")
            )
        )

        self.detail_vars["replicate"].set(
            self._display_text(
                row.get("replicate")
            )
        )

        self.detail_vars["mill"].set(
            self._display_text(
                row.get("grinding_mill_type")
            )
        )

        self.detail_vars["feed_mass"].set(
            self._format_with_unit(
                row.get("feed_mass"),
                "g",
            )
        )

        self.detail_vars["time"].set(
            self._format_with_unit(
                row.get("grinding_time"),
                "min",
            )
        )

        self.detail_vars["media"].set(
            self._format_with_unit(
                row.get("grinding_media_mass"),
                "g",
            )
        )

        self.detail_vars["speed"].set(
            self._format_number(
                row.get("grinding_speed"),
                2,
            )
        )

        self.detail_vars["feed_size"].set(
            self._format_with_unit(
                row.get("grinding_feed_size"),
                "µm",
            )
        )

        self.detail_vars["product_size"].set(
            self._format_with_unit(
                row.get("grinding_product_size"),
                "µm",
            )
        )

        ratio = self._safe_float(
            row.get("reduction_ratio")
        )

        if ratio is None:
            self.detail_vars["ratio"].set("N/A")
        else:
            self.detail_vars["ratio"].set(
                f"{ratio:.3f}:1"
            )

        reduction = self._safe_float(
            row.get("size_reduction_percent")
        )

        if reduction is None:
            self.detail_vars["reduction"].set("N/A")
        else:
            self.detail_vars["reduction"].set(
                f"{reduction:.2f}%"
            )

        yield_value = self._safe_float(
            row.get("grinding_yield")
        )

        if yield_value is None:
            self.detail_vars["yield"].set("N/A")
        else:
            self.detail_vars["yield"].set(
                f"{yield_value:.2f}%"
            )

        mass_loss = self._safe_float(
            row.get("mass_loss_percent")
        )

        if mass_loss is None:
            self.detail_vars["mass_loss"].set("N/A")
        else:
            self.detail_vars["mass_loss"].set(
                f"{mass_loss:.2f}%"
            )

        media_ratio = self._safe_float(
            row.get("media_to_feed_ratio")
        )

        if media_ratio is None:
            self.detail_vars["media_ratio"].set("N/A")
        else:
            self.detail_vars["media_ratio"].set(
                f"{media_ratio:.3f}"
            )

        specific_energy = self._safe_float(
            row.get("specific_energy")
        )

        if specific_energy is None:
            self.detail_vars["specific_energy"].set(
                "N/A"
            )
        else:
            self.detail_vars["specific_energy"].set(
                f"{specific_energy:.3f} kWh/t"
            )

        p80 = self._safe_float(
            row.get("p80")
        )

        if p80 is None:
            self.detail_vars["p80"].set("N/A")
        else:
            self.detail_vars["p80"].set(
                f"{p80:.2f} µm"
            )

    # -------------------------------------------------------------------------

    def _clear_details(self):
        """
        Clear the details panel.
        """

        for variable in self.detail_vars.values():
            variable.set("N/A")

    # =========================================================================
    # NUMERIC HELPERS
    # =========================================================================

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """
        Safely convert a value to float.

        Returns None for:
            - None
            - NaN
            - invalid text
        """

        if value is None:
            return None

        try:

            if pd.isna(value):
                return None

            number = float(value)

            if not np.isfinite(number):
                return None

            return number

        except (
            ValueError,
            TypeError,
            OverflowError,
        ):
            return None

    # -------------------------------------------------------------------------

    @classmethod
    def _numeric_mean(
        cls,
        df: pd.DataFrame,
        column: str,
    ) -> Optional[float]:
        """
        Calculate the mean of a numeric column safely.
        """

        if column not in df.columns:
            return None

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if series.empty:
            return None

        return float(series.mean())

    # -------------------------------------------------------------------------

    @classmethod
    def _numeric_mean_or_sum(
        cls,
        df: pd.DataFrame,
        column: str,
        operation: str = "mean",
    ) -> Optional[float]:
        """
        Calculate either the mean or sum of a numeric column safely.
        """

        if column not in df.columns:
            return None

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if series.empty:
            return None

        if operation == "sum":
            return float(series.sum())

        return float(series.mean())

    # =========================================================================
    # DISPLAY HELPERS
    # =========================================================================

    @classmethod
    def _display_text(cls, value: Any) -> str:
        """
        Convert any value to safe display text.
        """

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

        return text

    # -------------------------------------------------------------------------

    @classmethod
    def _format_number(
        cls,
        value: Any,
        decimals: int = 2,
    ) -> str:
        """
        Format a numeric value for display.
        """

        number = cls._safe_float(value)

        if number is None:
            return "N/A"

        return f"{number:,.{decimals}f}"

    # -------------------------------------------------------------------------

    @classmethod
    def _format_with_unit(
        cls,
        value: Any,
        unit: str,
        decimals: int = 2,
    ) -> str:
        """
        Format a numeric value and append a unit.
        """

        number = cls._safe_float(value)

        if number is None:
            return "N/A"

        return f"{number:,.{decimals}f} {unit}"

    # =========================================================================
    # OPTIONAL PROGRAMMATIC API
    # =========================================================================

    def get_current_data(self) -> pd.DataFrame:
        """
        Return the currently filtered grinding DataFrame.

        This can be useful to other parts of the application.
        """

        return self._get_filtered_dataframe().copy()

    # -------------------------------------------------------------------------

    def get_selected_record(self) -> Optional[pd.Series]:
        """
        Return the currently selected grinding record.

        Returns
        -------
        pandas.Series or None
            Selected record, if one exists.
        """

        selection = self.tree.selection()

        if not selection:
            return None

        try:
            index = int(selection[0])
        except ValueError:
            return None

        filtered = self._get_filtered_dataframe()

        if index not in filtered.index:
            return None

        return filtered.loc[index].copy()


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTION
# =============================================================================

def create_grinding_view(parent, controller):
    """
    Convenience function for creating a GrindingView.

    This does not replace the class. It simply provides an optional
    programmatic entry point.
    """

    return GrindingView(
        parent,
        controller,
    )


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "GrindingView",
    "create_grinding_view",
]