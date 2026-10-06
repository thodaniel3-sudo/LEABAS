"""
LEBAS - Reports View
File: ui/reports_view.py

Purpose:
    Provides the report-generation interface for LEBAS.

Features:
    - Dataset summary
    - Experimental run summary
    - Beneficiation summary
    - Material balance summary
    - Sieve analysis summary
    - Generate text/HTML-style report through report_generator when available
    - Export results through export_service when available
    - Safe fallback report generation if service functions differ
    - No hard-coded experimental values
    - Prevents double-counting repeated sieve rows
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from config import Config
from services.data_loader import load_dataset

logger = logging.getLogger(__name__)


class ReportsView:
    """Report generation and export interface."""

    def __init__(self, parent, controller):
        self.controller = controller
        self.frame = ttk.Frame(parent)

        self.df = pd.DataFrame()
        self.process_df = pd.DataFrame()

        self.status_var = tk.StringVar(
            value="Ready"
        )

        self.report_path_var = tk.StringVar(
            value="No report generated"
        )

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
        """Reload the dataset and update the report summary."""

        try:
            self.df = load_dataset()

            if self.df is None:
                self.df = pd.DataFrame()

            self.process_df = (
                self._prepare_process_dataframe(
                    self.df
                )
            )

            self._update_summary()

            if self.df.empty:
                self.status_var.set(
                    "No dataset loaded"
                )
            else:
                self.status_var.set(
                    f"{len(self.df):,} dataset row(s) ready for reporting"
                )

        except Exception as exc:
            logger.exception(
                "Failed to refresh reports view"
            )

            self.status_var.set(
                "Unable to load report data"
            )

    # ==================================================================
    # STYLE
    # ==================================================================

    def _configure_style(self):
        """Configure ttk styles."""

        style = ttk.Style()

        try:
            style.configure(
                "ReportsTitle.TLabel",
                font=("Segoe UI", 20, "bold"),
            )

            style.configure(
                "ReportsSubtitle.TLabel",
                font=("Segoe UI", 10),
            )

            style.configure(
                "ReportsSection.TLabel",
                font=("Segoe UI", 11, "bold"),
            )

            style.configure(
                "ReportsValue.TLabel",
                font=("Segoe UI", 10),
            )

        except Exception:
            logger.exception(
                "Could not configure report styles"
            )

    # ==================================================================
    # UI
    # ==================================================================

    def _build_ui(self):
        """Build the reports interface."""

        self.frame.grid_rowconfigure(
            2,
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
            text="Reports & Export",
            style="ReportsTitle.TLabel",
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        ttk.Label(
            header,
            text=(
                "Generate laboratory reports, review dataset summaries "
                "and export analysis results"
            ),
            style="ReportsSubtitle.TLabel",
        ).grid(
            row=1,
            column=0,
            sticky="w",
            pady=(3, 0),
        )

        # --------------------------------------------------------------
        # ACTION BAR
        # --------------------------------------------------------------

        actions = ttk.Frame(
            self.frame
        )

        actions.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=20,
            pady=(0, 12),
        )

        ttk.Button(
            actions,
            text="↻ Refresh Data",
            command=self.refresh,
        ).pack(
            side="left",
            padx=(0, 8),
        )

        ttk.Button(
            actions,
            text="Generate Report",
            command=self.generate_report,
        ).pack(
            side="left",
            padx=4,
        )

        ttk.Button(
            actions,
            text="Export CSV",
            command=self.export_csv,
        ).pack(
            side="left",
            padx=4,
        )

        ttk.Button(
            actions,
            text="Export Excel",
            command=self.export_excel,
        ).pack(
            side="left",
            padx=4,
        )

        ttk.Button(
            actions,
            text="Save Summary",
            command=self.save_summary,
        ).pack(
            side="left",
            padx=4,
        )

        # --------------------------------------------------------------
        # MAIN AREA
        # --------------------------------------------------------------

        main = ttk.Frame(
            self.frame
        )

        main.grid(
            row=2,
            column=0,
            sticky="nsew",
            padx=20,
            pady=(0, 10),
        )

        main.grid_rowconfigure(
            0,
            weight=1,
        )

        main.grid_columnconfigure(
            0,
            weight=1,
        )

        # --------------------------------------------------------------
        # NOTEBOOK
        # --------------------------------------------------------------

        notebook = ttk.Notebook(
            main
        )

        notebook.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        self.summary_tab = ttk.Frame(
            notebook,
            padding=15,
        )

        self.results_tab = ttk.Frame(
            notebook,
            padding=15,
        )

        self.report_tab = ttk.Frame(
            notebook,
            padding=15,
        )

        notebook.add(
            self.summary_tab,
            text="Dataset Summary",
        )

        notebook.add(
            self.results_tab,
            text="Analysis Summary",
        )

        notebook.add(
            self.report_tab,
            text="Generated Report",
        )

        self._build_summary_tab()
        self._build_results_tab()
        self._build_report_tab()

        # --------------------------------------------------------------
        # STATUS
        # --------------------------------------------------------------

        status = ttk.Frame(
            self.frame
        )

        status.grid(
            row=3,
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

        ttk.Label(
            status,
            textvariable=self.report_path_var,
        ).pack(
            side="right"
        )

    # ==================================================================
    # SUMMARY TAB
    # ==================================================================

    def _build_summary_tab(self):
        """Build dataset summary tab."""

        self.summary_tab.grid_rowconfigure(
            1,
            weight=1,
        )

        self.summary_tab.grid_columnconfigure(
            0,
            weight=1,
        )

        ttk.Label(
            self.summary_tab,
            text="Dataset Overview",
            style="ReportsSection.TLabel",
        ).grid(
            row=0,
            column=0,
            sticky="w",
            pady=(0, 10),
        )

        container = ttk.Frame(
            self.summary_tab
        )

        container.grid(
            row=1,
            column=0,
            sticky="nsew",
        )

        container.grid_rowconfigure(
            0,
            weight=1,
        )

        container.grid_columnconfigure(
            0,
            weight=1,
        )

        self.summary_text = tk.Text(
            container,
            wrap="word",
            font=("Consolas", 10),
            padx=12,
            pady=12,
        )

        summary_scroll = ttk.Scrollbar(
            container,
            orient="vertical",
            command=self.summary_text.yview,
        )

        self.summary_text.configure(
            yscrollcommand=summary_scroll.set
        )

        self.summary_text.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        summary_scroll.grid(
            row=0,
            column=1,
            sticky="ns",
        )

    # ==================================================================
    # RESULTS TAB
    # ==================================================================

    def _build_results_tab(self):
        """Build analysis summary table."""

        self.results_tab.grid_rowconfigure(
            1,
            weight=1,
        )

        self.results_tab.grid_columnconfigure(
            0,
            weight=1,
        )

        ttk.Label(
            self.results_tab,
            text="Process Performance Summary",
            style="ReportsSection.TLabel",
        ).grid(
            row=0,
            column=0,
            sticky="w",
            pady=(0, 10),
        )

        container = ttk.Frame(
            self.results_tab
        )

        container.grid(
            row=1,
            column=0,
            sticky="nsew",
        )

        container.grid_rowconfigure(
            0,
            weight=1,
        )

        container.grid_columnconfigure(
            0,
            weight=1,
        )

        columns = (
            "metric",
            "value",
        )

        self.results_tree = ttk.Treeview(
            container,
            columns=columns,
            show="headings",
            height=18,
        )

        self.results_tree.heading(
            "metric",
            text="Metric",
        )

        self.results_tree.heading(
            "value",
            text="Value",
        )

        self.results_tree.column(
            "metric",
            width=320,
            anchor="w",
        )

        self.results_tree.column(
            "value",
            width=250,
            anchor="center",
        )

        scrollbar = ttk.Scrollbar(
            container,
            orient="vertical",
            command=self.results_tree.yview,
        )

        self.results_tree.configure(
            yscrollcommand=scrollbar.set
        )

        self.results_tree.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        scrollbar.grid(
            row=0,
            column=1,
            sticky="ns",
        )

    # ==================================================================
    # REPORT TAB
    # ==================================================================

    def _build_report_tab(self):
        """Build generated report preview."""

        self.report_tab.grid_rowconfigure(
            1,
            weight=1,
        )

        self.report_tab.grid_columnconfigure(
            0,
            weight=1,
        )

        ttk.Label(
            self.report_tab,
            text="Report Preview",
            style="ReportsSection.TLabel",
        ).grid(
            row=0,
            column=0,
            sticky="w",
            pady=(0, 10),
        )

        container = ttk.Frame(
            self.report_tab
        )

        container.grid(
            row=1,
            column=0,
            sticky="nsew",
        )

        container.grid_rowconfigure(
            0,
            weight=1,
        )

        container.grid_columnconfigure(
            0,
            weight=1,
        )

        self.report_text = tk.Text(
            container,
            wrap="word",
            font=("Consolas", 10),
            padx=12,
            pady=12,
        )

        scrollbar = ttk.Scrollbar(
            container,
            orient="vertical",
            command=self.report_text.yview,
        )

        self.report_text.configure(
            yscrollcommand=scrollbar.set
        )

        self.report_text.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        scrollbar.grid(
            row=0,
            column=1,
            sticky="ns",
        )

    # ==================================================================
    # DATA PREPARATION
    # ==================================================================

    def _prepare_process_dataframe(
        self,
        df,
    ):
        """
        Create one process-level record per sample/replicate.

        Sieve rows can repeat the same feed, beneficiation and
        material-balance information. Therefore these process-level
        values must not be summed across raw CSV rows.
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

        relevant = [
            "sample_id",
            "replicate",
            "material_type",
            "experiment_date",
            "operator",
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

        existing = [
            column
            for column in relevant
            if column in working.columns
        ]

        if not existing:
            return pd.DataFrame()

        working["_completeness"] = (
            working[existing]
            .notna()
            .sum(axis=1)
        )

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
            .drop(
                columns=["_completeness"],
                errors="ignore",
            )
            .reset_index(drop=True)
        )

        return self._calculate_metrics(
            working
        )

    def _calculate_metrics(
        self,
        df,
    ):
        """Calculate reporting metrics."""

        result = df.copy()

        numeric_columns = [
            "feed_mass",
            "feed_grade",
            "feed_moisture",
            "crush_time",
            "crush_power",
            "crush_feed_size",
            "crush_product_size",
            "grinding_time",
            "grinding_media_mass",
            "grinding_speed",
            "grinding_feed_size",
            "grinding_product_size",
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
            "solid_recovery",
        ]

        for column in numeric_columns:

            if column in result.columns:

                result[column] = pd.to_numeric(
                    result[column],
                    errors="coerce",
                )

        result["mass_recovery"] = np.nan
        result["yield"] = np.nan
        result["grade_recovery"] = np.nan
        result["enrichment_ratio"] = np.nan
        result["mass_balance"] = np.nan
        result["balance_error_percent"] = np.nan

        if all(
            column in result.columns
            for column in [
                "feed_mass",
                "concentrate_mass",
            ]
        ):

            valid = (
                result["feed_mass"].notna()
                & result["concentrate_mass"].notna()
                & (result["feed_mass"] > 0)
            )

            result.loc[
                valid,
                "mass_recovery"
            ] = (
                result.loc[
                    valid,
                    "concentrate_mass"
                ]
                / result.loc[
                    valid,
                    "feed_mass"
                ]
            ) * 100

            result.loc[
                valid,
                "yield"
            ] = result.loc[
                valid,
                "mass_recovery"
            ]

        if all(
            column in result.columns
            for column in [
                "feed_mass",
                "feed_grade",
                "concentrate_mass",
                "concentrate_grade",
            ]
        ):

            denominator = (
                result["feed_mass"]
                * result["feed_grade"]
            )

            valid = (
                denominator.notna()
                & (denominator > 0)
                & result["concentrate_mass"].notna()
                & result["concentrate_grade"].notna()
            )

            result.loc[
                valid,
                "grade_recovery"
            ] = (
                (
                    result.loc[
                        valid,
                        "concentrate_mass"
                    ]
                    * result.loc[
                        valid,
                        "concentrate_grade"
                    ]
                )
                / denominator[valid]
            ) * 100

        if all(
            column in result.columns
            for column in [
                "feed_grade",
                "concentrate_grade",
            ]
        ):

            valid = (
                result["feed_grade"].notna()
                & result["concentrate_grade"].notna()
                & (result["feed_grade"] > 0)
            )

            result.loc[
                valid,
                "enrichment_ratio"
            ] = (
                result.loc[
                    valid,
                    "concentrate_grade"
                ]
                / result.loc[
                    valid,
                    "feed_grade"
                ]
            )

        if all(
            column in result.columns
            for column in [
                "feed_mass",
                "concentrate_mass",
                "tailings_mass",
            ]
        ):

            losses = (
                result["losses_mass"].fillna(0)
                if "losses_mass" in result.columns
                else pd.Series(
                    0.0,
                    index=result.index,
                )
            )

            output_mass = (
                result["concentrate_mass"]
                + result["tailings_mass"]
                + losses
            )

            valid = (
                result["feed_mass"].notna()
                & result["concentrate_mass"].notna()
                & result["tailings_mass"].notna()
                & (result["feed_mass"] > 0)
            )

            result.loc[
                valid,
                "mass_balance"
            ] = (
                result.loc[
                    valid,
                    "feed_mass"
                ]
                - output_mass[valid]
            )

            result.loc[
                valid,
                "balance_error_percent"
            ] = (
                (
                    result.loc[
                        valid,
                        "mass_balance"
                    ].abs()
                )
                / result.loc[
                    valid,
                    "feed_mass"
                ]
            ) * 100

        return result

    # ==================================================================
    # SUMMARY
    # ==================================================================

    def _update_summary(self):
        """Update the dataset summary and analysis summary."""

        self._update_dataset_summary()
        self._update_analysis_summary()

    def _update_dataset_summary(self):
        """Update dataset overview text."""

        self.summary_text.delete(
            "1.0",
            tk.END,
        )

        if self.df.empty:
            self.summary_text.insert(
                tk.END,
                "No dataset is currently loaded.\n\n"
                "Place dataset.csv in the project root and refresh "
                "the application.",
            )
            return

        df = self.df

        lines = []

        lines.append(
            "LEBAS DATASET REPORT"
        )

        lines.append(
            "=" * 70
        )

        lines.append("")

        lines.append(
            f"Generated: "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        lines.append(
            f"Dataset rows: {len(df):,}"
        )

        lines.append(
            f"Dataset columns: {len(df.columns):,}"
        )

        if "sample_id" in df.columns:

            lines.append(
                f"Unique samples: "
                f"{df['sample_id'].nunique():,}"
            )

        if (
            "sample_id" in df.columns
            and "replicate" in df.columns
        ):

            runs = (
                df[
                    [
                        "sample_id",
                        "replicate",
                    ]
                ]
                .drop_duplicates()
                .shape[0]
            )

            lines.append(
                f"Experimental runs: {runs:,}"
            )

        if "material_type" in df.columns:

            materials = (
                df["material_type"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if materials:

                lines.append(
                    "Material types: "
                    + ", ".join(
                        sorted(materials)
                    )
                )

        if "experiment_date" in df.columns:

            dates = pd.to_datetime(
                df["experiment_date"],
                errors="coerce",
            ).dropna()

            if not dates.empty:

                lines.append(
                    f"Experiment date range: "
                    f"{dates.min().strftime('%Y-%m-%d')} "
                    f"to "
                    f"{dates.max().strftime('%Y-%m-%d')}"
                )

        lines.append("")

        lines.append(
            "COLUMN INFORMATION"
        )

        lines.append(
            "-" * 70
        )

        for column in df.columns:

            non_null = int(
                df[column].notna().sum()
            )

            lines.append(
                f"{column}: "
                f"{non_null:,} non-empty value(s)"
            )

        lines.append("")

        lines.append(
            "SIEVE DATA"
        )

        lines.append(
            "-" * 70
        )

        if "sieve_size" in df.columns:

            sieve = pd.to_numeric(
                df["sieve_size"],
                errors="coerce",
            ).dropna()

            lines.append(
                f"Sieve records: {len(sieve):,}"
            )

            if not sieve.empty:

                lines.append(
                    f"Smallest sieve size: "
                    f"{sieve.min():,.3f}"
                )

                lines.append(
                    f"Largest sieve size: "
                    f"{sieve.max():,.3f}"
                )

                lines.append(
                    "Note: sieve sizes are read dynamically "
                    "from dataset.csv."
                )

        else:

            lines.append(
                "No sieve_size column found."
            )

        self.summary_text.insert(
            tk.END,
            "\n".join(lines),
        )

    # ==================================================================
    # ANALYSIS SUMMARY
    # ==================================================================

    def _update_analysis_summary(self):
        """Update process-performance summary."""

        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        if self.process_df.empty:
            return

        df = self.process_df

        metrics = []

        metrics.append(
            (
                "Experimental runs",
                self._count_runs(df),
            )
        )

        metrics.append(
            (
                "Average feed mass",
                self._mean(
                    df,
                    "feed_mass",
                ),
            )
        )

        metrics.append(
            (
                "Total feed mass",
                self._sum(
                    df,
                    "feed_mass",
                ),
            )
        )

        metrics.append(
            (
                "Average feed grade",
                self._mean(
                    df,
                    "feed_grade",
                ),
            )
        )

        metrics.append(
            (
                "Average concentrate mass",
                self._mean(
                    df,
                    "concentrate_mass",
                ),
            )
        )

        metrics.append(
            (
                "Total concentrate mass",
                self._sum(
                    df,
                    "concentrate_mass",
                ),
            )
        )

        metrics.append(
            (
                "Average concentrate grade",
                self._mean(
                    df,
                    "concentrate_grade",
                ),
            )
        )

        metrics.append(
            (
                "Average tailings mass",
                self._mean(
                    df,
                    "tailings_mass",
                ),
            )
        )

        metrics.append(
            (
                "Average mass recovery",
                self._mean(
                    df,
                    "mass_recovery",
                    "%",
                ),
            )
        )

        metrics.append(
            (
                "Average yield",
                self._mean(
                    df,
                    "yield",
                    "%",
                ),
            )
        )

        metrics.append(
            (
                "Average grade recovery",
                self._mean(
                    df,
                    "grade_recovery",
                    "%",
                ),
            )
        )

        metrics.append(
            (
                "Average enrichment ratio",
                self._mean(
                    df,
                    "enrichment_ratio",
                ),
            )
        )

        metrics.append(
            (
                "Average mass-balance difference",
                self._mean(
                    df,
                    "mass_balance",
                ),
            )
        )

        metrics.append(
            (
                "Average balance error",
                self._mean(
                    df,
                    "balance_error_percent",
                    "%",
                ),
            )
        )

        if "solid_recovery" in df.columns:

            metrics.append(
                (
                    "Average recorded solid recovery",
                    self._mean(
                        df,
                        "solid_recovery",
                        "%",
                    ),
                )
            )

        if "crush_time" in df.columns:

            metrics.append(
                (
                    "Average crushing time",
                    self._mean(
                        df,
                        "crush_time",
                    ),
                )
            )

        if "grinding_time" in df.columns:

            metrics.append(
                (
                    "Average grinding time",
                    self._mean(
                        df,
                        "grinding_time",
                    ),
                )
            )

        for metric, value in metrics:

            self.results_tree.insert(
                "",
                "end",
                values=(
                    metric,
                    value,
                ),
            )

    # ==================================================================
    # REPORT GENERATION
    # ==================================================================

    def generate_report(self):
        """
        Generate a report.

        The application first attempts to use the project's
        report_generator service. If the service API is not yet
        available, a complete local text report is generated instead.
        """

        if self.df.empty:

            messagebox.showwarning(
                "No Data",
                "There is no dataset available to generate a report.",
            )

            return

        try:

            report_content = (
                self._build_report_content()
            )

            generated_path = (
                self._try_report_service(
                    report_content
                )
            )

            if generated_path:

                self.report_path_var.set(
                    f"Report: {generated_path}"
                )

                self.status_var.set(
                    "Report generated successfully"
                )

            else:

                path = self._save_default_report(
                    report_content
                )

                self.report_path_var.set(
                    f"Report: {path}"
                )

                self.status_var.set(
                    "Report generated successfully"
                )

            self.report_text.delete(
                "1.0",
                tk.END,
            )

            self.report_text.insert(
                tk.END,
                report_content,
            )

            messagebox.showinfo(
                "Report Generated",
                "The laboratory report was generated successfully.",
            )

        except Exception as exc:

            logger.exception(
                "Report generation failed"
            )

            messagebox.showerror(
                "Report Error",
                f"Unable to generate the report.\n\n{exc}",
            )

    def _build_report_content(self):
        """Build a complete plain-text laboratory report."""

        df = self.process_df

        lines = []

        app_name = getattr(
            Config,
            "APP_NAME",
            "LEBAS",
        )

        lines.append(
            app_name.upper()
        )

        lines.append(
            "LABORATORY EXTRACTION AND BENEFICIATION ANALYSIS REPORT"
        )

        lines.append(
            "=" * 78
        )

        lines.append("")

        lines.append(
            f"Report generated: "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        lines.append(
            f"Dataset rows: {len(self.df):,}"
        )

        lines.append(
            f"Experimental runs: "
            f"{self._count_runs(df)}"
        )

        lines.append("")

        # --------------------------------------------------------------
        # DATASET
        # --------------------------------------------------------------

        lines.append(
            "1. DATASET SUMMARY"
        )

        lines.append(
            "-" * 78
        )

        if "sample_id" in self.df.columns:

            lines.append(
                f"Unique samples: "
                f"{self.df['sample_id'].nunique():,}"
            )

        if "material_type" in self.df.columns:

            materials = (
                self.df["material_type"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            lines.append(
                "Material types: "
                + (
                    ", ".join(
                        sorted(materials)
                    )
                    if materials
                    else "N/A"
                )
            )

        if "experiment_date" in self.df.columns:

            dates = pd.to_datetime(
                self.df["experiment_date"],
                errors="coerce",
            ).dropna()

            if not dates.empty:

                lines.append(
                    f"Experiment period: "
                    f"{dates.min().strftime('%Y-%m-%d')} "
                    f"to "
                    f"{dates.max().strftime('%Y-%m-%d')}"
                )

        lines.append("")

        # --------------------------------------------------------------
        # CRUSHING
        # --------------------------------------------------------------

        lines.append(
            "2. CRUSHING SUMMARY"
        )

        lines.append(
            "-" * 78
        )

        lines.append(
            f"Average crushing time: "
            f"{self._mean(df, 'crush_time')}"
        )

        lines.append(
            f"Average crushing power: "
            f"{self._mean(df, 'crush_power')}"
        )

        lines.append(
            f"Average crusher feed size: "
            f"{self._mean(df, 'crush_feed_size')}"
        )

        lines.append(
            f"Average crusher product size: "
            f"{self._mean(df, 'crush_product_size')}"
        )

        lines.append("")

        # --------------------------------------------------------------
        # GRINDING
        # --------------------------------------------------------------

        lines.append(
            "3. GRINDING SUMMARY"
        )

        lines.append(
            "-" * 78
        )

        lines.append(
            f"Average grinding time: "
            f"{self._mean(df, 'grinding_time')}"
        )

        lines.append(
            f"Average grinding speed: "
            f"{self._mean(df, 'grinding_speed')}"
        )

        lines.append(
            f"Average grinding media mass: "
            f"{self._mean(df, 'grinding_media_mass')}"
        )

        lines.append(
            f"Average grinding feed size: "
            f"{self._mean(df, 'grinding_feed_size')}"
        )

        lines.append(
            f"Average grinding product size: "
            f"{self._mean(df, 'grinding_product_size')}"
        )

        lines.append("")

        # --------------------------------------------------------------
        # SIEVE
        # --------------------------------------------------------------

        lines.append(
            "4. SIEVE ANALYSIS SUMMARY"
        )

        lines.append(
            "-" * 78
        )

        if "sieve_size" in self.df.columns:

            sieve = pd.to_numeric(
                self.df["sieve_size"],
                errors="coerce",
            ).dropna()

            lines.append(
                f"Sieve records: {len(sieve):,}"
            )

            if not sieve.empty:

                lines.append(
                    f"Smallest sieve size: "
                    f"{sieve.min():,.3f}"
                )

                lines.append(
                    f"Largest sieve size: "
                    f"{sieve.max():,.3f}"
                )

        else:

            lines.append(
                "No sieve analysis data available."
            )

        lines.append("")

        # --------------------------------------------------------------
        # BENEFICIATION
        # --------------------------------------------------------------

        lines.append(
            "5. BENEFICIATION SUMMARY"
        )

        lines.append(
            "-" * 78
        )

        lines.append(
            f"Average feed grade: "
            f"{self._mean(df, 'feed_grade')}"
        )

        lines.append(
            f"Average concentrate grade: "
            f"{self._mean(df, 'concentrate_grade')}"
        )

        lines.append(
            f"Average mass recovery: "
            f"{self._mean(df, 'mass_recovery', '%')}"
        )

        lines.append(
            f"Average yield: "
            f"{self._mean(df, 'yield', '%')}"
        )

        lines.append(
            f"Average grade recovery: "
            f"{self._mean(df, 'grade_recovery', '%')}"
        )

        lines.append(
            f"Average enrichment ratio: "
            f"{self._mean(df, 'enrichment_ratio')}"
        )

        lines.append("")

        # --------------------------------------------------------------
        # MATERIAL BALANCE
        # --------------------------------------------------------------

        lines.append(
            "6. MATERIAL BALANCE SUMMARY"
        )

        lines.append(
            "-" * 78
        )

        lines.append(
            f"Total feed mass: "
            f"{self._sum(df, 'feed_mass')}"
        )

        lines.append(
            f"Total concentrate mass: "
            f"{self._sum(df, 'concentrate_mass')}"
        )

        lines.append(
            f"Total tailings mass: "
            f"{self._sum(df, 'tailings_mass')}"
        )

        lines.append(
            f"Total losses mass: "
            f"{self._sum(df, 'losses_mass')}"
        )

        lines.append(
            f"Average balance difference: "
            f"{self._mean(df, 'mass_balance')}"
        )

        lines.append(
            f"Average balance error: "
            f"{self._mean(df, 'balance_error_percent', '%')}"
        )

        lines.append("")

        # --------------------------------------------------------------
        # DATA QUALITY
        # --------------------------------------------------------------

        lines.append(
            "7. DATA QUALITY"
        )

        lines.append(
            "-" * 78
        )

        missing = int(
            self.df.isna().sum().sum()
        )

        total_cells = (
            self.df.shape[0]
            * self.df.shape[1]
        )

        lines.append(
            f"Missing cells: {missing:,}"
        )

        if total_cells > 0:

            missing_percent = (
                missing
                / total_cells
                * 100
            )

            lines.append(
                f"Missing-data percentage: "
                f"{missing_percent:.2f}%"
            )

        lines.append("")

        # --------------------------------------------------------------
        # NOTES
        # --------------------------------------------------------------

        lines.append(
            "8. REPORTING NOTES"
        )

        lines.append(
            "-" * 78
        )

        lines.append(
            "Process-level values are aggregated using one record "
            "per sample and replicate to prevent repeated sieve "
            "rows from artificially inflating feed, concentrate, "
            "tailings or beneficiation totals."
        )

        lines.append(
            "Sieve records are retained individually because each "
            "sieve fraction represents a separate particle-size "
            "measurement."
        )

        lines.append(
            "Sieve sizes are read dynamically from the dataset and "
            "are not restricted to a predefined list."
        )

        lines.append("")

        lines.append(
            "END OF REPORT"
        )

        return "\n".join(lines)

    # ==================================================================
    # REPORT SERVICE
    # ==================================================================

    def _try_report_service(
        self,
        report_content,
    ):
        """
        Attempt to use services/report_generator.py.

        Because the report service may be implemented with a different
        API while the project is being assembled, this method checks
        several common function names safely.
        """

        try:

            from services import report_generator

        except Exception:

            return None

        candidates = [
            "generate_report",
            "create_report",
            "save_report",
            "generate_text_report",
        ]

        reports_dir = self._get_reports_directory()

        filename = (
            "LEBAS_Report_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            + ".txt"
        )

        output_path = (
            reports_dir
            / filename
        )

        for function_name in candidates:

            function = getattr(
                report_generator,
                function_name,
                None,
            )

            if not callable(function):
                continue

            attempts = [
                lambda: function(
                    self.df,
                    output_path,
                ),
                lambda: function(
                    self.df,
                    str(output_path),
                ),
                lambda: function(
                    report_content,
                    output_path,
                ),
                lambda: function(
                    report_content,
                    str(output_path),
                ),
                lambda: function(
                    self.df,
                ),
            ]

            for attempt in attempts:

                try:

                    result = attempt()

                    if result is not None:

                        if isinstance(
                            result,
                            (str, Path),
                        ):

                            return str(result)

                        if output_path.exists():

                            return str(
                                output_path
                            )

                except TypeError:
                    continue

                except Exception:
                    logger.exception(
                        "Report service function failed: %s",
                        function_name,
                    )

        return None

    # ==================================================================
    # EXPORT
    # ==================================================================

    def export_csv(self):
        """Export the process-level analysis results to CSV."""

        if self.process_df.empty:

            messagebox.showwarning(
                "No Data",
                "There are no analysis results to export.",
            )

            return

        default_name = (
            "LEBAS_results_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            + ".csv"
        )

        path = filedialog.asksaveasfilename(
            title="Export Results as CSV",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[
                (
                    "CSV files",
                    "*.csv",
                ),
                (
                    "All files",
                    "*.*",
                ),
            ],
        )

        if not path:
            return

        try:

            self.process_df.to_csv(
                path,
                index=False,
            )

            self.status_var.set(
                "CSV export completed"
            )

            messagebox.showinfo(
                "Export Complete",
                f"Results exported successfully.\n\n{path}",
            )

        except Exception as exc:

            logger.exception(
                "CSV export failed"
            )

            messagebox.showerror(
                "Export Error",
                f"Unable to export CSV.\n\n{exc}",
            )

    def export_excel(self):
        """Export process-level results to Excel."""

        if self.process_df.empty:

            messagebox.showwarning(
                "No Data",
                "There are no analysis results to export.",
            )

            return

        default_name = (
            "LEBAS_results_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            + ".xlsx"
        )

        path = filedialog.asksaveasfilename(
            title="Export Results as Excel",
            defaultextension=".xlsx",
            initialfile=default_name,
            filetypes=[
                (
                    "Excel files",
                    "*.xlsx",
                ),
                (
                    "All files",
                    "*.*",
                ),
            ],
        )

        if not path:
            return

        try:

            with pd.ExcelWriter(
                path,
                engine="openpyxl",
            ) as writer:

                self.process_df.to_excel(
                    writer,
                    sheet_name="Results",
                    index=False,
                )

                if not self.df.empty:

                    self.df.to_excel(
                        writer,
                        sheet_name="Raw Dataset",
                        index=False,
                    )

                summary = self._build_summary_dataframe()

                summary.to_excel(
                    writer,
                    sheet_name="Summary",
                    index=False,
                )

            self.status_var.set(
                "Excel export completed"
            )

            messagebox.showinfo(
                "Export Complete",
                f"Excel workbook exported successfully.\n\n{path}",
            )

        except ImportError:

            messagebox.showerror(
                "Excel Export Error",
                (
                    "The openpyxl package is required for Excel export.\n\n"
                    "Install project requirements and try again."
                ),
            )

        except Exception as exc:

            logger.exception(
                "Excel export failed"
            )

            messagebox.showerror(
                "Export Error",
                f"Unable to export Excel workbook.\n\n{exc}",
            )

    def save_summary(self):
        """Save the generated report text manually."""

        if self.df.empty:

            messagebox.showwarning(
                "No Data",
                "There is no dataset available.",
            )

            return

        content = (
            self._build_report_content()
        )

        default_name = (
            "LEBAS_Report_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            + ".txt"
        )

        path = filedialog.asksaveasfilename(
            title="Save LEBAS Report",
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[
                (
                    "Text files",
                    "*.txt",
                ),
                (
                    "All files",
                    "*.*",
                ),
            ],
        )

        if not path:
            return

        try:

            Path(path).write_text(
                content,
                encoding="utf-8",
            )

            self.report_path_var.set(
                f"Report: {path}"
            )

            self.status_var.set(
                "Report saved successfully"
            )

            messagebox.showinfo(
                "Report Saved",
                f"Report saved successfully.\n\n{path}",
            )

        except Exception as exc:

            logger.exception(
                "Saving report failed"
            )

            messagebox.showerror(
                "Save Error",
                f"Unable to save report.\n\n{exc}",
            )

    # ==================================================================
    # DEFAULT REPORT LOCATION
    # ==================================================================

    def _get_reports_directory(self):
        """Return/create the project reports directory."""

        configured = getattr(
            Config,
            "REPORTS_DIR",
            None,
        )

        if configured:

            directory = Path(
                configured
            )

        else:

            directory = (
                Path(
                    getattr(
                        Config,
                        "BASE_DIR",
                        Path(__file__).resolve().parent.parent,
                    )
                )
                / "reports"
            )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return directory

    def _save_default_report(
        self,
        content,
    ):
        """Save a generated report to the reports directory."""

        reports_dir = (
            self._get_reports_directory()
        )

        filename = (
            "LEBAS_Report_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            + ".txt"
        )

        path = (
            reports_dir
            / filename
        )

        path.write_text(
            content,
            encoding="utf-8",
        )

        return str(path)

    # ==================================================================
    # SUMMARY DATAFRAME
    # ==================================================================

    def _build_summary_dataframe(self):
        """Create a compact summary DataFrame."""

        df = self.process_df

        rows = [
            (
                "Experimental runs",
                self._count_runs(df),
            ),
            (
                "Average feed mass",
                self._mean(df, "feed_mass"),
            ),
            (
                "Total feed mass",
                self._sum(df, "feed_mass"),
            ),
            (
                "Average feed grade",
                self._mean(df, "feed_grade"),
            ),
            (
                "Average concentrate mass",
                self._mean(
                    df,
                    "concentrate_mass",
                ),
            ),
            (
                "Total concentrate mass",
                self._sum(
                    df,
                    "concentrate_mass",
                ),
            ),
            (
                "Average concentrate grade",
                self._mean(
                    df,
                    "concentrate_grade",
                ),
            ),
            (
                "Average recovery",
                self._mean(
                    df,
                    "mass_recovery",
                    "%",
                ),
            ),
            (
                "Average yield",
                self._mean(
                    df,
                    "yield",
                    "%",
                ),
            ),
            (
                "Average grade recovery",
                self._mean(
                    df,
                    "grade_recovery",
                    "%",
                ),
            ),
            (
                "Average enrichment ratio",
                self._mean(
                    df,
                    "enrichment_ratio",
                ),
            ),
            (
                "Average balance error",
                self._mean(
                    df,
                    "balance_error_percent",
                    "%",
                ),
            ),
        ]

        return pd.DataFrame(
            rows,
            columns=[
                "Metric",
                "Value",
            ],
        )

    # ==================================================================
    # NUMERIC HELPERS
    # ==================================================================

    @staticmethod
    def _mean(
        dataframe,
        column,
        suffix="",
    ):
        """Safely calculate an average."""

        if (
            dataframe is None
            or dataframe.empty
            or column not in dataframe.columns
        ):
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

    @staticmethod
    def _sum(
        dataframe,
        column,
    ):
        """Safely calculate a sum."""

        if (
            dataframe is None
            or dataframe.empty
            or column not in dataframe.columns
        ):
            return "N/A"

        values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            return "N/A"

        value = values.sum()

        if not np.isfinite(value):
            return "N/A"

        return f"{value:,.3f}"

    @staticmethod
    def _count_runs(
        dataframe,
    ):
        """Count unique sample/replicate combinations."""

        if dataframe is None or dataframe.empty:
            return 0

        if (
            "sample_id" not in dataframe.columns
            or "replicate" not in dataframe.columns
        ):
            return len(dataframe)

        return (
            dataframe[
                [
                    "sample_id",
                    "replicate",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        )


__all__ = ["ReportsView"]