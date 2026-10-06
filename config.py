"""
LEBAS Configuration Module

This module contains all application-wide configuration settings.

All paths are based on the directory containing this config.py file.
This ensures that dataset.csv is correctly located inside the LEBAS
project folder regardless of the folder from which the application
is launched.
"""

from pathlib import Path
from typing import Dict, List, Tuple


class Config:
    """
    Application-wide configuration settings.
    """

    # =====================================================================
    # APPLICATION INFORMATION
    # =====================================================================

    APP_NAME = "LEBAS"

    APP_VERSION = "1.0.0"

    APP_DESCRIPTION = (
        "Laboratory Extraction and Beneficiation Analysis System"
    )

    # =====================================================================
    # PROJECT PATHS
    # =====================================================================

    # config.py is directly inside the LEBAS project folder.
    #
    # Therefore:
    #
    # Path(__file__).resolve().parent
    #
    # correctly points to:
    #
    # C:\...\blessing best program\LEBAS
    #
    # The previous code used parent.parent, which moved one directory
    # too far upward.
    PROJECT_ROOT = Path(__file__).resolve().parent

    # =====================================================================
    # DIRECTORY PATHS
    # =====================================================================

    REPORTS_DIR = PROJECT_ROOT / "reports"

    EXPORTS_DIR = PROJECT_ROOT / "exports"

    ASSETS_DIR = PROJECT_ROOT / "assets"

    ICONS_DIR = ASSETS_DIR / "icons"

    # =====================================================================
    # DATASET PATH
    # =====================================================================

    # The main LEBAS dataset must be located here:
    #
    # LEBAS/
    #     dataset.csv
    #
    DATASET_PATH = PROJECT_ROOT / "dataset.csv"

    # =====================================================================
    # EXPORT SETTINGS
    # =====================================================================

    SUPPORTED_EXPORT_FORMATS: List[str] = [
        "excel",
        "csv",
        "pdf",
    ]

    # =====================================================================
    # REPORT SETTINGS
    # =====================================================================

    REPORT_SETTINGS = {
        "company_name": "LEBAS Laboratory",
        "report_title": "Mineral Beneficiation Analysis Report",
        "page_size": "A4",
        "font_family": "Helvetica",
        "font_size": 10,
    }

    # =====================================================================
    # CHART SETTINGS
    # =====================================================================

    CHART_SETTINGS = {
        "figure_dpi": 100,
        "style": "seaborn-v0_8-whitegrid",
        "color_palette": "viridis",
        "default_figsize": (10, 6),
        "font_family": "Arial",
        "font_size": 10,
        "title_font_size": 14,
        "label_font_size": 12,
        "legend_font_size": 10,
        "tick_font_size": 10,
        "grid_alpha": 0.3,
        "save_dpi": 300,
    }

    # =====================================================================
    # USER INTERFACE SETTINGS
    # =====================================================================

    UI_SETTINGS = {
        "window_title": f"{APP_NAME} - {APP_DESCRIPTION}",
        "window_size": (1200, 800),
        "min_window_size": (800, 600),
        "theme": "light",
        "show_status_bar": True,
        "show_toolbar": True,
    }

    # =====================================================================
    # DATA VALIDATION SETTINGS
    # =====================================================================

    VALIDATION_SETTINGS = {
        "max_mass": 10000.0,
        "min_mass": 0.0,

        "max_grade": 100.0,
        "min_grade": 0.0,

        "max_moisture": 100.0,
        "min_moisture": 0.0,

        "min_ph": 0.0,
        "max_ph": 14.0,

        "max_recovery": 100.0,
        "min_recovery": 0.0,

        "max_sieve_size": 100000.0,
        "min_sieve_size": 0.0,

        "max_temperature": 100.0,
        "min_temperature": -20.0,
    }

    # =====================================================================
    # DATASET COLUMN NAMES
    # =====================================================================

    COLUMN_NAMES = {
        "sample_id": "sample_id",
        "material_type": "material_type",
        "experiment_date": "experiment_date",
        "operator": "operator",
        "replicate": "replicate",

        "feed_mass": "feed_mass",
        "feed_grade": "feed_grade",
        "feed_moisture": "feed_moisture",

        "crusher_type": "crusher_type",
        "crush_time": "crush_time",
        "crush_power": "crush_power",
        "crush_feed_size": "crush_feed_size",
        "crush_product_size": "crush_product_size",

        "grinding_time": "grinding_time",
        "grinding_mill_type": "grinding_mill_type",
        "grinding_media_mass": "grinding_media_mass",
        "grinding_speed": "grinding_speed",
        "grinding_feed_size": "grinding_feed_size",
        "grinding_product_size": "grinding_product_size",

        "sieve_size": "sieve_size",
        "mass_retained": "mass_retained",
        "sieve_shaking_time": "sieve_shaking_time",

        "beneficiation_method": "beneficiation_method",
        "beneficiation_time": "beneficiation_time",
        "reagent_type": "reagent_type",
        "reagent_dosage": "reagent_dosage",
        "ph_level": "ph_level",
        "pulp_density": "pulp_density",
        "temperature": "temperature",

        "concentrate_mass": "concentrate_mass",
        "concentrate_grade": "concentrate_grade",

        "tailings_mass": "tailings_mass",
        "tailings_grade": "tailings_grade",

        "losses_mass": "losses_mass",
        "water_added": "water_added",

        "solid_recovery": "solid_recovery",
    }

    # =====================================================================
    # REQUIRED DATASET COLUMNS
    # =====================================================================

    REQUIRED_COLUMNS: List[str] = [
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
        "crush_power",
        "crush_feed_size",
        "crush_product_size",

        "grinding_time",
        "grinding_mill_type",
        "grinding_media_mass",
        "grinding_speed",
        "grinding_feed_size",
        "grinding_product_size",

        "sieve_size",
        "mass_retained",
        "sieve_shaking_time",

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

    # =====================================================================
    # LABORATORY SIEVE SIZES
    # =====================================================================

    # These are the standard/reference sizes used by LEBAS.
    #
    # IMPORTANT:
    # 150 µm is included because the laboratory sieve shaker used for
    # this project has a 150 µm sieve as its smallest opening.
    #
    # The application should still accept other positive sieve sizes
    # present in dataset.csv.
    STANDARD_SIEVE_SIZES: List[float] = [
        2000.0,
        1400.0,
        1000.0,
        710.0,
        500.0,
        355.0,
        250.0,
        180.0,
        150.0,
        125.0,
        90.0,
        63.0,
        45.0,
    ]

    # =====================================================================
    # MATERIAL TYPES
    # =====================================================================

    MATERIAL_TYPES: List[str] = [
        "limestone",
        "copper_ore",
        "gold_ore",
        "iron_ore",
        "bauxite",
        "phosphate_ore",
        "coal",
        "dolomite",
    ]

    # =====================================================================
    # BENEFICIATION METHODS
    # =====================================================================

    BENEFICIATION_METHODS: List[str] = [
        "flotation",
        "magnetic",
        "gravity",
        "leaching",
        "roasting",
        "none",
    ]

    # =====================================================================
    # CRUSHER TYPES
    # =====================================================================

    CRUSHER_TYPES: List[str] = [
        "jaw_crusher",
        "cone_crusher",
        "roll_crusher",
        "impact_crusher",
        "gyratory_crusher",
    ]

    # =====================================================================
    # GRINDING MILL TYPES
    # =====================================================================

    MILL_TYPES: List[str] = [
        "ball_mill",
        "rod_mill",
        "sag_mill",
        "pebble_mill",
        "attritor_mill",
    ]

    # =====================================================================
    # REAGENT TYPES
    # =====================================================================

    REAGENT_TYPES: List[str] = [
        "xanthate",
        "frother",
        "collector",
        "depressant",
        "activator",
        "fatty_acid",
        "none",
    ]

    # =====================================================================
    # PATH HELPER
    # =====================================================================

    @classmethod
    def get_paths(cls) -> Dict[str, Path]:
        """
        Return all important application paths.

        Returns
        -------
        Dict[str, Path]
            Dictionary containing the main LEBAS paths.
        """

        return {
            "project_root": cls.PROJECT_ROOT,
            "reports_dir": cls.REPORTS_DIR,
            "exports_dir": cls.EXPORTS_DIR,
            "assets_dir": cls.ASSETS_DIR,
            "icons_dir": cls.ICONS_DIR,
            "dataset_path": cls.DATASET_PATH,
        }

    # =====================================================================
    # DIRECTORY CREATION
    # =====================================================================

    @classmethod
    def ensure_directories(cls) -> None:
        """
        Create required application directories if they do not exist.
        """

        directories = [
            cls.REPORTS_DIR,
            cls.EXPORTS_DIR,
            cls.ASSETS_DIR,
            cls.ICONS_DIR,
        ]

        for directory in directories:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    # =====================================================================
    # VALIDATION LIMITS
    # =====================================================================

    @classmethod
    def get_validation_limits(
        cls,
        parameter: str,
    ) -> Tuple[float, float]:
        """
        Get minimum and maximum validation limits.

        Parameters
        ----------
        parameter:
            Parameter category, such as:
            mass, grade, moisture, ph, recovery,
            sieve_size, or temperature.

        Returns
        -------
        Tuple[float, float]
            Minimum and maximum allowed values.
        """

        mapping = {
            "mass": (
                cls.VALIDATION_SETTINGS["min_mass"],
                cls.VALIDATION_SETTINGS["max_mass"],
            ),

            "grade": (
                cls.VALIDATION_SETTINGS["min_grade"],
                cls.VALIDATION_SETTINGS["max_grade"],
            ),

            "moisture": (
                cls.VALIDATION_SETTINGS["min_moisture"],
                cls.VALIDATION_SETTINGS["max_moisture"],
            ),

            "ph": (
                cls.VALIDATION_SETTINGS["min_ph"],
                cls.VALIDATION_SETTINGS["max_ph"],
            ),

            "recovery": (
                cls.VALIDATION_SETTINGS["min_recovery"],
                cls.VALIDATION_SETTINGS["max_recovery"],
            ),

            "sieve_size": (
                cls.VALIDATION_SETTINGS["min_sieve_size"],
                cls.VALIDATION_SETTINGS["max_sieve_size"],
            ),

            "temperature": (
                cls.VALIDATION_SETTINGS["min_temperature"],
                cls.VALIDATION_SETTINGS["max_temperature"],
            ),
        }

        return mapping.get(
            parameter,
            (0.0, 0.0),
        )


# =============================================================================
# SINGLETON CONFIGURATION INSTANCE
# =============================================================================

config = Config()


# =============================================================================
# CREATE REQUIRED DIRECTORIES
# =============================================================================

config.ensure_directories()
