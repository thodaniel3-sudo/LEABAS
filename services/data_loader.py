"""
LEBAS - Dataset Loading Service

File:
    services/data_loader.py

Purpose:
    Centralized loading, cleaning, validation, caching, and summary
    handling for the LEBAS experimental dataset.

Dataset:
    dataset.csv

Important design principles:
    - dataset.csv is the ONLY experimental data file that should need
      replacement when real laboratory data becomes available.
    - The application does not hard-code experimental results.
    - Sieve sizes are detected dynamically from the dataset.
    - The smallest sieve opening can be 150 micrometres or any other
      valid positive value contained in the dataset.
    - Multiple rows belonging to the same sample/replicate are allowed,
      especially for sieve analysis.
    - Process-level operations should use unique sample_id + replicate
      combinations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config import Config


# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

REQUIRED_COLUMNS = [
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


# Numeric columns used throughout the application.
NUMERIC_COLUMNS = [
    "replicate",
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
    "sieve_size",
    "mass_retained",
    "sieve_shaking_time",
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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _safe_numeric(series: pd.Series) -> pd.Series:
    """
    Safely convert a pandas Series to numeric values.

    Invalid values are converted to NaN instead of crashing the
    application.
    """
    return pd.to_numeric(series, errors="coerce")


def _clean_text(value: Any) -> Any:
    """
    Clean a single text value.

    Empty strings and common null-like strings become None.
    """
    if pd.isna(value):
        return None

    text = str(value).strip()

    if text == "":
        return None

    if text.lower() in {
        "nan",
        "none",
        "null",
        "n/a",
        "na",
        "-",
    }:
        return None

    return text


def _normalise_column_name(column: Any) -> str:
    """
    Convert a dataframe column name into a clean string.
    """
    return str(column).strip()


def _normalise_sample_id(value: Any) -> Any:
    """
    Clean sample IDs without changing their actual identity.
    """
    if pd.isna(value):
        return None

    text = str(value).strip()

    if text == "":
        return None

    return text


# ============================================================================
# DATA LOADER
# ============================================================================

class DataLoader:
    """
    Centralized dataset loader for the LEBAS application.

    Responsibilities:
        - Locate dataset.csv
        - Load CSV data
        - Clean data
        - Convert numeric columns
        - Validate basic dataset structure
        - Cache loaded data
        - Provide dataset summaries
        - Provide unique samples and replicates
        - Detect sieve sizes dynamically
    """

    # ------------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------------

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
    ) -> None:
        """
        Initialize the DataLoader.

        Parameters
        ----------
        dataset_path:
            Optional custom dataset path.

        If no path is supplied, Config.DATASET_PATH is used.
        """

        self._dataset_path: Path = self._resolve_dataset_path(
            dataset_path
        )

        self._dataframe: Optional[pd.DataFrame] = None

        self._is_loaded: bool = False

        self._last_error: Optional[str] = None

        self._validation_result: Optional[Any] = None

    # ------------------------------------------------------------------------
    # PATH HANDLING
    # ------------------------------------------------------------------------

    @staticmethod
    def _resolve_dataset_path(
        dataset_path: Optional[Path],
    ) -> Path:
        """
        Resolve the dataset path safely.
        """

        if dataset_path is not None:
            return Path(dataset_path).expanduser().resolve()

        config_path = getattr(
            Config,
            "DATASET_PATH",
            None,
        )

        if config_path is not None:
            return Path(config_path).expanduser().resolve()

        # Fallback.
        return (
            Path(__file__)
            .resolve()
            .parent
            .parent
            / "dataset.csv"
        )

    # ------------------------------------------------------------------------
    # SET DATASET PATH
    # ------------------------------------------------------------------------

    def set_dataset_path(
        self,
        dataset_path: Path,
    ) -> None:
        """
        Change the dataset path and clear the existing cache.
        """

        self._dataset_path = (
            Path(dataset_path)
            .expanduser()
            .resolve()
        )

        self.clear_cache()

        logger.info(
            "Dataset path changed to: %s",
            self._dataset_path,
        )

    # ------------------------------------------------------------------------
    # GET DATASET PATH
    # ------------------------------------------------------------------------

    @property
    def dataset_path(self) -> Path:
        """
        Return the current dataset path.
        """

        return self._dataset_path

    # ------------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------------

    def load(
        self,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """
        Load dataset.csv.

        Parameters
        ----------
        force_reload:
            If True, reload the file even when cached data exists.

        Returns
        -------
        pandas.DataFrame
            Cleaned dataset.
        """

        # Return cached data when available.
        if (
            self._dataframe is not None
            and self._is_loaded
            and not force_reload
        ):
            return self._dataframe.copy()

        logger.info(
            "Loading dataset from %s",
            self._dataset_path,
        )

        self._last_error = None

        # Check that the file exists.
        if not self._dataset_path.exists():
            error_message = (
                f"Dataset file not found: "
                f"{self._dataset_path}"
            )

            self._last_error = error_message
            self._is_loaded = False

            logger.error(error_message)

            raise FileNotFoundError(error_message)

        # Check that it is actually a file.
        if not self._dataset_path.is_file():
            error_message = (
                f"Dataset path is not a file: "
                f"{self._dataset_path}"
            )

            self._last_error = error_message
            self._is_loaded = False

            logger.error(error_message)

            raise FileNotFoundError(error_message)

        try:
            # Read CSV.
            dataframe = pd.read_csv(
                self._dataset_path
            )

            logger.info(
                "CSV loaded successfully. Rows: %s, Columns: %s",
                len(dataframe),
                len(dataframe.columns),
            )

            # Clean column names.
            dataframe.columns = [
                _normalise_column_name(column)
                for column in dataframe.columns
            ]

            # Validate required columns.
            self._check_required_columns(
                dataframe
            )

            # Clean dataframe.
            dataframe = self._clean_dataframe(
                dataframe
            )

            # Store cache.
            self._dataframe = dataframe.copy()

            self._is_loaded = True

            logger.info(
                "Dataset loaded and cleaned successfully. "
                "Rows: %s, Columns: %s",
                len(dataframe),
                len(dataframe.columns),
            )

            return dataframe.copy()

        except Exception as exc:
            self._last_error = str(exc)
            self._is_loaded = False

            logger.exception(
                "Failed to load dataset: %s",
                exc,
            )

            raise

    # ------------------------------------------------------------------------
    # RELOAD
    # ------------------------------------------------------------------------

    def reload(self) -> pd.DataFrame:
        """
        Force a fresh reload from dataset.csv.
        """

        return self.load(
            force_reload=True
        )

    # ------------------------------------------------------------------------
    # COMPATIBILITY LOAD DATASET METHOD
    # ------------------------------------------------------------------------

    def load_dataset(
        self,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """
        Compatibility method for modules that call:

            data_loader.load_dataset()

        It uses the same centralized loading system as load().
        """

        return self.load(
            force_reload=force_reload
        )

    # ------------------------------------------------------------------------
    # GET CACHED DATA
    # ------------------------------------------------------------------------

    def get_dataframe(
        self,
        reload_if_needed: bool = True,
    ) -> pd.DataFrame:
        """
        Return the current dataframe.

        If data has not yet been loaded and reload_if_needed is True,
        the dataset will be loaded automatically.
        """

        if (
            self._dataframe is None
            or not self._is_loaded
        ):
            if reload_if_needed:
                return self.load()

            return pd.DataFrame()

        return self._dataframe.copy()

    # ------------------------------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # ------------------------------------------------------------------------

    def _check_required_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Check that the expected dataset columns exist.

        Extra columns are allowed.
        """

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                "Dataset is missing required columns: "
                + ", ".join(missing_columns)
            )

    # ------------------------------------------------------------------------
    # CLEAN DATAFRAME
    # ------------------------------------------------------------------------

    def _clean_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Clean and normalize the loaded dataframe.
        """

        dataframe = dataframe.copy()

        # ------------------------------------------------------------
        # Remove completely empty rows.
        # ------------------------------------------------------------

        dataframe = dataframe.dropna(
            how="all"
        ).reset_index(drop=True)

        # ------------------------------------------------------------
        # Clean text fields.
        # ------------------------------------------------------------

        text_columns = [
            column
            for column in dataframe.columns
            if column not in NUMERIC_COLUMNS
        ]

        for column in text_columns:
            if column in dataframe.columns:
                dataframe[column] = dataframe[column].map(
                    _clean_text
                )

        # ------------------------------------------------------------
        # Clean sample ID.
        # ------------------------------------------------------------

        if "sample_id" in dataframe.columns:
            dataframe["sample_id"] = dataframe[
                "sample_id"
            ].map(
                _normalise_sample_id
            )

        # ------------------------------------------------------------
        # Convert numeric columns.
        # ------------------------------------------------------------

        for column in NUMERIC_COLUMNS:
            if column in dataframe.columns:
                dataframe[column] = _safe_numeric(
                    dataframe[column]
                )

        # ------------------------------------------------------------
        # Clean experiment date.
        # ------------------------------------------------------------

        if "experiment_date" in dataframe.columns:
            dataframe["experiment_date"] = pd.to_datetime(
                dataframe["experiment_date"],
                errors="coerce",
            )

        # ------------------------------------------------------------
        # Sort data where possible.
        # ------------------------------------------------------------

        sort_columns = [
            column
            for column in [
                "sample_id",
                "replicate",
                "sieve_size",
            ]
            if column in dataframe.columns
        ]

        if sort_columns:
            dataframe = dataframe.sort_values(
                by=sort_columns,
                na_position="last",
            )

        # ------------------------------------------------------------
        # Reset index.
        # ------------------------------------------------------------

        dataframe = dataframe.reset_index(
            drop=True
        )

        return dataframe

    # ------------------------------------------------------------------------
    # VALIDATE
    # ------------------------------------------------------------------------

    def validate(
        self,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Perform basic validation of the dataset.

        This method intentionally does not reject the dataset merely
        because some experimental fields are blank.

        Blank experimental values can be valid when a laboratory test
        has not yet been performed.

        Returns
        -------
        dict
            Validation information.
        """

        if dataframe is None:
            dataframe = self.get_dataframe(
                reload_if_needed=True
            )

        result: Dict[str, Any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
        }

        # ------------------------------------------------------------
        # Empty dataset.
        # ------------------------------------------------------------

        if dataframe.empty:
            result["valid"] = False

            result["errors"].append(
                "Dataset is empty."
            )

            return result

        # ------------------------------------------------------------
        # Required columns.
        # ------------------------------------------------------------

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in dataframe.columns
        ]

        if missing_columns:
            result["valid"] = False

            result["errors"].append(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        # ------------------------------------------------------------
        # Sample ID checks.
        # ------------------------------------------------------------

        if "sample_id" in dataframe.columns:

            missing_sample_ids = dataframe[
                "sample_id"
            ].isna().sum()

            if missing_sample_ids > 0:
                result["warnings"].append(
                    f"{missing_sample_ids} row(s) have "
                    "no sample_id."
                )

        # ------------------------------------------------------------
        # Replicate checks.
        # ------------------------------------------------------------

        if "replicate" in dataframe.columns:

            invalid_replicates = (
                dataframe["replicate"]
                .notna()
                & (
                    dataframe["replicate"] <= 0
                )
            ).sum()

            if invalid_replicates > 0:
                result["warnings"].append(
                    f"{invalid_replicates} row(s) have "
                    "non-positive replicate numbers."
                )

        # ------------------------------------------------------------
        # Sieve-size checks.
        # ------------------------------------------------------------

        if "sieve_size" in dataframe.columns:

            sieve_values = _safe_numeric(
                dataframe["sieve_size"]
            ).dropna()

            invalid_sieves = (
                sieve_values <= 0
            ).sum()

            if invalid_sieves > 0:
                result["warnings"].append(
                    f"{invalid_sieves} sieve size value(s) "
                    "are not positive."
                )

        # ------------------------------------------------------------
        # Feed mass checks.
        # ------------------------------------------------------------

        if "feed_mass" in dataframe.columns:

            feed_mass = _safe_numeric(
                dataframe["feed_mass"]
            )

            negative_feed = (
                feed_mass < 0
            ).sum()

            if negative_feed > 0:
                result["warnings"].append(
                    f"{negative_feed} row(s) have "
                    "negative feed mass."
                )

        # ------------------------------------------------------------
        # Return result.
        # ------------------------------------------------------------

        return result

    # ------------------------------------------------------------------------
    # IS DATASET VALID
    # ------------------------------------------------------------------------

    def is_dataset_valid(
        self,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> bool:
        """
        Return True when the basic dataset structure is valid.
        """

        result = self.validate(
            dataframe
        )

        return bool(
            result.get("valid", False)
        )

    # ------------------------------------------------------------------------
    # CLEAR CACHE
    # ------------------------------------------------------------------------

    def clear_cache(self) -> None:
        """
        Clear the currently cached dataframe.
        """

        self._dataframe = None

        self._is_loaded = False

        self._validation_result = None

        logger.info(
            "Dataset cache cleared."
        )

    # ------------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------------

    def get_summary(
        self,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Return a general summary of the dataset.
        """

        if dataframe is None:
            dataframe = self.get_dataframe()

        if dataframe.empty:
            return {
                "rows": 0,
                "columns": 0,
                "unique_samples": 0,
                "unique_replicates": 0,
                "unique_materials": 0,
                "sieve_sizes": [],
                "dataset_path": str(
                    self._dataset_path
                ),
            }

        return {
            "rows": int(
                len(dataframe)
            ),

            "columns": int(
                len(dataframe.columns)
            ),

            "unique_samples": int(
                dataframe["sample_id"]
                .dropna()
                .nunique()
                if "sample_id" in dataframe.columns
                else 0
            ),

            "unique_replicates": int(
                self._count_unique_replicates(
                    dataframe
                )
            ),

            "unique_materials": int(
                dataframe["material_type"]
                .dropna()
                .nunique()
                if "material_type" in dataframe.columns
                else 0
            ),

            "sieve_sizes": self.get_sieve_sizes(
                dataframe
            ),

            "dataset_path": str(
                self._dataset_path
            ),
        }

    # ------------------------------------------------------------------------
    # UNIQUE REPLICATE COUNT
    # ------------------------------------------------------------------------

    @staticmethod
    def _count_unique_replicates(
        dataframe: pd.DataFrame,
    ) -> int:
        """
        Count unique sample + replicate combinations.

        This prevents multiple sieve rows from being counted as
        separate laboratory replicates.
        """

        if (
            "sample_id" not in dataframe.columns
            or "replicate" not in dataframe.columns
        ):
            return 0

        pairs = dataframe[
            [
                "sample_id",
                "replicate",
            ]
        ].dropna()

        if pairs.empty:
            return 0

        return int(
            pairs.drop_duplicates().shape[0]
        )

    # ------------------------------------------------------------------------
    # UNIQUE SAMPLES
    # ------------------------------------------------------------------------

    def get_unique_samples(
        self,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> List[str]:
        """
        Return unique sample IDs.
        """

        if dataframe is None:
            dataframe = self.get_dataframe()

        if (
            dataframe.empty
            or "sample_id" not in dataframe.columns
        ):
            return []

        samples = (
            dataframe["sample_id"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        samples = [
            sample
            for sample in samples.unique().tolist()
            if sample
        ]

        return sorted(
            samples
        )

    # ------------------------------------------------------------------------
    # UNIQUE REPLICATES
    # ------------------------------------------------------------------------

    def get_unique_replicates(
        self,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> List[int]:
        """
        Return unique replicate numbers.
        """

        if dataframe is None:
            dataframe = self.get_dataframe()

        if (
            dataframe.empty
            or "replicate" not in dataframe.columns
        ):
            return []

        values = _safe_numeric(
            dataframe["replicate"]
        ).dropna()

        values = sorted(
            {
                int(value)
                for value in values
                if value > 0
            }
        )

        return values

    # ------------------------------------------------------------------------
    # UNIQUE MATERIALS
    # ------------------------------------------------------------------------

    def get_unique_materials(
        self,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> List[str]:
        """
        Return unique material types.
        """

        if dataframe is None:
            dataframe = self.get_dataframe()

        if (
            dataframe.empty
            or "material_type" not in dataframe.columns
        ):
            return []

        materials = (
            dataframe["material_type"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        materials = [
            material
            for material in materials.unique().tolist()
            if material
        ]

        return sorted(
            materials
        )

    # ------------------------------------------------------------------------
    # GET SIEVE SIZES
    # ------------------------------------------------------------------------

    def get_sieve_sizes(
        self,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> List[float]:
        """
        Detect sieve sizes dynamically from the dataset.

        No sieve size is hard-coded.

        Therefore, 150 micrometres is accepted automatically when
        it exists in dataset.csv.
        """

        if dataframe is None:
            dataframe = self.get_dataframe()

        if (
            dataframe.empty
            or "sieve_size" not in dataframe.columns
        ):
            return []

        sieve_values = _safe_numeric(
            dataframe["sieve_size"]
        ).dropna()

        sieve_values = sieve_values[
            sieve_values > 0
        ]

        unique_values = sorted(
            sieve_values.unique().tolist(),
            reverse=True,
        )

        return [
            float(value)
            for value in unique_values
        ]

    # ------------------------------------------------------------------------
    # GET SAMPLE DATA
    # ------------------------------------------------------------------------

    def get_sample_data(
        self,
        sample_id: str,
        replicate: Optional[int] = None,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Return all rows belonging to a specific sample.

        If replicate is supplied, only that replicate is returned.
        """

        if dataframe is None:
            dataframe = self.get_dataframe()

        if dataframe.empty:
            return pd.DataFrame(
                columns=dataframe.columns
            )

        if "sample_id" not in dataframe.columns:
            return pd.DataFrame(
                columns=dataframe.columns
            )

        result = dataframe[
            dataframe["sample_id"].astype(str).str.strip()
            == str(sample_id).strip()
        ].copy()

        if (
            replicate is not None
            and "replicate" in result.columns
        ):
            replicate_numeric = _safe_numeric(
                result["replicate"]
            )

            result = result[
                replicate_numeric
                == float(replicate)
            ]

        return result.reset_index(
            drop=True
        )

    # ------------------------------------------------------------------------
    # GET PROCESS-LEVEL DATA
    # ------------------------------------------------------------------------

    def get_process_data(
        self,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Return one representative row per sample + replicate.

        This is important because sieve analysis may contain several
        rows for one sample/replicate.

        The first row is retained for process-level calculations.
        """

        if dataframe is None:
            dataframe = self.get_dataframe()

        if dataframe.empty:
            return dataframe.copy()

        if not {
            "sample_id",
            "replicate",
        }.issubset(dataframe.columns):
            return dataframe.copy()

        process_data = dataframe.drop_duplicates(
            subset=[
                "sample_id",
                "replicate",
            ],
            keep="first",
        ).copy()

        return process_data.reset_index(
            drop=True
        )

    # ------------------------------------------------------------------------
    # DATASET STATUS
    # ------------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """
        Return the current loader status.
        """

        return {
            "dataset_path": str(
                self._dataset_path
            ),

            "exists": self._dataset_path.exists(),

            "is_loaded": self._is_loaded,

            "rows": (
                int(len(self._dataframe))
                if self._dataframe is not None
                else 0
            ),

            "columns": (
                int(len(self._dataframe.columns))
                if self._dataframe is not None
                else 0
            ),

            "last_error": self._last_error,
        }


# ============================================================================
# SINGLETON DATA LOADER
# ============================================================================

data_loader = DataLoader()


# ============================================================================
# MODULE-LEVEL FUNCTIONS
# ============================================================================

def load_dataset(
    force_reload: bool = False,
) -> pd.DataFrame:
    """
    Load the central LEBAS dataset.

    This function is provided for modules that use:

        from services.data_loader import load_dataset
    """

    return data_loader.load(
        force_reload=force_reload
    )


def get_dataset() -> pd.DataFrame:
    """
    Return the cached dataset.

    If it has not been loaded, it will be loaded automatically.
    """

    return data_loader.get_dataframe(
        reload_if_needed=True
    )


def reload_dataset() -> pd.DataFrame:
    """
    Force a fresh reload of the dataset.
    """

    return data_loader.reload()


def validate_dataset(
    dataframe: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Validate the dataset using the central DataLoader.
    """

    return data_loader.validate(
        dataframe
    )


def get_dataset_summary(
    dataframe: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Return dataset summary information.
    """

    return data_loader.get_summary(
        dataframe
    )


def get_sieve_sizes(
    dataframe: Optional[pd.DataFrame] = None,
) -> List[float]:
    """
    Return dynamically detected sieve sizes.
    """

    return data_loader.get_sieve_sizes(
        dataframe
    )


# ============================================================================
# BACKWARD-COMPATIBILITY ALIASES
# ============================================================================

load_data = load_dataset

get_data = get_dataset

reload_data = reload_dataset


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "DataLoader",
    "data_loader",
    "load_dataset",
    "get_dataset",
    "reload_dataset",
    "validate_dataset",
    "get_dataset_summary",
    "get_sieve_sizes",
    "load_data",
    "get_data",
    "reload_data",
]