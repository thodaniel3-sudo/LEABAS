"""
Grinding Calculation Module for LEBAS

This module provides calculations for grinding analysis based on the experimental
data loaded from dataset.csv. All calculations operate on the dataset values
and do not hard-code experimental results.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field

from config import Config


@dataclass
class GrindingResult:
    """
    Container for grinding calculation results.
    """
    sample_id: str = ""
    material_type: str = ""
    replicate: Optional[int] = None
    feed_mass: Optional[float] = None
    ground_product_mass: Optional[float] = None  # Mass after grinding
    mass_loss: Optional[float] = None
    mass_loss_percent: Optional[float] = None
    grinding_yield: Optional[float] = None  # Product mass / Feed mass * 100
    reduction_ratio: Optional[float] = None
    feed_size: Optional[float] = None  # F80 or feed P80
    product_size: Optional[float] = None  # Product P80
    size_reduction_ratio: Optional[float] = None  # Feed size / Product size
    grinding_time: Optional[float] = None
    grinding_speed: Optional[float] = None
    media_mass: Optional[float] = None
    media_to_feed_ratio: Optional[float] = None  # Media mass / Feed mass
    specific_energy: Optional[float] = None  # kWh/ton
    p80_feed: Optional[float] = None
    p80_product: Optional[float] = None
    p80_reduction_ratio: Optional[float] = None
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the grinding result to a dictionary."""
        return {
            "sample_id": self.sample_id,
            "material_type": self.material_type,
            "replicate": self.replicate,
            "feed_mass": self.feed_mass,
            "ground_product_mass": self.ground_product_mass,
            "mass_loss": self.mass_loss,
            "mass_loss_percent": self.mass_loss_percent,
            "grinding_yield": self.grinding_yield,
            "reduction_ratio": self.reduction_ratio,
            "feed_size": self.feed_size,
            "product_size": self.product_size,
            "size_reduction_ratio": self.size_reduction_ratio,
            "grinding_time": self.grinding_time,
            "grinding_speed": self.grinding_speed,
            "media_mass": self.media_mass,
            "media_to_feed_ratio": self.media_to_feed_ratio,
            "specific_energy": self.specific_energy,
            "p80_feed": self.p80_feed,
            "p80_product": self.p80_product,
            "p80_reduction_ratio": self.p80_reduction_ratio,
            "is_valid": self.is_valid,
            "warnings": self.warnings,
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert the grinding result to a DataFrame."""
        return pd.DataFrame([self.to_dict()])


class GrindingCalculator:
    """
    Performs grinding analysis calculations on the dataset.
    
    This class calculates various grinding metrics from the experimental data
    including mass balance, reduction ratios, and energy consumption.
    """
    
    def __init__(self):
        """Initialize the grinding calculator."""
        self.column_names = Config.COLUMN_NAMES
        
    def calculate_grinding_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate grinding metrics for all samples in the dataset.
        
        Args:
            df: DataFrame containing the dataset.
            
        Returns:
            pd.DataFrame: DataFrame with calculated grinding metrics.
        """
        if df.empty:
            return pd.DataFrame()
        
        # Check required columns
        required_cols = [
            "sample_id", "replicate", "feed_mass",
            "grinding_time", "grinding_speed", "grinding_media_mass",
            "grinding_feed_size", "grinding_product_size"
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for grinding analysis: {missing_cols}")
        
        # Group by sample_id and replicate
        results = []
        
        for (sample_id, replicate), group in df.groupby(["sample_id", "replicate"]):
            # Get first row of the group (assumes same grinding parameters for all sieve rows)
            first_row = group.iloc[0]
            
            # Get sieve data for P80 calculation
            sieve_data = group[["sieve_size", "mass_retained"]].dropna()
            
            result = self._calculate_sample_grinding(
                sample_id=sample_id,
                replicate=replicate,
                data=first_row,
                sieve_data=sieve_data
            )
            results.append(result.to_dict())
        
        return pd.DataFrame(results)
    
    def _calculate_sample_grinding(self, sample_id: str, replicate: int, 
                                   data: pd.Series, 
                                   sieve_data: Optional[pd.DataFrame] = None) -> GrindingResult:
        """
        Calculate grinding metrics for a single sample.
        
        Args:
            sample_id: Sample identifier.
            replicate: Replicate number.
            data: Series containing the sample data.
            sieve_data: DataFrame with sieve analysis data for P80 calculation.
            
        Returns:
            GrindingResult: Calculated grinding metrics.
        """
        result = GrindingResult(
            sample_id=sample_id,
            replicate=replicate,
            material_type=data.get("material_type", ""),
            grinding_time=self._safe_float(data.get("grinding_time")),
            grinding_speed=self._safe_float(data.get("grinding_speed")),
            media_mass=self._safe_float(data.get("grinding_media_mass")),
        )
        
        # Get feed mass (from dataset)
        feed_mass = self._safe_float(data.get("feed_mass"))
        result.feed_mass = feed_mass
        
        # Get product mass after grinding (from dataset if available)
        # Grinding mass loss can be calculated from losses
        losses = self._safe_float(data.get("losses_mass"))
        
        if feed_mass is not None and losses is not None:
            # Product mass = feed mass - losses
            result.ground_product_mass = feed_mass - losses
            
            # Mass loss calculation
            result.mass_loss = losses
            
            if feed_mass > 0:
                result.mass_loss_percent = (losses / feed_mass) * 100
                result.grinding_yield = (result.ground_product_mass / feed_mass) * 100
        else:
            # If losses not available, assume no mass loss
            result.ground_product_mass = feed_mass
            result.mass_loss = 0.0
            result.mass_loss_percent = 0.0
            if feed_mass is not None and feed_mass > 0:
                result.grinding_yield = 100.0
            else:
                result.grinding_yield = None
        
        # Media to feed ratio
        if result.media_mass is not None and feed_mass is not None and feed_mass > 0:
            result.media_to_feed_ratio = result.media_mass / feed_mass
        else:
            result.media_to_feed_ratio = None
        
        # Size reduction calculations
        feed_size = self._safe_float(data.get("grinding_feed_size"))
        product_size = self._safe_float(data.get("grinding_product_size"))
        
        result.feed_size = feed_size
        result.product_size = product_size
        
        if feed_size is not None and product_size is not None:
            if feed_size > 0:
                result.reduction_ratio = feed_size / product_size if product_size > 0 else None
                result.size_reduction_ratio = result.reduction_ratio
            else:
                result.reduction_ratio = None
                result.size_reduction_ratio = None
        else:
            result.reduction_ratio = None
            result.size_reduction_ratio = None
        
        # Calculate P80 from sieve data if available
        if sieve_data is not None and not sieve_data.empty:
            p80_feed, p80_product = self._calculate_p80(sieve_data, feed_mass)
            result.p80_feed = p80_feed
            result.p80_product = p80_product
            
            if p80_feed is not None and p80_product is not None and p80_product > 0:
                result.p80_reduction_ratio = p80_feed / p80_product
            else:
                result.p80_reduction_ratio = None
        
        # Calculate specific energy (kWh/ton) - using power from crushing as proxy
        # Since dataset doesn't have grinding power directly, we use crushing power
        # as an approximation or leave as None
        power = self._safe_float(data.get("crush_power"))
        time = result.grinding_time
        
        if power is not None and time is not None and feed_mass is not None and feed_mass > 0:
            # Energy = Power (kW) * Time (hours)
            energy_kwh = power * (time / 60.0)  # Convert minutes to hours
            result.specific_energy = (energy_kwh / feed_mass) * 1000  # kWh/ton
        else:
            result.specific_energy = None
        
        # Validate results
        self._validate_result(result)
        
        return result
    
    def _calculate_p80(self, sieve_data: pd.DataFrame, feed_mass: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate P80 (80% passing size) from sieve analysis data.
        
        Args:
            sieve_data: DataFrame with sieve_size and mass_retained columns.
            feed_mass: Total feed mass for normalization.
            
        Returns:
            Tuple[Optional[float], Optional[float]]: (P80_feed, P80_product)
        """
        if sieve_data.empty:
            return None, None
        
        # Sort by sieve size (ascending)
        sieve_data = sieve_data.sort_values("sieve_size")
        
        # Calculate total mass
        total_mass = sieve_data["mass_retained"].sum()
        
        if total_mass <= 0:
            return None, None
        
        # Calculate cumulative mass retained (from coarsest to finest)
        # For P80, we need cumulative passing from finest to coarsest
        # Or cumulative retained from coarsest to finest
        sieve_data = sieve_data.sort_values("sieve_size", ascending=False)
        cumulative_retained = 0
        cumulative_passing = {}
        
        for idx, row in sieve_data.iterrows():
            cumulative_retained += row["mass_retained"]
            cumulative_passing[row["sieve_size"]] = 100 * (1 - cumulative_retained / total_mass)
        
        # Calculate P80 (80% passing)
        p80 = self._interpolate_percentile(cumulative_passing, 80)
        
        # Also calculate P50 (median) for reference
        # p50 = self._interpolate_percentile(cumulative_passing, 50)
        
        return p80, None  # Return P80 for feed size approximation
    
    def _interpolate_percentile(self, data: Dict[float, float], percentile: float) -> Optional[float]:
        """
        Interpolate to find the size at a given percentile passing.
        
        Args:
            data: Dictionary mapping sieve_size to cumulative passing percentage.
            percentile: Target percentile (e.g., 80 for P80).
            
        Returns:
            Optional[float]: Size at the target percentile.
        """
        if not data:
            return None
        
        # Sort by sieve size
        sorted_sizes = sorted(data.keys())
        
        # Find the interval containing the percentile
        for i in range(len(sorted_sizes) - 1):
            size1 = sorted_sizes[i]
            size2 = sorted_sizes[i + 1]
            p1 = data[size1]
            p2 = data[size2]
            
            # Check if percentile falls between p1 and p2
            if (p1 >= percentile >= p2) or (p1 <= percentile <= p2):
                # Linear interpolation
                if p2 != p1:
                    fraction = (percentile - p1) / (p2 - p1)
                    return size1 + fraction * (size2 - size1)
                else:
                    return size1
        
        # If percentile is outside the range, return the nearest value
        if sorted_sizes:
            if data[sorted_sizes[0]] >= percentile:
                return sorted_sizes[0]
            elif data[sorted_sizes[-1]] <= percentile:
                return sorted_sizes[-1]
        
        return None
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """
        Safely convert a value to float, handling None and NaN.
        
        Args:
            value: Value to convert.
            
        Returns:
            Optional[float]: Converted float or None.
        """
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _validate_result(self, result: GrindingResult) -> None:
        """
        Validate grinding results and add warnings.
        
        Args:
            result: GrindingResult to validate.
        """
        # Check for unrealistic mass loss
        if result.mass_loss_percent is not None:
            if result.mass_loss_percent > 10:
                result.warnings.append(
                    f"High mass loss during grinding: {result.mass_loss_percent:.1f}%"
                )
            elif result.mass_loss_percent < 0:
                result.warnings.append(
                    f"Negative mass loss: {result.mass_loss_percent:.1f}%"
                )
        
        # Check for unrealistic reduction ratio
        if result.reduction_ratio is not None:
            if result.reduction_ratio < 1:
                result.warnings.append(
                    f"Reduction ratio less than 1: {result.reduction_ratio:.2f}"
                )
            elif result.reduction_ratio > 1000:
                result.warnings.append(
                    f"Very high reduction ratio: {result.reduction_ratio:.2f}"
                )
        
        # Check for unrealistic media to feed ratio
        if result.media_to_feed_ratio is not None:
            if result.media_to_feed_ratio > 10:
                result.warnings.append(
                    f"High media to feed ratio: {result.media_to_feed_ratio:.2f}"
                )
            elif result.media_to_feed_ratio < 0.1:
                result.warnings.append(
                    f"Low media to feed ratio: {result.media_to_feed_ratio:.2f}"
                )
        
        # Check for missing critical data
        if result.feed_mass is None:
            result.warnings.append("Missing feed mass data")
            result.is_valid = False
        
        if result.feed_size is None:
            result.warnings.append("Missing feed size data")
        
        if result.product_size is None:
            result.warnings.append("Missing product size data")
        
        # Mark as invalid if feed mass is missing or zero
        if result.feed_mass is None or result.feed_mass <= 0:
            result.is_valid = False
    
    def get_grinding_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of grinding analysis.
        
        Args:
            df: DataFrame containing the dataset.
            
        Returns:
            Dict[str, Any]: Summary statistics for grinding.
        """
        grinding_df = self.calculate_grinding_metrics(df)
        
        if grinding_df.empty:
            return {
                "total_samples": 0,
                "average_yield": None,
                "average_reduction_ratio": None,
                "min_reduction_ratio": None,
                "max_reduction_ratio": None,
                "average_specific_energy": None,
                "average_media_ratio": None,
                "total_time": None,
            }
        
        # Calculate summary statistics
        summary = {
            "total_samples": len(grinding_df),
            "average_yield": grinding_df["grinding_yield"].mean() if "grinding_yield" in grinding_df else None,
            "average_reduction_ratio": grinding_df["reduction_ratio"].mean() if "reduction_ratio" in grinding_df else None,
            "min_reduction_ratio": grinding_df["reduction_ratio"].min() if "reduction_ratio" in grinding_df else None,
            "max_reduction_ratio": grinding_df["reduction_ratio"].max() if "reduction_ratio" in grinding_df else None,
            "average_specific_energy": grinding_df["specific_energy"].mean() if "specific_energy" in grinding_df else None,
            "average_media_ratio": grinding_df["media_to_feed_ratio"].mean() if "media_to_feed_ratio" in grinding_df else None,
            "total_time": grinding_df["grinding_time"].sum() if "grinding_time" in grinding_df else None,
        }
        
        return summary
    
    def calculate_grinding_by_material(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate grinding metrics grouped by material type.
        
        Args:
            df: DataFrame containing the dataset.
            
        Returns:
            pd.DataFrame: Grinding metrics by material type.
        """
        grinding_df = self.calculate_grinding_metrics(df)
        
        if grinding_df.empty:
            return pd.DataFrame()
        
        # Group by material type
        grouped = grinding_df.groupby("material_type").agg({
            "sample_id": "count",
            "grinding_yield": "mean",
            "reduction_ratio": "mean",
            "specific_energy": "mean",
            "media_to_feed_ratio": "mean",
            "feed_mass": "sum",
        }).reset_index()
        
        grouped.columns = [
            "material_type", 
            "sample_count", 
            "avg_grinding_yield_percent",
            "avg_reduction_ratio",
            "avg_specific_energy_kwh_ton",
            "avg_media_to_feed_ratio",
            "total_feed_mass_g"
        ]
        
        return grouped


# Convenience functions

def calculate_grinding_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate grinding metrics for all samples.
    
    Args:
        df: DataFrame containing the dataset.
        
    Returns:
        pd.DataFrame: DataFrame with calculated grinding metrics.
    """
    calculator = GrindingCalculator()
    return calculator.calculate_grinding_metrics(df)


def get_grinding_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get a summary of grinding analysis.
    
    Args:
        df: DataFrame containing the dataset.
        
    Returns:
        Dict[str, Any]: Summary statistics for grinding.
    """
    calculator = GrindingCalculator()
    return calculator.get_grinding_summary(df)


def calculate_grinding_by_material(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate grinding metrics grouped by material type.
    
    Args:
        df: DataFrame containing the dataset.
        
    Returns:
        pd.DataFrame: Grinding metrics by material type.
    """
    calculator = GrindingCalculator()
    return calculator.calculate_grinding_by_material(df)


def get_grinding_result_for_sample(df: pd.DataFrame, sample_id: str, 
                                   replicate: Optional[int] = None) -> Optional[GrindingResult]:
    """
    Get grinding results for a specific sample.
    
    Args:
        df: DataFrame containing the dataset.
        sample_id: Sample identifier.
        replicate: Optional replicate number.
        
    Returns:
        Optional[GrindingResult]: Grinding results for the sample.
    """
    calculator = GrindingCalculator()
    
    # Filter data for specific sample
    mask = df["sample_id"] == sample_id
    if replicate is not None:
        mask = mask & (df["replicate"] == replicate)
    
    sample_data = df[mask]
    
    if sample_data.empty:
        return None
    
    first_row = sample_data.iloc[0]
    
    # Get sieve data for P80 calculation
    sieve_data = sample_data[["sieve_size", "mass_retained"]].dropna()
    
    result = calculator._calculate_sample_grinding(
        sample_id=sample_id,
        replicate=first_row.get("replicate", None),
        data=first_row,
        sieve_data=sieve_data
    )
    
    return result


# Export public API
__all__ = [
    'GrindingCalculator',
    'GrindingResult',
    'calculate_grinding_metrics',
    'get_grinding_summary',
    'calculate_grinding_by_material',
    'get_grinding_result_for_sample',
]