"""
Crushing Calculation Module for LEBAS

This module provides calculations for crushing analysis based on the experimental
data loaded from dataset.csv. All calculations operate on the dataset values
and do not hard-code experimental results.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field

from config import Config


@dataclass
class CrushingResult:
    """
    Container for crushing calculation results.
    """
    sample_id: str = ""
    material_type: str = ""
    replicate: Optional[int] = None
    feed_mass: Optional[float] = None
    crushed_product_mass: Optional[float] = None  # Mass after crushing
    mass_loss: Optional[float] = None
    mass_loss_percent: Optional[float] = None
    crushing_yield: Optional[float] = None  # Product mass / Feed mass * 100
    reduction_ratio: Optional[float] = None
    feed_size: Optional[float] = None  # F80 or feed P80
    product_size: Optional[float] = None  # Product P80
    power_consumption: Optional[float] = None
    specific_energy: Optional[float] = None  # kWh/ton
    crush_time: Optional[float] = None
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the crushing result to a dictionary."""
        return {
            "sample_id": self.sample_id,
            "material_type": self.material_type,
            "replicate": self.replicate,
            "feed_mass": self.feed_mass,
            "crushed_product_mass": self.crushed_product_mass,
            "mass_loss": self.mass_loss,
            "mass_loss_percent": self.mass_loss_percent,
            "crushing_yield": self.crushing_yield,
            "reduction_ratio": self.reduction_ratio,
            "feed_size": self.feed_size,
            "product_size": self.product_size,
            "power_consumption": self.power_consumption,
            "specific_energy": self.specific_energy,
            "crush_time": self.crush_time,
            "is_valid": self.is_valid,
            "warnings": self.warnings,
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert the crushing result to a DataFrame."""
        return pd.DataFrame([self.to_dict()])


class CrushingCalculator:
    """
    Performs crushing analysis calculations on the dataset.
    
    This class calculates various crushing metrics from the experimental data
    including mass balance, reduction ratios, and energy consumption.
    """
    
    def __init__(self):
        """Initialize the crushing calculator."""
        self.column_names = Config.COLUMN_NAMES
    
    def calculate_crushing_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate crushing metrics for all samples in the dataset.
        
        Args:
            df: DataFrame containing the dataset.
            
        Returns:
            pd.DataFrame: DataFrame with calculated crushing metrics.
        """
        if df.empty:
            return pd.DataFrame()
        
        # Check required columns
        required_cols = [
            "sample_id", "replicate", "feed_mass",
            "crush_time", "crush_power", "crush_feed_size", "crush_product_size"
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for crushing analysis: {missing_cols}")
        
        # Group by sample_id and replicate
        results = []
        
        for (sample_id, replicate), group in df.groupby(["sample_id", "replicate"]):
            # Get first row of the group (assumes same crushing parameters for all sieve rows)
            first_row = group.iloc[0]
            
            result = self._calculate_sample_crushing(
                sample_id=sample_id,
                replicate=replicate,
                data=first_row
            )
            results.append(result.to_dict())
        
        return pd.DataFrame(results)
    
    def _calculate_sample_crushing(self, sample_id: str, replicate: int, 
                                   data: pd.Series) -> CrushingResult:
        """
        Calculate crushing metrics for a single sample.
        
        Args:
            sample_id: Sample identifier.
            replicate: Replicate number.
            data: Series containing the sample data.
            
        Returns:
            CrushingResult: Calculated crushing metrics.
        """
        result = CrushingResult(
            sample_id=sample_id,
            replicate=replicate,
            material_type=data.get("material_type", ""),
            crush_time=data.get("crush_time"),
        )
        
        # Get feed mass (from dataset)
        feed_mass = self._safe_float(data.get("feed_mass"))
        result.feed_mass = feed_mass
        
        # For crushing, we typically have the same mass before and after crushing
        # The mass after crushing might be approximated from the total mass of crushed product
        # or from the sum of mass retained in sieve analysis after grinding
        # Since crushing is followed by grinding in the dataset, we use the feed mass
        # as an approximation of the crushed mass (neglecting losses)
        
        # Calculate mass loss during crushing (if any)
        # Crushing typically has minimal mass loss, but we can calculate from any losses
        losses = self._safe_float(data.get("losses_mass"))
        
        if feed_mass is not None and losses is not None:
            # Product mass = feed mass - losses
            result.crushed_product_mass = feed_mass - losses
            
            # Mass loss calculation
            result.mass_loss = losses
            
            if feed_mass > 0:
                result.mass_loss_percent = (losses / feed_mass) * 100
                result.crushing_yield = (result.crushed_product_mass / feed_mass) * 100
        else:
            # If losses not available, assume no mass loss
            result.crushed_product_mass = feed_mass
            result.mass_loss = 0.0
            result.mass_loss_percent = 0.0
            if feed_mass is not None and feed_mass > 0:
                result.crushing_yield = 100.0
            else:
                result.crushing_yield = None
        
        # Reduction ratio calculation
        feed_size = self._safe_float(data.get("crush_feed_size"))
        product_size = self._safe_float(data.get("crush_product_size"))
        
        result.feed_size = feed_size
        result.product_size = product_size
        
        if feed_size is not None and product_size is not None and product_size > 0:
            result.reduction_ratio = feed_size / product_size
        else:
            result.reduction_ratio = None
        
        # Power and energy consumption
        power = self._safe_float(data.get("crush_power"))
        time = self._safe_float(data.get("crush_time"))
        
        result.power_consumption = power
        
        # Calculate specific energy (kWh/ton)
        if power is not None and time is not None and feed_mass is not None and feed_mass > 0:
            # Energy = Power (kW) * Time (hours)
            energy_kwh = power * (time / 60.0)  # Convert minutes to hours
            result.specific_energy = (energy_kwh / feed_mass) * 1000  # kWh/ton
        else:
            result.specific_energy = None
        
        # Validate results
        self._validate_result(result)
        
        return result
    
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
    
    def _validate_result(self, result: CrushingResult) -> None:
        """
        Validate crushing results and add warnings.
        
        Args:
            result: CrushingResult to validate.
        """
        # Check for unrealistic mass loss
        if result.mass_loss_percent is not None:
            if result.mass_loss_percent > 10:
                result.warnings.append(
                    f"High mass loss during crushing: {result.mass_loss_percent:.1f}%"
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
            elif result.reduction_ratio > 100:
                result.warnings.append(
                    f"Very high reduction ratio: {result.reduction_ratio:.2f}"
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
    
    def get_crushing_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of crushing analysis.
        
        Args:
            df: DataFrame containing the dataset.
            
        Returns:
            Dict[str, Any]: Summary statistics for crushing.
        """
        crushing_df = self.calculate_crushing_metrics(df)
        
        if crushing_df.empty:
            return {
                "total_samples": 0,
                "average_yield": None,
                "average_reduction_ratio": None,
                "min_reduction_ratio": None,
                "max_reduction_ratio": None,
                "average_specific_energy": None,
                "total_power": None,
                "total_time": None,
            }
        
        # Calculate summary statistics
        summary = {
            "total_samples": len(crushing_df),
            "average_yield": crushing_df["crushing_yield"].mean() if "crushing_yield" in crushing_df else None,
            "average_reduction_ratio": crushing_df["reduction_ratio"].mean() if "reduction_ratio" in crushing_df else None,
            "min_reduction_ratio": crushing_df["reduction_ratio"].min() if "reduction_ratio" in crushing_df else None,
            "max_reduction_ratio": crushing_df["reduction_ratio"].max() if "reduction_ratio" in crushing_df else None,
            "average_specific_energy": crushing_df["specific_energy"].mean() if "specific_energy" in crushing_df else None,
            "total_power": crushing_df["power_consumption"].sum() if "power_consumption" in crushing_df else None,
            "total_time": crushing_df["crush_time"].sum() if "crush_time" in crushing_df else None,
        }
        
        return summary
    
    def calculate_crushing_by_material(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate crushing metrics grouped by material type.
        
        Args:
            df: DataFrame containing the dataset.
            
        Returns:
            pd.DataFrame: Crushing metrics by material type.
        """
        crushing_df = self.calculate_crushing_metrics(df)
        
        if crushing_df.empty:
            return pd.DataFrame()
        
        # Group by material type
        grouped = crushing_df.groupby("material_type").agg({
            "sample_id": "count",
            "crushing_yield": "mean",
            "reduction_ratio": "mean",
            "specific_energy": "mean",
            "feed_mass": "sum",
        }).reset_index()
        
        grouped.columns = [
            "material_type", 
            "sample_count", 
            "avg_crushing_yield_percent",
            "avg_reduction_ratio",
            "avg_specific_energy_kwh_ton",
            "total_feed_mass_g"
        ]
        
        return grouped


# Convenience functions

def calculate_crushing_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate crushing metrics for all samples.
    
    Args:
        df: DataFrame containing the dataset.
        
    Returns:
        pd.DataFrame: DataFrame with calculated crushing metrics.
    """
    calculator = CrushingCalculator()
    return calculator.calculate_crushing_metrics(df)


def get_crushing_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get a summary of crushing analysis.
    
    Args:
        df: DataFrame containing the dataset.
        
    Returns:
        Dict[str, Any]: Summary statistics for crushing.
    """
    calculator = CrushingCalculator()
    return calculator.get_crushing_summary(df)


def calculate_crushing_by_material(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate crushing metrics grouped by material type.
    
    Args:
        df: DataFrame containing the dataset.
        
    Returns:
        pd.DataFrame: Crushing metrics by material type.
    """
    calculator = CrushingCalculator()
    return calculator.calculate_crushing_by_material(df)


def get_crushing_result_for_sample(df: pd.DataFrame, sample_id: str, 
                                   replicate: Optional[int] = None) -> Optional[CrushingResult]:
    """
    Get crushing results for a specific sample.
    
    Args:
        df: DataFrame containing the dataset.
        sample_id: Sample identifier.
        replicate: Optional replicate number.
        
    Returns:
        Optional[CrushingResult]: Crushing results for the sample.
    """
    calculator = CrushingCalculator()
    
    # Filter data for specific sample
    mask = df["sample_id"] == sample_id
    if replicate is not None:
        mask = mask & (df["replicate"] == replicate)
    
    sample_data = df[mask]
    
    if sample_data.empty:
        return None
    
    first_row = sample_data.iloc[0]
    result = calculator._calculate_sample_crushing(
        sample_id=sample_id,
        replicate=first_row.get("replicate", None),
        data=first_row
    )
    
    return result


# Export public API
__all__ = [
    'CrushingCalculator',
    'CrushingResult',
    'calculate_crushing_metrics',
    'get_crushing_summary',
    'calculate_crushing_by_material',
    'get_crushing_result_for_sample',
]