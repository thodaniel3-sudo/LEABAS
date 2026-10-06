"""
Beneficiation Calculation Module for LEBAS

This module provides calculations for mineral beneficiation analysis based on
the experimental data loaded from dataset.csv. All calculations operate on
the dataset values and do not hard-code experimental results.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from config import Config


@dataclass
class BeneficiationResult:
    """
    Container for beneficiation calculation results.
    """
    sample_id: str = ""
    material_type: str = ""
    replicate: Optional[int] = None
    feed_mass: Optional[float] = None
    feed_grade: Optional[float] = None
    concentrate_mass: Optional[float] = None
    concentrate_grade: Optional[float] = None
    tailings_mass: Optional[float] = None
    tailings_grade: Optional[float] = None
    losses_mass: Optional[float] = None
    
    # Calculated values
    feed_valuable_mass: Optional[float] = None
    concentrate_valuable_mass: Optional[float] = None
    tailings_valuable_mass: Optional[float] = None
    losses_valuable_mass: Optional[float] = None
    
    concentrate_yield: Optional[float] = None  # %
    tailings_yield: Optional[float] = None  # %
    recovery: Optional[float] = None  # %
    grade_recovery_product: Optional[float] = None  # Grade * Recovery / 100
    enrichment_ratio: Optional[float] = None
    mass_balance_closure: Optional[float] = None  # %
    separation_efficiency: Optional[float] = None  # %
    selectivity_index: Optional[float] = None
    
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the beneficiation result to a dictionary."""
        return {
            "sample_id": self.sample_id,
            "material_type": self.material_type,
            "replicate": self.replicate,
            "feed_mass": self.feed_mass,
            "feed_grade": self.feed_grade,
            "concentrate_mass": self.concentrate_mass,
            "concentrate_grade": self.concentrate_grade,
            "tailings_mass": self.tailings_mass,
            "tailings_grade": self.tailings_grade,
            "losses_mass": self.losses_mass,
            "feed_valuable_mass": self.feed_valuable_mass,
            "concentrate_valuable_mass": self.concentrate_valuable_mass,
            "tailings_valuable_mass": self.tailings_valuable_mass,
            "losses_valuable_mass": self.losses_valuable_mass,
            "concentrate_yield_percent": self.concentrate_yield,
            "tailings_yield_percent": self.tailings_yield,
            "recovery_percent": self.recovery,
            "grade_recovery_product": self.grade_recovery_product,
            "enrichment_ratio": self.enrichment_ratio,
            "mass_balance_closure_percent": self.mass_balance_closure,
            "separation_efficiency_percent": self.separation_efficiency,
            "selectivity_index": self.selectivity_index,
            "is_valid": self.is_valid,
            "warnings": self.warnings,
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert the beneficiation result to a DataFrame."""
        return pd.DataFrame([self.to_dict()])


class BeneficiationCalculator:
    """
    Performs mineral beneficiation calculations on the dataset.
    
    This class calculates various beneficiation metrics from the experimental data
    including yield, recovery, enrichment ratios, and mass balance.
    """
    
    def __init__(self):
        """Initialize the beneficiation calculator."""
        self.column_names = Config.COLUMN_NAMES
        
    def calculate_beneficiation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate beneficiation metrics for all samples in the dataset.
        
        Args:
            df: DataFrame containing the dataset.
            
        Returns:
            pd.DataFrame: DataFrame with calculated beneficiation metrics.
        """
        if df.empty:
            return pd.DataFrame()
        
        # Check required columns
        required_cols = [
            "sample_id", "replicate", "material_type",
            "feed_mass", "feed_grade",
            "concentrate_mass", "concentrate_grade",
            "tailings_mass", "tailings_grade"
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for beneficiation analysis: {missing_cols}")
        
        # Group by sample_id and replicate
        results = []
        
        for (sample_id, replicate), group in df.groupby(["sample_id", "replicate"]):
            # Get first row of the group
            first_row = group.iloc[0]
            
            result = self._calculate_sample_beneficiation(
                sample_id=sample_id,
                replicate=replicate,
                data=first_row
            )
            results.append(result.to_dict())
        
        return pd.DataFrame(results)
    
    def _calculate_sample_beneficiation(self, sample_id: str, replicate: int,
                                       data: pd.Series) -> BeneficiationResult:
        """
        Calculate beneficiation metrics for a single sample.
        
        Args:
            sample_id: Sample identifier.
            replicate: Replicate number.
            data: Series containing the sample data.
            
        Returns:
            BeneficiationResult: Calculated beneficiation metrics.
        """
        result = BeneficiationResult(
            sample_id=sample_id,
            replicate=replicate,
            material_type=data.get("material_type", ""),
        )
        
        # Get experimental values
        result.feed_mass = self._safe_float(data.get("feed_mass"))
        result.feed_grade = self._safe_float(data.get("feed_grade"))
        result.concentrate_mass = self._safe_float(data.get("concentrate_mass"))
        result.concentrate_grade = self._safe_float(data.get("concentrate_grade"))
        result.tailings_mass = self._safe_float(data.get("tailings_mass"))
        result.tailings_grade = self._safe_float(data.get("tailings_grade"))
        result.losses_mass = self._safe_float(data.get("losses_mass"))
        
        # Calculate valuable component masses
        self._calculate_valuable_masses(result)
        
        # Calculate yields
        self._calculate_yields(result)
        
        # Calculate recovery
        self._calculate_recovery(result)
        
        # Calculate enrichment ratio
        self._calculate_enrichment_ratio(result)
        
        # Calculate grade-recovery product
        self._calculate_grade_recovery_product(result)
        
        # Calculate mass balance closure
        self._calculate_mass_balance(result)
        
        # Calculate separation efficiency
        self._calculate_separation_efficiency(result)
        
        # Calculate selectivity index
        self._calculate_selectivity_index(result)
        
        # Validate results
        self._validate_result(result)
        
        return result
    
    def _calculate_valuable_masses(self, result: BeneficiationResult) -> None:
        """
        Calculate valuable component masses for all streams.
        
        Valuable component mass = mass × grade / 100
        
        Args:
            result: BeneficiationResult to update.
        """
        # Feed valuable mass
        if result.feed_mass is not None and result.feed_grade is not None:
            result.feed_valuable_mass = result.feed_mass * result.feed_grade / 100
        else:
            result.feed_valuable_mass = None
        
        # Concentrate valuable mass
        if result.concentrate_mass is not None and result.concentrate_grade is not None:
            result.concentrate_valuable_mass = result.concentrate_mass * result.concentrate_grade / 100
        else:
            result.concentrate_valuable_mass = None
        
        # Tailings valuable mass
        if result.tailings_mass is not None and result.tailings_grade is not None:
            result.tailings_valuable_mass = result.tailings_mass * result.tailings_grade / 100
        else:
            result.tailings_valuable_mass = None
        
        # Losses valuable mass (if losses mass and feed grade available)
        if result.losses_mass is not None and result.feed_grade is not None:
            result.losses_valuable_mass = result.losses_mass * result.feed_grade / 100
        else:
            result.losses_valuable_mass = None
    
    def _calculate_yields(self, result: BeneficiationResult) -> None:
        """
        Calculate concentrate and tailings yields.
        
        Yield (%) = (Product Mass / Feed Mass) × 100
        
        Args:
            result: BeneficiationResult to update.
        """
        # Concentrate yield
        if result.feed_mass is not None and result.concentrate_mass is not None:
            if result.feed_mass > 0:
                result.concentrate_yield = (result.concentrate_mass / result.feed_mass) * 100
            else:
                result.concentrate_yield = None
        else:
            result.concentrate_yield = None
        
        # Tailings yield
        if result.feed_mass is not None and result.tailings_mass is not None:
            if result.feed_mass > 0:
                result.tailings_yield = (result.tailings_mass / result.feed_mass) * 100
            else:
                result.tailings_yield = None
        else:
            result.tailings_yield = None
    
    def _calculate_recovery(self, result: BeneficiationResult) -> None:
        """
        Calculate recovery of valuable component.
        
        Recovery (%) = (Valuable in Concentrate / Valuable in Feed) × 100
        
        Args:
            result: BeneficiationResult to update.
        """
        if (result.concentrate_valuable_mass is not None and 
            result.feed_valuable_mass is not None):
            if result.feed_valuable_mass > 0:
                result.recovery = (result.concentrate_valuable_mass / result.feed_valuable_mass) * 100
            else:
                result.recovery = None
        else:
            result.recovery = None
    
    def _calculate_enrichment_ratio(self, result: BeneficiationResult) -> None:
        """
        Calculate enrichment ratio.
        
        Enrichment Ratio = Concentrate Grade / Feed Grade
        
        Args:
            result: BeneficiationResult to update.
        """
        if (result.concentrate_grade is not None and 
            result.feed_grade is not None):
            if result.feed_grade > 0:
                result.enrichment_ratio = result.concentrate_grade / result.feed_grade
            else:
                result.enrichment_ratio = None
        else:
            result.enrichment_ratio = None
    
    def _calculate_grade_recovery_product(self, result: BeneficiationResult) -> None:
        """
        Calculate grade-recovery product (used for assessing flotation performance).
        
        Grade-Recovery Product = Grade × Recovery / 100
        
        Args:
            result: BeneficiationResult to update.
        """
        if (result.concentrate_grade is not None and 
            result.recovery is not None):
            result.grade_recovery_product = result.concentrate_grade * result.recovery / 100
        else:
            result.grade_recovery_product = None
    
    def _calculate_mass_balance(self, result: BeneficiationResult) -> None:
        """
        Calculate mass balance closure.
        
        Mass Balance Closure (%) = 
        (Concentrate + Tailings + Losses - Feed) / Feed × 100
        
        Args:
            result: BeneficiationResult to update.
        """
        # Check if all required values are available
        values = [
            result.feed_mass,
            result.concentrate_mass,
            result.tailings_mass
        ]
        
        if all(v is not None for v in values):
            feed = result.feed_mass
            outputs = result.concentrate_mass + result.tailings_mass
            
            # Add losses if available
            if result.losses_mass is not None:
                outputs += result.losses_mass
            
            if feed > 0:
                result.mass_balance_closure = ((outputs - feed) / feed) * 100
            else:
                result.mass_balance_closure = None
        else:
            result.mass_balance_closure = None
    
    def _calculate_separation_efficiency(self, result: BeneficiationResult) -> None:
        """
        Calculate separation efficiency.
        
        Separation Efficiency = (Recovery - Tailings Yield) × 100
        
        Args:
            result: BeneficiationResult to update.
        """
        if result.recovery is not None and result.tailings_yield is not None:
            # Ensure both are in percentage form (not decimal)
            result.separation_efficiency = result.recovery - result.tailings_yield
        else:
            result.separation_efficiency = None
    
    def _calculate_selectivity_index(self, result: BeneficiationResult) -> None:
        """
        Calculate selectivity index.
        
        Selectivity Index = Recovery / (100 - Recovery) × 100
        
        Args:
            result: BeneficiationResult to update.
        """
        if result.recovery is not None:
            if result.recovery < 100:
                result.selectivity_index = result.recovery / (100 - result.recovery) * 100
            else:
                result.selectivity_index = None
        else:
            result.selectivity_index = None
    
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
    
    def _validate_result(self, result: BeneficiationResult) -> None:
        """
        Validate beneficiation results and add warnings.
        
        Args:
            result: BeneficiationResult to validate.
        """
        # Check for negative masses
        mass_fields = [
            ("feed_mass", result.feed_mass),
            ("concentrate_mass", result.concentrate_mass),
            ("tailings_mass", result.tailings_mass),
            ("losses_mass", result.losses_mass),
        ]
        
        for field_name, value in mass_fields:
            if value is not None and value < 0:
                result.warnings.append(f"Negative {field_name}: {value:.2f}g")
                result.is_valid = False
        
        # Check for grades outside 0-100 range
        grade_fields = [
            ("feed_grade", result.feed_grade),
            ("concentrate_grade", result.concentrate_grade),
            ("tailings_grade", result.tailings_grade),
        ]
        
        for field_name, value in grade_fields:
            if value is not None and (value < 0 or value > 100):
                result.warnings.append(
                    f"{field_name} outside 0-100% range: {value:.2f}%"
                )
                result.is_valid = False
        
        # Check yield
        if result.concentrate_yield is not None:
            if result.concentrate_yield > 100:
                result.warnings.append(
                    f"Concentrate yield > 100%: {result.concentrate_yield:.2f}%"
                )
            elif result.concentrate_yield < 0:
                result.warnings.append(
                    f"Negative concentrate yield: {result.concentrate_yield:.2f}%"
                )
        
        # Check recovery
        if result.recovery is not None:
            if result.recovery > 100:
                result.warnings.append(
                    f"Recovery > 100%: {result.recovery:.2f}%"
                )
                result.is_valid = False
            elif result.recovery < 0:
                result.warnings.append(
                    f"Negative recovery: {result.recovery:.2f}%"
                )
                result.is_valid = False
        
        # Check mass balance closure
        if result.mass_balance_closure is not None:
            if abs(result.mass_balance_closure) > 10:
                result.warnings.append(
                    f"Poor mass balance closure: {result.mass_balance_closure:.2f}%"
                )
        
        # Check enrichment ratio
        if result.enrichment_ratio is not None:
            if result.enrichment_ratio < 1:
                result.warnings.append(
                    f"Enrichment ratio < 1: {result.enrichment_ratio:.2f}"
                )
                result.is_valid = False
        
        # Check that concentrate grade > tailings grade
        if (result.concentrate_grade is not None and 
            result.tailings_grade is not None):
            if result.concentrate_grade < result.tailings_grade:
                result.warnings.append(
                    f"Concentrate grade ({result.concentrate_grade:.2f}%) < "
                    f"Tailings grade ({result.tailings_grade:.2f}%)"
                )
                result.is_valid = False
        
        # Check feed grade is reasonable
        if result.feed_grade is not None:
            if result.feed_grade == 0:
                result.warnings.append("Feed grade is 0%")
        
        # Check for missing critical data
        if result.feed_mass is None:
            result.warnings.append("Missing feed mass")
            result.is_valid = False
        
        if result.feed_grade is None:
            result.warnings.append("Missing feed grade")
            result.is_valid = False
        
        if result.concentrate_mass is None:
            result.warnings.append("Missing concentrate mass")
        
        if result.concentrate_grade is None:
            result.warnings.append("Missing concentrate grade")
    
    def get_beneficiation_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of beneficiation analysis.
        
        Args:
            df: DataFrame containing the dataset.
            
        Returns:
            Dict[str, Any]: Summary statistics for beneficiation.
        """
        ben_df = self.calculate_beneficiation(df)
        
        if ben_df.empty:
            return {
                "total_samples": 0,
                "average_yield": None,
                "average_recovery": None,
                "min_recovery": None,
                "max_recovery": None,
                "average_enrichment": None,
                "average_mass_balance": None,
            }
        
        # Calculate summary statistics
        summary = {
            "total_samples": len(ben_df),
            "average_yield": ben_df["concentrate_yield_percent"].mean() if "concentrate_yield_percent" in ben_df else None,
            "average_recovery": ben_df["recovery_percent"].mean() if "recovery_percent" in ben_df else None,
            "min_recovery": ben_df["recovery_percent"].min() if "recovery_percent" in ben_df else None,
            "max_recovery": ben_df["recovery_percent"].max() if "recovery_percent" in ben_df else None,
            "average_enrichment": ben_df["enrichment_ratio"].mean() if "enrichment_ratio" in ben_df else None,
            "average_mass_balance": ben_df["mass_balance_closure_percent"].mean() if "mass_balance_closure_percent" in ben_df else None,
            "average_separation_efficiency": ben_df["separation_efficiency_percent"].mean() if "separation_efficiency_percent" in ben_df else None,
        }
        
        return summary
    
    def calculate_beneficiation_by_material(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate beneficiation metrics grouped by material type.
        
        Args:
            df: DataFrame containing the dataset.
            
        Returns:
            pd.DataFrame: Beneficiation metrics by material type.
        """
        ben_df = self.calculate_beneficiation(df)
        
        if ben_df.empty:
            return pd.DataFrame()
        
        # Group by material type
        grouped = ben_df.groupby("material_type").agg({
            "sample_id": "count",
            "concentrate_yield_percent": "mean",
            "recovery_percent": "mean",
            "enrichment_ratio": "mean",
            "feed_grade": "mean",
            "concentrate_grade": "mean",
            "tailings_grade": "mean",
            "mass_balance_closure_percent": "mean",
        }).reset_index()
        
        grouped.columns = [
            "material_type",
            "sample_count",
            "avg_concentrate_yield_percent",
            "avg_recovery_percent",
            "avg_enrichment_ratio",
            "avg_feed_grade_percent",
            "avg_concentrate_grade_percent",
            "avg_tailings_grade_percent",
            "avg_mass_balance_closure_percent",
        ]
        
        return grouped
    
    def calculate_recovery_curve_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate data for recovery curves (grade vs recovery).
        
        Args:
            df: DataFrame containing the dataset.
            
        Returns:
            pd.DataFrame: Grade-recovery curve data.
        """
        ben_df = self.calculate_beneficiation(df)
        
        if ben_df.empty:
            return pd.DataFrame()
        
        # Select relevant columns for recovery curve
        curve_data = ben_df[["sample_id", "material_type", "replicate",
                           "concentrate_grade", "recovery_percent"]].copy()
        
        # Sort by recovery for curve plotting
        curve_data = curve_data.sort_values("recovery_percent", ascending=True)
        
        return curve_data


# Convenience functions

def calculate_beneficiation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate beneficiation metrics for all samples.
    
    Args:
        df: DataFrame containing the dataset.
        
    Returns:
        pd.DataFrame: DataFrame with calculated beneficiation metrics.
    """
    calculator = BeneficiationCalculator()
    return calculator.calculate_beneficiation(df)


def get_beneficiation_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get a summary of beneficiation analysis.
    
    Args:
        df: DataFrame containing the dataset.
        
    Returns:
        Dict[str, Any]: Summary statistics for beneficiation.
    """
    calculator = BeneficiationCalculator()
    return calculator.get_beneficiation_summary(df)


def calculate_beneficiation_by_material(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate beneficiation metrics grouped by material type.
    
    Args:
        df: DataFrame containing the dataset.
        
    Returns:
        pd.DataFrame: Beneficiation metrics by material type.
    """
    calculator = BeneficiationCalculator()
    return calculator.calculate_beneficiation_by_material(df)


def calculate_recovery_curve_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate data for recovery curves (grade vs recovery).
    
    Args:
        df: DataFrame containing the dataset.
        
    Returns:
        pd.DataFrame: Grade-recovery curve data.
    """
    calculator = BeneficiationCalculator()
    return calculator.calculate_recovery_curve_data(df)


def get_beneficiation_result_for_sample(df: pd.DataFrame, sample_id: str,
                                       replicate: Optional[int] = None) -> Optional[BeneficiationResult]:
    """
    Get beneficiation results for a specific sample.
    
    Args:
        df: DataFrame containing the dataset.
        sample_id: Sample identifier.
        replicate: Optional replicate number.
        
    Returns:
        Optional[BeneficiationResult]: Beneficiation results for the sample.
    """
    calculator = BeneficiationCalculator()
    
    # Filter data for specific sample
    mask = df["sample_id"] == sample_id
    if replicate is not None:
        mask = mask & (df["replicate"] == replicate)
    
    sample_data = df[mask]
    
    if sample_data.empty:
        return None
    
    first_row = sample_data.iloc[0]
    result = calculator._calculate_sample_beneficiation(
        sample_id=sample_id,
        replicate=first_row.get("replicate", None),
        data=first_row
    )
    
    return result


# Export public API
__all__ = [
    'BeneficiationCalculator',
    'BeneficiationResult',
    'calculate_beneficiation',
    'get_beneficiation_summary',
    'calculate_beneficiation_by_material',
    'calculate_recovery_curve_data',
    'get_beneficiation_result_for_sample',
]