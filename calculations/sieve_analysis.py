"""
Sieve Analysis Calculation Module for LEBAS

This module provides calculations for particle size analysis based on sieve
analysis data loaded from dataset.csv. All calculations operate on the dataset
values and do not hard-code experimental results.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from config import Config


@dataclass
class SieveAnalysisResult:
    """
    Container for sieve analysis calculation results.
    """
    sample_id: str = ""
    material_type: str = ""
    replicate: Optional[int] = None
    sieve_size: Optional[float] = None
    mass_retained: Optional[float] = None
    total_sample_mass: Optional[float] = None
    percent_retained: Optional[float] = None
    cumulative_percent_retained: Optional[float] = None
    percent_passing: Optional[float] = None
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the sieve analysis result to a dictionary."""
        return {
            "sample_id": self.sample_id,
            "material_type": self.material_type,
            "replicate": self.replicate,
            "sieve_size": self.sieve_size,
            "mass_retained": self.mass_retained,
            "total_sample_mass": self.total_sample_mass,
            "percent_retained": self.percent_retained,
            "cumulative_percent_retained": self.cumulative_percent_retained,
            "percent_passing": self.percent_passing,
            "is_valid": self.is_valid,
            "warnings": self.warnings,
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert the sieve analysis result to a DataFrame."""
        return pd.DataFrame([self.to_dict()])


@dataclass
class ParticleSizeDistribution:
    """
    Container for particle size distribution results.
    """
    sample_id: str = ""
    replicate: Optional[int] = None
    total_mass: Optional[float] = None
    d10: Optional[float] = None  # 10% passing
    d20: Optional[float] = None  # 20% passing
    d50: Optional[float] = None  # 50% passing (median)
    d80: Optional[float] = None  # 80% passing
    d90: Optional[float] = None  # 90% passing
    d95: Optional[float] = None  # 95% passing
    d98: Optional[float] = None  # 98% passing
    uniformity_coefficient: Optional[float] = None  # D60/D10
    curvature_coefficient: Optional[float] = None  # (D30^2)/(D60*D10)
    distribution_span: Optional[float] = None  # (D90 - D10) / D50
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the particle size distribution to a dictionary."""
        return {
            "sample_id": self.sample_id,
            "replicate": self.replicate,
            "total_mass": self.total_mass,
            "d10": self.d10,
            "d20": self.d20,
            "d50": self.d50,
            "d80": self.d80,
            "d90": self.d90,
            "d95": self.d95,
            "d98": self.d98,
            "uniformity_coefficient": self.uniformity_coefficient,
            "curvature_coefficient": self.curvature_coefficient,
            "distribution_span": self.distribution_span,
            "is_valid": self.is_valid,
            "warnings": self.warnings,
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert the particle size distribution to a DataFrame."""
        return pd.DataFrame([self.to_dict()])


class SieveAnalysisCalculator:
    """
    Performs sieve analysis calculations on the dataset.
    
    This class calculates particle size distributions, cumulative percentages,
    and characteristic particle sizes from sieve analysis data.
    """
    
    def __init__(self):
        """Initialize the sieve analysis calculator."""
        self.column_names = Config.COLUMN_NAMES
        self.standard_sieve_sizes = Config.STANDARD_SIEVE_SIZES
    
    def calculate_sieve_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sieve analysis metrics for all samples in the dataset.
        
        Args:
            df: DataFrame containing the dataset.
            
        Returns:
            pd.DataFrame: DataFrame with calculated sieve analysis metrics.
        """
        if df.empty:
            return pd.DataFrame()
        
        # Check required columns
        required_cols = ["sample_id", "sieve_size", "mass_retained"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for sieve analysis: {missing_cols}")
        
        # Group by sample_id and replicate
        results = []
        
        for (sample_id, replicate), group in df.groupby(["sample_id", "replicate"]):
            # Get material type from first row
            material_type = group.iloc[0].get("material_type", "")
            
            # Calculate total sample mass (sum of mass retained on all sieves + pan)
            # Use feed_mass if available, otherwise sum of mass retained
            feed_mass = group.iloc[0].get("feed_mass")
            
            if pd.notna(feed_mass) and feed_mass > 0:
                total_mass = float(feed_mass)
            else:
                total_mass = float(group["mass_retained"].sum())
            
            # Sort by sieve size (descending - coarsest to finest)
            sorted_group = group.sort_values("sieve_size", ascending=False)
            
            # Calculate cumulative metrics
            cumulative_retained = 0.0
            results_for_sample = []
            
            for _, row in sorted_group.iterrows():
                sieve_size = float(row["sieve_size"])
                mass_retained = float(row["mass_retained"]) if pd.notna(row["mass_retained"]) else 0.0
                
                # Calculate percentage retained
                if total_mass > 0:
                    percent_retained = (mass_retained / total_mass) * 100
                else:
                    percent_retained = 0.0
                
                # Update cumulative retained
                cumulative_retained += percent_retained
                
                # Calculate percentage passing
                percent_passing = max(0, 100 - cumulative_retained)
                
                # Create result object
                result = SieveAnalysisResult(
                    sample_id=sample_id,
                    material_type=material_type,
                    replicate=replicate,
                    sieve_size=sieve_size,
                    mass_retained=mass_retained,
                    total_sample_mass=total_mass,
                    percent_retained=percent_retained,
                    cumulative_percent_retained=cumulative_retained,
                    percent_passing=percent_passing,
                )
                
                # Validate result
                self._validate_result(result)
                
                results_for_sample.append(result.to_dict())
            
            # Add to overall results
            results.extend(results_for_sample)
        
        return pd.DataFrame(results)
    
    def calculate_particle_size_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate particle size distribution metrics for all samples.
        
        Args:
            df: DataFrame containing the dataset.
            
        Returns:
            pd.DataFrame: DataFrame with particle size distribution metrics.
        """
        if df.empty:
            return pd.DataFrame()
        
        # Check required columns
        required_cols = ["sample_id", "sieve_size", "mass_retained"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for particle size distribution: {missing_cols}")
        
        results = []
        
        for (sample_id, replicate), group in df.groupby(["sample_id", "replicate"]):
            # Calculate PSD for this sample
            psd = self._calculate_sample_psd(sample_id, replicate, group)
            if psd is not None:
                results.append(psd.to_dict())
        
        return pd.DataFrame(results)
    
    def _calculate_sample_psd(self, sample_id: str, replicate: int, 
                              group: pd.DataFrame) -> Optional[ParticleSizeDistribution]:
        """
        Calculate particle size distribution for a single sample.
        
        Args:
            sample_id: Sample identifier.
            replicate: Replicate number.
            group: DataFrame containing sieve data for the sample.
            
        Returns:
            Optional[ParticleSizeDistribution]: Particle size distribution metrics.
        """
        # Get total mass
        feed_mass = group.iloc[0].get("feed_mass")
        if pd.notna(feed_mass) and feed_mass > 0:
            total_mass = float(feed_mass)
        else:
            total_mass = float(group["mass_retained"].sum())
        
        if total_mass <= 0:
            return None
        
        # Sort by sieve size (descending - coarsest to finest)
        sorted_group = group.sort_values("sieve_size", ascending=False)
        
        # Calculate cumulative passing
        cumulative_retained = 0.0
        cumulative_passing = {}
        
        for _, row in sorted_group.iterrows():
            sieve_size = float(row["sieve_size"])
            mass_retained = float(row["mass_retained"]) if pd.notna(row["mass_retained"]) else 0.0
            
            percent_retained = (mass_retained / total_mass) * 100
            cumulative_retained += percent_retained
            percent_passing = max(0, 100 - cumulative_retained)
            
            cumulative_passing[sieve_size] = percent_passing
        
        # Calculate characteristic particle sizes
        psd = ParticleSizeDistribution(
            sample_id=sample_id,
            replicate=replicate,
            total_mass=total_mass,
        )
        
        # Interpolate for D10, D20, D50, D80, D90, D95, D98
        psd.d10 = self._interpolate_percentile(cumulative_passing, 10)
        psd.d20 = self._interpolate_percentile(cumulative_passing, 20)
        psd.d50 = self._interpolate_percentile(cumulative_passing, 50)
        psd.d80 = self._interpolate_percentile(cumulative_passing, 80)
        psd.d90 = self._interpolate_percentile(cumulative_passing, 90)
        psd.d95 = self._interpolate_percentile(cumulative_passing, 95)
        psd.d98 = self._interpolate_percentile(cumulative_passing, 98)
        
        # Calculate uniformity coefficient (D60/D10)
        d60 = self._interpolate_percentile(cumulative_passing, 60)
        if psd.d10 is not None and d60 is not None and psd.d10 > 0:
            psd.uniformity_coefficient = d60 / psd.d10
        
        # Calculate curvature coefficient ((D30^2)/(D60*D10))
        d30 = self._interpolate_percentile(cumulative_passing, 30)
        if d60 is not None and psd.d10 is not None and d60 > 0 and psd.d10 > 0:
            if d30 is not None:
                psd.curvature_coefficient = (d30 ** 2) / (d60 * psd.d10)
        
        # Calculate distribution span ((D90 - D10) / D50)
        if psd.d90 is not None and psd.d10 is not None and psd.d50 is not None and psd.d50 > 0:
            psd.distribution_span = (psd.d90 - psd.d10) / psd.d50
        
        # Validate PSD
        self._validate_psd(psd)
        
        return psd
    
    def _interpolate_percentile(self, data: Dict[float, float], 
                               percentile: float) -> Optional[float]:
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
        
        # Sort by sieve size (ascending - finest to coarsest)
        sorted_sizes = sorted(data.keys())
        
        # If percentile is outside the range, return the nearest value
        if percentile >= data[sorted_sizes[0]]:
            return sorted_sizes[0]
        if percentile <= data[sorted_sizes[-1]]:
            return sorted_sizes[-1]
        
        # Find the interval containing the percentile
        for i in range(len(sorted_sizes) - 1):
            size1 = sorted_sizes[i]
            size2 = sorted_sizes[i + 1]
            p1 = data[size1]
            p2 = data[size2]
            
            # Check if percentile falls between p1 and p2
            if (p1 >= percentile >= p2) or (p1 <= percentile <= p2):
                # Linear interpolation (log scale for sieve sizes)
                if p2 != p1:
                    # Use log interpolation for better accuracy with sieve sizes
                    log_size1 = np.log10(size1)
                    log_size2 = np.log10(size2)
                    fraction = (percentile - p1) / (p2 - p1)
                    interpolated_log = log_size1 + fraction * (log_size2 - log_size1)
                    return 10 ** interpolated_log
                else:
                    return size1
        
        return None
    
    def _validate_result(self, result: SieveAnalysisResult) -> None:
        """
        Validate sieve analysis results and add warnings.
        
        Args:
            result: SieveAnalysisResult to validate.
        """
        # Check for negative mass
        if result.mass_retained is not None and result.mass_retained < 0:
            result.warnings.append(f"Negative mass retained: {result.mass_retained:.2f}g")
            result.is_valid = False
        
        # Check for total mass consistency
        if result.total_sample_mass is not None and result.total_sample_mass <= 0:
            result.warnings.append(f"Invalid total sample mass: {result.total_sample_mass:.2f}g")
            result.is_valid = False
        
        # Check for cumulative percentage exceeding 100%
        if result.cumulative_percent_retained is not None:
            if result.cumulative_percent_retained > 100.1:  # Allow small rounding errors
                result.warnings.append(
                    f"Cumulative retained exceeds 100%: {result.cumulative_percent_retained:.1f}%"
                )
                result.is_valid = False
        
        # Check for negative percentages
        if result.percent_passing is not None and result.percent_passing < 0:
            result.warnings.append(f"Negative percentage passing: {result.percent_passing:.1f}%")
            result.is_valid = False
    
    def _validate_psd(self, psd: ParticleSizeDistribution) -> None:
        """
        Validate particle size distribution results.
        
        Args:
            psd: ParticleSizeDistribution to validate.
        """
        # Check if D50 > D80 (should be the other way around)
        if psd.d50 is not None and psd.d80 is not None:
            if psd.d50 > psd.d80:
                psd.warnings.append(
                    f"D50 ({psd.d50:.2f}µm) > D80 ({psd.d80:.2f}µm) - inconsistent"
                )
                psd.is_valid = False
        
        # Check if D10 > D50 (should be the other way around)
        if psd.d10 is not None and psd.d50 is not None:
            if psd.d10 > psd.d50:
                psd.warnings.append(
                    f"D10 ({psd.d10:.2f}µm) > D50 ({psd.d50:.2f}µm) - inconsistent"
                )
                psd.is_valid = False
        
        # Check for unrealistic uniformity coefficient
        if psd.uniformity_coefficient is not None:
            if psd.uniformity_coefficient > 100:
                psd.warnings.append(
                    f"Very high uniformity coefficient: {psd.uniformity_coefficient:.2f}"
                )
            elif psd.uniformity_coefficient < 1:
                psd.warnings.append(
                    f"Uniformity coefficient less than 1: {psd.uniformity_coefficient:.2f}"
                )
    
    def get_particle_size_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of particle size analysis.
        
        Args:
            df: DataFrame containing the dataset.
            
        Returns:
            Dict[str, Any]: Summary statistics for particle size analysis.
        """
        psd_df = self.calculate_particle_size_distribution(df)
        
        if psd_df.empty:
            return {
                "total_samples": 0,
                "average_d10": None,
                "average_d50": None,
                "average_d80": None,
                "average_d90": None,
                "average_uniformity": None,
            }
        
        # Calculate summary statistics
        summary = {
            "total_samples": len(psd_df),
            "average_d10": psd_df["d10"].mean() if "d10" in psd_df else None,
            "average_d50": psd_df["d50"].mean() if "d50" in psd_df else None,
            "average_d80": psd_df["d80"].mean() if "d80" in psd_df else None,
            "average_d90": psd_df["d90"].mean() if "d90" in psd_df else None,
            "average_uniformity": psd_df["uniformity_coefficient"].mean() if "uniformity_coefficient" in psd_df else None,
            "min_d50": psd_df["d50"].min() if "d50" in psd_df else None,
            "max_d50": psd_df["d50"].max() if "d50" in psd_df else None,
        }
        
        return summary
    
    def calculate_sieve_by_material(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sieve analysis metrics grouped by material type.
        
        Args:
            df: DataFrame containing the dataset.
            
        Returns:
            pd.DataFrame: Sieve analysis metrics by material type.
        """
        sieve_df = self.calculate_sieve_analysis(df)
        
        if sieve_df.empty:
            return pd.DataFrame()
        
        # Group by material type and sieve size
        grouped = sieve_df.groupby(["material_type", "sieve_size"]).agg({
            "sample_id": "count",
            "mass_retained": "mean",
            "percent_retained": "mean",
            "cumulative_percent_retained": "mean",
            "percent_passing": "mean",
        }).reset_index()
        
        return grouped
    
    def get_psd_data_for_chart(self, df: pd.DataFrame, 
                               sample_id: Optional[str] = None,
                               replicate: Optional[int] = None) -> pd.DataFrame:
        """
        Get particle size distribution data formatted for charting.
        
        Args:
            df: DataFrame containing the dataset.
            sample_id: Optional sample ID to filter.
            replicate: Optional replicate number to filter.
            
        Returns:
            pd.DataFrame: PSD data ready for charting.
        """
        # Filter data if sample_id provided
        if sample_id is not None:
            df = df[df["sample_id"] == sample_id]
        
        if replicate is not None:
            df = df[df["replicate"] == replicate]
        
        if df.empty:
            return pd.DataFrame()
        
        # Calculate sieve analysis
        return self.calculate_sieve_analysis(df)


# Convenience functions

def calculate_sieve_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate sieve analysis metrics for all samples.
    
    Args:
        df: DataFrame containing the dataset.
        
    Returns:
        pd.DataFrame: DataFrame with calculated sieve analysis metrics.
    """
    calculator = SieveAnalysisCalculator()
    return calculator.calculate_sieve_analysis(df)


def calculate_particle_size_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate particle size distribution metrics for all samples.
    
    Args:
        df: DataFrame containing the dataset.
        
    Returns:
        pd.DataFrame: DataFrame with particle size distribution metrics.
    """
    calculator = SieveAnalysisCalculator()
    return calculator.calculate_particle_size_distribution(df)


def get_particle_size_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get a summary of particle size analysis.
    
    Args:
        df: DataFrame containing the dataset.
        
    Returns:
        Dict[str, Any]: Summary statistics for particle size analysis.
    """
    calculator = SieveAnalysisCalculator()
    return calculator.get_particle_size_summary(df)


def calculate_sieve_by_material(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate sieve analysis metrics grouped by material type.
    
    Args:
        df: DataFrame containing the dataset.
        
    Returns:
        pd.DataFrame: Sieve analysis metrics by material type.
    """
    calculator = SieveAnalysisCalculator()
    return calculator.calculate_sieve_by_material(df)


def get_psd_data_for_chart(df: pd.DataFrame, 
                           sample_id: Optional[str] = None,
                           replicate: Optional[int] = None) -> pd.DataFrame:
    """
    Get particle size distribution data formatted for charting.
    
    Args:
        df: DataFrame containing the dataset.
        sample_id: Optional sample ID to filter.
        replicate: Optional replicate number to filter.
        
    Returns:
        pd.DataFrame: PSD data ready for charting.
    """
    calculator = SieveAnalysisCalculator()
    return calculator.get_psd_data_for_chart(df, sample_id, replicate)


def get_p80_from_sample(df: pd.DataFrame, sample_id: str, 
                        replicate: Optional[int] = None) -> Optional[float]:
    """
    Get P80 (80% passing size) for a specific sample.
    
    Args:
        df: DataFrame containing the dataset.
        sample_id: Sample identifier.
        replicate: Optional replicate number.
        
    Returns:
        Optional[float]: P80 value in micrometres.
    """
    calculator = SieveAnalysisCalculator()
    
    # Filter data for specific sample
    mask = df["sample_id"] == sample_id
    if replicate is not None:
        mask = mask & (df["replicate"] == replicate)
    
    sample_data = df[mask]
    
    if sample_data.empty:
        return None
    
    # Calculate PSD for this sample
    psd_df = calculator.calculate_particle_size_distribution(sample_data)
    
    if psd_df.empty:
        return None
    
    return float(psd_df.iloc[0]["d80"]) if "d80" in psd_df.columns else None


def get_p50_from_sample(df: pd.DataFrame, sample_id: str, 
                        replicate: Optional[int] = None) -> Optional[float]:
    """
    Get P50 (50% passing size) for a specific sample.
    
    Args:
        df: DataFrame containing the dataset.
        sample_id: Sample identifier.
        replicate: Optional replicate number.
        
    Returns:
        Optional[float]: P50 value in micrometres.
    """
    calculator = SieveAnalysisCalculator()
    
    # Filter data for specific sample
    mask = df["sample_id"] == sample_id
    if replicate is not None:
        mask = mask & (df["replicate"] == replicate)
    
    sample_data = df[mask]
    
    if sample_data.empty:
        return None
    
    # Calculate PSD for this sample
    psd_df = calculator.calculate_particle_size_distribution(sample_data)
    
    if psd_df.empty:
        return None
    
    return float(psd_df.iloc[0]["d50"]) if "d50" in psd_df.columns else None


# Export public API
__all__ = [
    'SieveAnalysisCalculator',
    'SieveAnalysisResult',
    'ParticleSizeDistribution',
    'calculate_sieve_analysis',
    'calculate_particle_size_distribution',
    'get_particle_size_summary',
    'calculate_sieve_by_material',
    'get_psd_data_for_chart',
    'get_p80_from_sample',
    'get_p50_from_sample',
]