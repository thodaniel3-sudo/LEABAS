"""
Graph Generation Module for LEBAS
Creates professional graphs in separate windows for all analysis modules
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import logging

from config import Config

logger = logging.getLogger(__name__)


class GraphWindow:
    """Separate window for displaying graphs with professional styling."""
    
    def __init__(self, parent, title: str, fig: Figure, width: int = 1000, height: int = 700):
        """
        Initialize graph window.
        
        Args:
            parent: Parent widget
            title: Window title
            fig: Matplotlib figure
            width: Window width
            height: Window height
        """
        self.window = tk.Toplevel(parent)
        self.window.title(f"📊 {title}")
        self.window.geometry(f"{width}x{height}")
        self.window.configure(bg='#1a1a2e')
        self.window.minsize(600, 450)
        
        # Create canvas with dark background
        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=15, pady=15)
        
        # Button frame
        btn_frame = tk.Frame(self.window, bg='#1a1a2e')
        btn_frame.pack(fill='x', pady=(0, 15))
        
        # Close button
        close_btn = tk.Button(
            btn_frame,
            text="✕ Close Graph",
            command=self.window.destroy,
            bg='#e74c3c',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief='flat',
            padx=25,
            pady=8,
            cursor='hand2'
        )
        close_btn.pack()
        close_btn.bind('<Enter>', lambda e, b=close_btn: b.configure(bg='#c0392b'))
        close_btn.bind('<Leave>', lambda e, b=close_btn: b.configure(bg='#e74c3c'))
        
        # Save button
        save_btn = tk.Button(
            btn_frame,
            text="💾 Save as PNG",
            command=lambda: self._save_figure(fig),
            bg='#2E86AB',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief='flat',
            padx=25,
            pady=8,
            cursor='hand2'
        )
        save_btn.pack(side='left', padx=10)
        save_btn.bind('<Enter>', lambda e, b=save_btn: b.configure(bg='#3498db'))
        save_btn.bind('<Leave>', lambda e, b=save_btn: b.configure(bg='#2E86AB'))
        
        # Bind escape key to close
        self.window.bind('<Escape>', lambda e: self.window.destroy())
    
    def _save_figure(self, fig):
        """Save figure as PNG."""
        from tkinter import filedialog
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if filepath:
            fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='#1a1a2e')
            tk.messagebox.showinfo("Save Complete", f"Graph saved to:\n{filepath}")


class GraphGenerator:
    """
    Generates all graphs for LEBAS analysis modules.
    """
    
    def __init__(self):
        """Initialize graph generator with default settings."""
        self.chart_settings = Config.CHART_SETTINGS
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#27AE60',
            'accent': '#F18F01',
            'purple': '#A23B72',
            'red': '#E74C3C',
            'blue': '#3498DB',
            'green': '#2ECC71',
            'orange': '#E67E22',
            'pink': '#E74C8C',
            'teal': '#1ABC9C',
            'dark_bg': '#1a1a2e',
            'card_bg': '#252540',
            'text_light': '#c8c8d3',
            'text_muted': '#a8a8b3',
            'grid': '#3a3a5a'
        }
        self.palette = [self.colors['primary'], self.colors['secondary'], 
                       self.colors['accent'], self.colors['purple'], 
                       self.colors['red'], self.colors['blue']]
    
    def _setup_axes(self, ax):
        """Setup professional axes styling."""
        ax.set_facecolor(self.colors['card_bg'])
        ax.tick_params(colors=self.colors['text_muted'])
        ax.grid(True, alpha=0.2, color=self.colors['grid'])
        for spine in ax.spines.values():
            spine.set_color(self.colors['grid'])
    
    def _add_value_labels(self, ax, bars, fmt='{:.1f}%'):
        """Add value labels on bars."""
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       fmt.format(height), ha='center', va='bottom', 
                       fontsize=8, color=self.colors['text_light'])
    
    # =========================================================================
    # CRUSHING GRAPHS
    # =========================================================================
    
    def crushing_yield_by_material(self, df: pd.DataFrame) -> Figure:
        """Graph 1: Crushing Yield by Material Type."""
        fig = Figure(figsize=(8, 5), dpi=100, facecolor=self.colors['dark_bg'])
        ax = fig.add_subplot(111)
        self._setup_axes(ax)
        
        if df.empty or 'crushing_yield' not in df.columns:
            ax.text(0.5, 0.5, 'No crushing data available', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        if 'material_type' in df.columns:
            grouped = df.groupby('material_type')['crushing_yield'].mean()
            materials = grouped.index.tolist()
            yields = grouped.values
            
            bars = ax.bar(materials, yields, color=self.colors['primary'], alpha=0.8)
            self._add_value_labels(ax, bars)
            
            ax.set_xlabel('Material Type', color=self.colors['text_muted'])
            ax.set_ylabel('Crushing Yield (%)', color=self.colors['text_muted'])
            ax.set_title('Crushing Yield by Material Type', color='white', fontweight='bold')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig.tight_layout()
        return fig
    
    def reduction_ratio_distribution(self, df: pd.DataFrame) -> Figure:
        """Graph 2: Reduction Ratio Distribution."""
        fig = Figure(figsize=(8, 5), dpi=100, facecolor=self.colors['dark_bg'])
        ax = fig.add_subplot(111)
        self._setup_axes(ax)
        
        if df.empty or 'reduction_ratio' not in df.columns:
            ax.text(0.5, 0.5, 'No reduction ratio data available', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        valid = df['reduction_ratio'].dropna()
        if not valid.empty:
            ax.hist(valid, bins=10, color=self.colors['secondary'], alpha=0.7, 
                   edgecolor='white', linewidth=0.5)
            ax.axvline(valid.mean(), color=self.colors['red'], linestyle='--', 
                      linewidth=2, label=f'Mean: {valid.mean():.2f}')
            ax.axvline(valid.median(), color=self.colors['accent'], linestyle='--', 
                      linewidth=2, label=f'Median: {valid.median():.2f}')
            
            ax.set_xlabel('Reduction Ratio', color=self.colors['text_muted'])
            ax.set_ylabel('Frequency', color=self.colors['text_muted'])
            ax.set_title('Reduction Ratio Distribution', color='white', fontweight='bold')
            ax.legend(loc='upper right')
        
        fig.tight_layout()
        return fig
    
    def crushing_power_vs_feed_size(self, df: pd.DataFrame) -> Figure:
        """Graph 3: Power Consumption vs Feed Size."""
        fig = Figure(figsize=(8, 5), dpi=100, facecolor=self.colors['dark_bg'])
        ax = fig.add_subplot(111)
        self._setup_axes(ax)
        
        if df.empty or 'crush_power' not in df.columns or 'crush_feed_size' not in df.columns:
            ax.text(0.5, 0.5, 'Insufficient data for power vs feed size', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        # Get unique samples
        sample_data = df.groupby('sample_id').first()
        valid = sample_data[['crush_power', 'crush_feed_size']].dropna()
        
        if not valid.empty:
            ax.scatter(valid['crush_feed_size'], valid['crush_power'], 
                      s=100, color=self.colors['primary'], alpha=0.7, edgecolors='white')
            
            # Add trend line
            if len(valid) > 1:
                z = np.polyfit(valid['crush_feed_size'], valid['crush_power'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(valid['crush_feed_size'].min(), 
                                    valid['crush_feed_size'].max(), 100)
                ax.plot(x_line, p(x_line), '--', color=self.colors['accent'], 
                       linewidth=2, label='Trend')
                ax.legend(loc='best')
            
            ax.set_xlabel('Feed Size (mm)', color=self.colors['text_muted'])
            ax.set_ylabel('Power Consumption (kW)', color=self.colors['text_muted'])
            ax.set_title('Power Consumption vs Feed Size', color='white', fontweight='bold')
        
        fig.tight_layout()
        return fig
    
    # =========================================================================
    # GRINDING GRAPHS
    # =========================================================================
    
    def grinding_yield_by_media_ratio(self, df: pd.DataFrame) -> Figure:
        """Graph 4: Grinding Yield by Ball-to-Ore Ratio."""
        fig = Figure(figsize=(10, 6), dpi=100, facecolor=self.colors['dark_bg'])
        ax = fig.add_subplot(111)
        self._setup_axes(ax)
        
        if df.empty or 'grinding_yield' not in df.columns:
            ax.text(0.5, 0.5, 'No grinding data available', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        # Extract media ratio from sample_id
        try:
            df = df.copy()
            df['media_ratio'] = df['sample_id'].str.extract(r'GRIND-(\d+)')[0]
            df['grind_time'] = df['sample_id'].str.extract(r'-(\d+)$')[0].astype(float)
            
            if 'media_ratio' in df.columns and 'grind_time' in df.columns:
                pivot = df.pivot_table(values='grinding_yield', 
                                      index='grind_time', 
                                      columns='media_ratio', 
                                      aggfunc='mean')
                
                pivot.plot(kind='bar', ax=ax, color=self.palette[:len(pivot.columns)])
                ax.set_xlabel('Grinding Time (min)', color=self.colors['text_muted'])
                ax.set_ylabel('Grinding Yield (%)', color=self.colors['text_muted'])
                ax.set_title('Grinding Yield by Media Ratio and Time', 
                            color='white', fontweight='bold')
                ax.legend(title='Media Ratio (ball:ore)', loc='best')
                ax.tick_params(colors=self.colors['text_muted'])
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error parsing data: {str(e)}', 
                   ha='center', va='center', color=self.colors['text_muted'])
        
        fig.tight_layout()
        return fig
    
    def p80_vs_grinding_time(self, df: pd.DataFrame, psd_df: pd.DataFrame) -> Figure:
        """Graph 5: P80 vs Grinding Time."""
        fig = Figure(figsize=(10, 6), dpi=100, facecolor=self.colors['dark_bg'])
        ax = fig.add_subplot(111)
        self._setup_axes(ax)
        
        if psd_df.empty or 'd80' not in psd_df.columns:
            ax.text(0.5, 0.5, 'No PSD data available', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        try:
            psd_df = psd_df.copy()
            psd_df['grind_time'] = psd_df['sample_id'].str.extract(r'-(\d+)$')[0].astype(float)
            psd_df['media_ratio'] = psd_df['sample_id'].str.extract(r'GRIND-(\d+)')[0]
            
            for media in psd_df['media_ratio'].unique():
                data = psd_df[psd_df['media_ratio'] == media].sort_values('grind_time')
                if not data.empty:
                    ax.plot(data['grind_time'], data['d80'], 
                           'o-', label=f'{media}:1', linewidth=2, markersize=8)
            
            ax.set_xlabel('Grinding Time (min)', color=self.colors['text_muted'])
            ax.set_ylabel('P80 (µm)', color=self.colors['text_muted'])
            ax.set_title('P80 vs Grinding Time by Media Ratio', 
                        color='white', fontweight='bold')
            ax.legend(title='Media Ratio (ball:ore)', loc='best')
            ax.tick_params(colors=self.colors['text_muted'])
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', 
                   ha='center', va='center', color=self.colors['text_muted'])
        
        fig.tight_layout()
        return fig
    
    def specific_energy_vs_p80(self, df: pd.DataFrame, psd_df: pd.DataFrame) -> Figure:
        """Graph 6: Specific Energy vs P80."""
        fig = Figure(figsize=(8, 5), dpi=100, facecolor=self.colors['dark_bg'])
        ax = fig.add_subplot(111)
        self._setup_axes(ax)
        
        if df.empty or psd_df.empty:
            ax.text(0.5, 0.5, 'Insufficient data for energy vs P80', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        if 'specific_energy' not in df.columns or 'd80' not in psd_df.columns:
            ax.text(0.5, 0.5, 'Missing specific energy or P80 data', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        # Merge data
        try:
            merged = pd.merge(df, psd_df, on='sample_id', suffixes=('_grind', '_psd'))
            valid = merged[['specific_energy', 'd80']].dropna()
            
            if not valid.empty:
                ax.scatter(valid['specific_energy'], valid['d80'], 
                          s=100, color=self.colors['purple'], alpha=0.7, edgecolors='white')
                
                # Add trend line (power law)
                if len(valid) > 2:
                    x = valid['specific_energy']
                    y = valid['d80']
                    log_x = np.log(x)
                    log_y = np.log(y)
                    z = np.polyfit(log_x, log_y, 1)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    y_line = np.exp(z[1]) * (x_line ** z[0])
                    ax.plot(x_line, y_line, '--', color=self.colors['accent'], 
                           linewidth=2, label='Power Law Fit')
                    ax.legend(loc='best')
                
                ax.set_xlabel('Specific Energy (kWh/ton)', color=self.colors['text_muted'])
                ax.set_ylabel('P80 (µm)', color=self.colors['text_muted'])
                ax.set_title('Specific Energy vs P80', color='white', fontweight='bold')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', 
                   ha='center', va='center', color=self.colors['text_muted'])
        
        fig.tight_layout()
        return fig
    
    # =========================================================================
    # SIEVE ANALYSIS GRAPHS
    # =========================================================================
    
    def particle_size_distribution(self, df: pd.DataFrame, sample_id: str = None) -> Figure:
        """Graph 7: Particle Size Distribution Curve."""
        fig = Figure(figsize=(9, 5), dpi=100, facecolor=self.colors['dark_bg'])
        ax = fig.add_subplot(111)
        self._setup_axes(ax)
        
        if df.empty or 'sieve_size' not in df.columns:
            ax.text(0.5, 0.5, 'No sieve data available', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        # Filter for specific sample or use all
        if sample_id:
            data = df[df['sample_id'] == sample_id]
            title = f'PSD - Sample: {sample_id}'
        else:
            data = df
            title = 'Particle Size Distribution'
        
        if data.empty:
            ax.text(0.5, 0.5, f'No data for sample {sample_id}', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        # Group by sample
        for sample, group in data.groupby('sample_id'):
            if len(group) > 1:
                sorted_group = group.sort_values('sieve_size', ascending=False)
                if 'percent_passing' in sorted_group.columns:
                    ax.plot(sorted_group['sieve_size'], sorted_group['percent_passing'],
                           'o-', label=sample[:10], linewidth=2, markersize=6)
        
        ax.set_xscale('log')
        ax.set_xlabel('Sieve Size (µm)', color=self.colors['text_muted'])
        ax.set_ylabel('Percent Passing (%)', color=self.colors['text_muted'])
        ax.set_title(title, color='white', fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.set_ylim(0, 105)
        
        fig.tight_layout()
        return fig
    
    def psd_comparison(self, df: pd.DataFrame, sample_ids: List[str] = None) -> Figure:
        """Graph 8: PSD Comparison - Multiple Samples."""
        fig = Figure(figsize=(10, 6), dpi=100, facecolor=self.colors['dark_bg'])
        ax = fig.add_subplot(111)
        self._setup_axes(ax)
        
        if df.empty or 'sieve_size' not in df.columns:
            ax.text(0.5, 0.5, 'No sieve data available', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        if sample_ids:
            data = df[df['sample_id'].isin(sample_ids)]
        else:
            # Get top 5 samples by row count
            top_samples = df['sample_id'].value_counts().head(5).index.tolist()
            data = df[df['sample_id'].isin(top_samples)]
        
        if data.empty:
            ax.text(0.5, 0.5, 'No samples to compare', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        colors = ['#2E86AB', '#27AE60', '#F18F01', '#A23B72', '#E74C3C']
        for idx, (sample, group) in enumerate(data.groupby('sample_id')):
            sorted_group = group.sort_values('sieve_size', ascending=False)
            if 'percent_passing' in sorted_group.columns:
                ax.plot(sorted_group['sieve_size'], sorted_group['percent_passing'],
                       'o-', label=sample, color=colors[idx % len(colors)], 
                       linewidth=2, markersize=6)
        
        ax.set_xscale('log')
        ax.set_xlabel('Sieve Size (µm)', color=self.colors['text_muted'])
        ax.set_ylabel('Percent Passing (%)', color=self.colors['text_muted'])
        ax.set_title('Particle Size Distribution Comparison', 
                    color='white', fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.set_ylim(0, 105)
        
        fig.tight_layout()
        return fig
    
    def psd_histogram(self, df: pd.DataFrame, sample_id: str = None) -> Figure:
        """Graph 9: Particle Size Distribution Histogram."""
        fig = Figure(figsize=(9, 5), dpi=100, facecolor=self.colors['dark_bg'])
        ax = fig.add_subplot(111)
        self._setup_axes(ax)
        
        if df.empty or 'mass_retained' not in df.columns:
            ax.text(0.5, 0.5, 'No sieve data available', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        if sample_id:
            data = df[df['sample_id'] == sample_id]
            title = f'PSD Histogram - {sample_id}'
        else:
            data = df
            title = 'Particle Size Distribution Histogram'
        
        if data.empty:
            ax.text(0.5, 0.5, f'No data for sample {sample_id}', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        sorted_data = data.sort_values('sieve_size', ascending=True)
        if 'mass_retained' in sorted_data.columns and 'sieve_size' in sorted_data.columns:
            bars = ax.bar(sorted_data['sieve_size'].astype(str), 
                         sorted_data['mass_retained'],
                         color=self.colors['primary'], alpha=0.7)
            self._add_value_labels(ax, bars, '{:.1f}g')
            
            ax.set_xlabel('Sieve Size (µm)', color=self.colors['text_muted'])
            ax.set_ylabel('Mass Retained (g)', color=self.colors['text_muted'])
            ax.set_title(title, color='white', fontweight='bold')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig.tight_layout()
        return fig
    
    # =========================================================================
    # BENEFICIATION GRAPHS
    # =========================================================================
    
    def recovery_by_material(self, df: pd.DataFrame) -> Figure:
        """Graph 10: Recovery by Material Type."""
        fig = Figure(figsize=(8, 5), dpi=100, facecolor=self.colors['dark_bg'])
        ax = fig.add_subplot(111)
        self._setup_axes(ax)
        
        if df.empty or 'recovery_percent' not in df.columns:
            ax.text(0.5, 0.5, 'No recovery data available', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        if 'material_type' in df.columns:
            grouped = df.groupby('material_type')['recovery_percent']
            stats = grouped.agg(['mean', 'std'])
            
            bars = ax.bar(stats.index, stats['mean'], 
                         yerr=stats['std'], capsize=5,
                         color=self.colors['secondary'], alpha=0.8,
                         error_kw={'ecolor': self.colors['red']})
            self._add_value_labels(ax, bars)
            
            ax.set_xlabel('Material Type', color=self.colors['text_muted'])
            ax.set_ylabel('Recovery (%)', color=self.colors['text_muted'])
            ax.set_title('Recovery by Material Type', color='white', fontweight='bold')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig.tight_layout()
        return fig
    
    def grade_vs_recovery(self, df: pd.DataFrame) -> Figure:
        """Graph 11: Grade vs Recovery Curve."""
        fig = Figure(figsize=(9, 6), dpi=100, facecolor=self.colors['dark_bg'])
        ax = fig.add_subplot(111)
        self._setup_axes(ax)
        
        if df.empty or 'concentrate_grade' not in df.columns or 'recovery_percent' not in df.columns:
            ax.text(0.5, 0.5, 'Insufficient data for grade vs recovery', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        # Color by material type
        if 'material_type' in df.columns:
            for material, group in df.groupby('material_type'):
                if not group.empty:
                    ax.scatter(group['concentrate_grade'], group['recovery_percent'],
                             label=material, s=100, alpha=0.7, edgecolors='white')
        else:
            ax.scatter(df['concentrate_grade'], df['recovery_percent'],
                      s=100, color=self.colors['primary'], alpha=0.7, edgecolors='white')
        
        # Add optimal region
        max_grade = df['concentrate_grade'].max()
        max_recovery = df['recovery_percent'].max()
        if max_grade and max_recovery:
            rect = mpatches.Rectangle((max_grade * 0.8, max_recovery * 0.8),
                                     max_grade * 0.2, max_recovery * 0.2,
                                     facecolor='green', alpha=0.1,
                                     edgecolor='green', linestyle='--', linewidth=1)
            ax.add_patch(rect)
            ax.text(max_grade * 0.9, max_recovery * 0.9,
                   'Optimal\nRegion', ha='center', va='center',
                   fontsize=10, color='green', alpha=0.7)
        
        ax.set_xlabel('Concentrate Grade (%)', color=self.colors['text_muted'])
        ax.set_ylabel('Recovery (%)', color=self.colors['text_muted'])
        ax.set_title('Grade vs Recovery', color='white', fontweight='bold')
        if 'material_type' in df.columns:
            ax.legend(loc='best', fontsize=8)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        
        fig.tight_layout()
        return fig
    
    def recovery_vs_yield_tradeoff(self, df: pd.DataFrame) -> Figure:
        """Graph 12: Recovery vs Yield Trade-off."""
        fig = Figure(figsize=(9, 6), dpi=100, facecolor=self.colors['dark_bg'])
        ax = fig.add_subplot(111)
        self._setup_axes(ax)
        
        if df.empty or 'concentrate_yield_percent' not in df.columns or 'recovery_percent' not in df.columns:
            ax.text(0.5, 0.5, 'Insufficient data for trade-off analysis', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        if 'material_type' in df.columns:
            for material, group in df.groupby('material_type'):
                if not group.empty:
                    ax.scatter(group['concentrate_yield_percent'], group['recovery_percent'],
                             label=material, s=100, alpha=0.7, edgecolors='white')
        else:
            ax.scatter(df['concentrate_yield_percent'], df['recovery_percent'],
                      s=100, color=self.colors['purple'], alpha=0.7, edgecolors='white')
        
        # Add sample labels
        for idx, row in df.iterrows():
            label = row.get('sample_id', '')
            if label:
                ax.annotate(label, (row['concentrate_yield_percent'], row['recovery_percent']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=7, color=self.colors['text_muted'], alpha=0.7)
        
        ax.set_xlabel('Concentrate Yield (%)', color=self.colors['text_muted'])
        ax.set_ylabel('Recovery (%)', color=self.colors['text_muted'])
        ax.set_title('Recovery vs Yield Trade-off', color='white', fontweight='bold')
        if 'material_type' in df.columns:
            ax.legend(loc='best', fontsize=8)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        
        fig.tight_layout()
        return fig
    
    def enrichment_comparison(self, df: pd.DataFrame) -> Figure:
        """Graph 13: Enrichment Ratio Comparison."""
        fig = Figure(figsize=(8, 5), dpi=100, facecolor=self.colors['dark_bg'])
        ax = fig.add_subplot(111)
        self._setup_axes(ax)
        
        if df.empty or 'enrichment_ratio' not in df.columns:
            ax.text(0.5, 0.5, 'No enrichment data available', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        if 'material_type' in df.columns:
            grouped = df.groupby('material_type')['enrichment_ratio'].mean()
            materials = grouped.index.tolist()
            enrichment = grouped.values
            
            bars = ax.bar(materials, enrichment, color=self.colors['accent'], alpha=0.8)
            self._add_value_labels(ax, bars, '{:.2f}x')
            
            ax.axhline(y=1.5, color=self.colors['red'], linestyle='--', 
                      linewidth=2, alpha=0.7, label='Good Enrichment (>1.5x)')
            
            ax.set_xlabel('Material Type', color=self.colors['text_muted'])
            ax.set_ylabel('Enrichment Ratio', color=self.colors['text_muted'])
            ax.set_title('Enrichment Ratio by Material Type', 
                        color='white', fontweight='bold')
            ax.legend(loc='best')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig.tight_layout()
        return fig
    
    def grade_comparison(self, df: pd.DataFrame) -> Figure:
        """Graph 14: Feed vs Concentrate vs Tailings Grade."""
        fig = Figure(figsize=(10, 6), dpi=100, facecolor=self.colors['dark_bg'])
        ax = fig.add_subplot(111)
        self._setup_axes(ax)
        
        required = ['feed_grade', 'concentrate_grade', 'tailings_grade']
        if df.empty or not all(col in df.columns for col in required):
            ax.text(0.5, 0.5, 'Insufficient grade data', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        # Get unique samples
        sample_data = df.groupby('sample_id').first()
        sample_data = sample_data[required].dropna()
        
        if sample_data.empty:
            ax.text(0.5, 0.5, 'No complete grade data available', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        x = np.arange(len(sample_data))
        width = 0.25
        
        bars1 = ax.bar(x - width, sample_data['feed_grade'], width, 
                      label='Feed Grade', color=self.colors['blue'], alpha=0.8)
        bars2 = ax.bar(x, sample_data['concentrate_grade'], width, 
                      label='Concentrate Grade', color=self.colors['secondary'], alpha=0.8)
        bars3 = ax.bar(x + width, sample_data['tailings_grade'], width, 
                      label='Tailings Grade', color=self.colors['red'], alpha=0.8)
        
        ax.set_xlabel('Sample ID', color=self.colors['text_muted'])
        ax.set_ylabel('Grade (%)', color=self.colors['text_muted'])
        ax.set_title('Grade Comparison: Feed vs Concentrate vs Tailings', 
                    color='white', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sample_data.index.tolist(), rotation=45, ha='right')
        ax.legend(loc='best')
        ax.set_ylim(0, 105)
        
        fig.tight_layout()
        return fig
    
    # =========================================================================
    # MATERIAL BALANCE GRAPHS
    # =========================================================================
    
    def mass_balance_closure(self, df: pd.DataFrame) -> Figure:
        """Graph 15: Mass Balance Closure."""
        fig = Figure(figsize=(9, 5), dpi=100, facecolor=self.colors['dark_bg'])
        ax = fig.add_subplot(111)
        self._setup_axes(ax)
        
        if df.empty or 'mass_balance_error' not in df.columns:
            ax.text(0.5, 0.5, 'No mass balance data available', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        # Get unique samples
        sample_data = df.groupby('sample_id').first()
        sample_data = sample_data[['mass_balance_error', 'mass_balance_closure']].dropna()
        
        if sample_data.empty:
            ax.text(0.5, 0.5, 'No complete mass balance data', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        x = np.arange(len(sample_data))
        width = 0.35
        
        # Color based on error magnitude
        colors = []
        for err in sample_data['mass_balance_error']:
            if abs(err) < 2:
                colors.append(self.colors['secondary'])
            elif abs(err) < 5:
                colors.append(self.colors['accent'])
            else:
                colors.append(self.colors['red'])
        
        bars = ax.bar(x, sample_data['mass_balance_error'], width, color=colors, alpha=0.8)
        
        ax.axhline(y=2, color=self.colors['secondary'], linestyle='--', 
                  alpha=0.5, label='Good (<2%)')
        ax.axhline(y=5, color=self.colors['accent'], linestyle='--', 
                  alpha=0.5, label='Acceptable (<5%)')
        
        ax.set_xlabel('Sample ID', color=self.colors['text_muted'])
        ax.set_ylabel('Mass Balance Error (%)', color=self.colors['text_muted'])
        ax.set_title('Mass Balance Closure', color='white', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sample_data.index.tolist(), rotation=45, ha='right')
        ax.legend(loc='best')
        ax.set_ylim(-10, 10)
        
        fig.tight_layout()
        return fig
    
    def mass_distribution_pie(self, df: pd.DataFrame, sample_id: str = None) -> Figure:
        """Graph 16: Mass Distribution Pie Chart."""
        fig = Figure(figsize=(8, 6), dpi=100, facecolor=self.colors['dark_bg'])
        ax = fig.add_subplot(111)
        self._setup_axes(ax)
        
        if df.empty:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        if sample_id:
            data = df[df['sample_id'] == sample_id]
        else:
            data = df.head(1)
        
        if data.empty:
            ax.text(0.5, 0.5, f'No data for sample {sample_id}', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        row = data.iloc[0]
        masses = []
        labels = []
        colors = []
        
        if 'concentrate_mass' in row and pd.notna(row['concentrate_mass']):
            masses.append(row['concentrate_mass'])
            labels.append('Concentrate')
            colors.append(self.colors['secondary'])
        
        if 'tailings_mass' in row and pd.notna(row['tailings_mass']):
            masses.append(row['tailings_mass'])
            labels.append('Tailings')
            colors.append(self.colors['red'])
        
        if 'losses_mass' in row and pd.notna(row['losses_mass']) and row['losses_mass'] > 0:
            masses.append(row['losses_mass'])
            labels.append('Losses')
            colors.append(self.colors['accent'])
        
        if not masses:
            ax.text(0.5, 0.5, 'No mass data available', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        # Filter out zero masses
        valid = [(m, l, c) for m, l, c in zip(masses, labels, colors) if m > 0]
        if not valid:
            ax.text(0.5, 0.5, 'All masses are zero', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        m, l, c = zip(*valid)
        
        wedges, texts, autotexts = ax.pie(m, labels=l, colors=c, autopct='%1.1f%%',
                                         startangle=90, explode=[0.02]*len(m))
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        
        ax.set_title(f'Mass Distribution - {sample_id if sample_id else "Sample"}', 
                    color='white', fontweight='bold')
        
        fig.tight_layout()
        return fig
    
    # =========================================================================
    # STATISTICS GRAPHS
    # =========================================================================
    
    def parameter_distribution(self, df: pd.DataFrame, column: str) -> Figure:
        """Graph 17: Parameter Distribution with Statistics."""
        fig = Figure(figsize=(8, 5), dpi=100, facecolor=self.colors['dark_bg'])
        ax = fig.add_subplot(111)
        self._setup_axes(ax)
        
        if column not in df.columns:
            ax.text(0.5, 0.5, f'Column "{column}" not found', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        valid = df[column].dropna()
        if valid.empty:
            ax.text(0.5, 0.5, f'No valid data in "{column}"', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        # Histogram
        n, bins, patches = ax.hist(valid, bins=15, color=self.colors['primary'], 
                                   alpha=0.7, edgecolor='white', linewidth=0.5)
        
        # Statistics lines
        mean = valid.mean()
        median = valid.median()
        std = valid.std()
        
        ax.axvline(mean, color=self.colors['red'], linestyle='--', 
                  linewidth=2, label=f'Mean: {mean:.2f}')
        ax.axvline(median, color=self.colors['accent'], linestyle='--', 
                  linewidth=2, label=f'Median: {median:.2f}')
        ax.axvline(mean - std, color=self.colors['blue'], linestyle=':', 
                  linewidth=1.5, label=f'±1 Std Dev')
        ax.axvline(mean + std, color=self.colors['blue'], linestyle=':', 
                  linewidth=1.5)
        
        ax.set_xlabel(column.replace('_', ' ').title(), color=self.colors['text_muted'])
        ax.set_ylabel('Frequency', color=self.colors['text_muted'])
        ax.set_title(f'Distribution of {column.replace("_", " ").title()}', 
                    color='white', fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        
        fig.tight_layout()
        return fig
    
    def boxplot_comparison(self, df: pd.DataFrame, column: str, group_by: str = 'material_type') -> Figure:
        """Graph 18: Box Plot Comparison by Group."""
        fig = Figure(figsize=(10, 6), dpi=100, facecolor=self.colors['dark_bg'])
        ax = fig.add_subplot(111)
        self._setup_axes(ax)
        
        if column not in df.columns or group_by not in df.columns:
            ax.text(0.5, 0.5, 'Required columns not found', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        # Prepare data for boxplot
        data_to_plot = []
        labels = []
        
        for group, group_data in df.groupby(group_by):
            values = group_data[column].dropna()
            if not values.empty:
                data_to_plot.append(values.values)
                labels.append(str(group))
        
        if not data_to_plot:
            ax.text(0.5, 0.5, 'No data to plot', 
                   ha='center', va='center', color=self.colors['text_muted'])
            fig.tight_layout()
            return fig
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                       widths=0.6)
        
        # Color the boxes
        colors = self.palette[:len(data_to_plot)]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add individual points
        for i, data in enumerate(data_to_plot):
            x = np.random.normal(i + 1, 0.04, size=len(data))
            ax.plot(x, data, 'o', alpha=0.3, color='gray', markersize=3)
        
        ax.set_xlabel(group_by.replace('_', ' ').title(), color=self.colors['text_muted'])
        ax.set_ylabel(column.replace('_', ' ').title(), color=self.colors['text_muted'])
        ax.set_title(f'{column.replace("_", " ").title()} by {group_by.replace("_", " ").title()}', 
                    color='white', fontweight='bold')
        ax.tick_params(colors=self.colors['text_muted'])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig.tight_layout()
        return fig


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def show_graph(parent, graph_func, *args, **kwargs):
    """
    Convenience function to show a graph in a new window.
    
    Args:
        parent: Parent widget
        graph_func: Graph generation function
        *args: Arguments for graph function
        **kwargs: Keyword arguments for graph function
    """
    generator = GraphGenerator()
    fig = graph_func(generator, *args, **kwargs)
    title = kwargs.get('title', 'LEBAS Graph')
    GraphWindow(parent, title, fig)


# Export all functions
__all__ = [
    'GraphWindow',
    'GraphGenerator',
    'show_graph',
]