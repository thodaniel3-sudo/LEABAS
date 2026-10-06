"""
Dashboard UI Module for LEBAS - Enhanced Professional Version
"""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import logging

from config import Config
from services.data_loader import DataLoader, load_dataset, get_dataset_summary
from services.validation import validate_dataset
from calculations.beneficiation import calculate_beneficiation, get_beneficiation_summary
from calculations.material_balance import calculate_material_balance
from calculations.sieve_analysis import calculate_sieve_analysis

logger = logging.getLogger(__name__)


class ModernCard(tk.Frame):
    """Modern card widget with shadow and gradient effect."""
    
    def __init__(self, parent, color="#2E86AB", **kwargs):
        super().__init__(parent, bg='#252540', **kwargs)
        self.color = color
        
        # Create gradient effect
        self.gradient = tk.Frame(self, bg=color, height=4)
        self.gradient.pack(fill='x')
        
        self.content = tk.Frame(self, bg='#252540')
        self.content.pack(fill='both', expand=True, padx=15, pady=10)
        
        # Border
        self.configure(highlightthickness=1, highlightcolor='#3a3a5a')
        
    def add_content(self, widget):
        widget.pack(in_=self.content)


class DashboardView:
    """Main dashboard view with professional styling."""
    
    def __init__(self, parent, controller):
        self.parent = parent
        self.controller = controller
        self.data_loader = DataLoader()
        self.df = None
        
        self.frame = tk.Frame(parent, bg='#1a1a2e')
        self.frame.grid(row=0, column=0, sticky="nsew")
        
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        
        self._build_dashboard()
        self._load_data()
    
    def _build_dashboard(self):
        """Build the dashboard with professional styling."""
        # Header
        self._create_header()
        
        # Main content
        self.content_frame = tk.Frame(self.frame, bg='#1a1a2e')
        self.content_frame.pack(fill='both', expand=True, padx=25, pady=15)
        
        # KPI Cards
        self._create_kpi_cards()
        
        # Quick Navigation
        self._create_navigation_cards()
        
        # Data Summary
        self._create_data_summary()
        
        # Status bar
        self._create_status_bar()
    
    def _create_header(self):
        """Create professional header."""
        header = tk.Frame(self.frame, bg='#1a1a2e', height=60)
        header.pack(fill='x', padx=25, pady=(15, 0))
        header.pack_propagate(False)
        
        # Title with icon
        title = tk.Label(
            header,
            text="📊 Dashboard Overview",
            font=('Segoe UI', 22, 'bold'),
            bg='#1a1a2e',
            fg='#ffffff'
        )
        title.pack(side='left')
        
        # Right side info
        info_frame = tk.Frame(header, bg='#1a1a2e')
        info_frame.pack(side='right')
        
        # Last update time
        self.update_label = tk.Label(
            info_frame,
            text="⏱ Last update: Just now",
            font=('Segoe UI', 9),
            bg='#1a1a2e',
            fg='#6c6c80'
        )
        self.update_label.pack()
    
    def _create_kpi_cards(self):
        """Create professional KPI cards."""
        kpi_frame = tk.Frame(self.content_frame, bg='#1a1a2e')
        kpi_frame.pack(fill='x', pady=(0, 15))
        
        # KPI Variables
        self.kpi_vars = {
            'samples': tk.StringVar(value="0"),
            'feed_mass': tk.StringVar(value="0.0 kg"),
            'beneficiation_tests': tk.StringVar(value="0"),
            'avg_recovery': tk.StringVar(value="0.0%"),
            'avg_yield': tk.StringVar(value="0.0%"),
            'sieve_records': tk.StringVar(value="0"),
            'dataset_status': tk.StringVar(value="Not Loaded"),
            'material_types': tk.StringVar(value="0"),
        }
        
        # Create cards in grid
        card_configs = [
            ("📋 Total Samples", self.kpi_vars['samples'], "#2E86AB", "Samples analyzed"),
            ("⚖️ Total Feed Mass", self.kpi_vars['feed_mass'], "#27AE60", "All samples"),
            ("🧪 Beneficiation Tests", self.kpi_vars['beneficiation_tests'], "#F18F01", "Flotation runs"),
            ("📈 Avg Recovery", self.kpi_vars['avg_recovery'], "#A23B72", "Beneficiation"),
            ("📊 Avg Yield", self.kpi_vars['avg_yield'], "#E74C3C", "Concentrate"),
            ("🔬 Sieve Records", self.kpi_vars['sieve_records'], "#3498DB", "Particle size"),
            ("🏷️ Material Types", self.kpi_vars['material_types'], "#1ABC9C", "Unique"),
            ("📂 Dataset Status", self.kpi_vars['dataset_status'], "#2ECC71", "Ready"),
        ]
        
        # Create 4-column grid
        for idx, (label, var, color, sub) in enumerate(card_configs):
            row = idx // 4
            col = idx % 4
            
            card = tk.Frame(
                kpi_frame,
                bg='#252540',
                highlightthickness=1,
                highlightcolor='#3a3a5a'
            )
            card.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
            
            # Color bar
            bar = tk.Frame(card, bg=color, height=4)
            bar.pack(fill='x')
            
            # Content
            content = tk.Frame(card, bg='#252540')
            content.pack(fill='both', expand=True, padx=15, pady=12)
            
            # Value
            value = tk.Label(
                content,
                textvariable=var,
                font=('Segoe UI', 22, 'bold'),
                bg='#252540',
                fg=color
            )
            value.pack(anchor='w')
            
            # Label
            label_w = tk.Label(
                content,
                text=label,
                font=('Segoe UI', 10),
                bg='#252540',
                fg='#a8a8b3'
            )
            label_w.pack(anchor='w')
            
            # Subtitle
            sub_w = tk.Label(
                content,
                text=sub,
                font=('Segoe UI', 8),
                bg='#252540',
                fg='#6c6c80'
            )
            sub_w.pack(anchor='w')
        
        # Configure grid weights
        for i in range(4):
            kpi_frame.grid_columnconfigure(i, weight=1)
    
    def _create_navigation_cards(self):
        """Create navigation cards."""
        nav_frame = tk.Frame(self.content_frame, bg='#1a1a2e')
        nav_frame.pack(fill='x', pady=(0, 15))
        
        nav_label = tk.Label(
            nav_frame,
            text="Quick Navigation",
            font=('Segoe UI', 14, 'bold'),
            bg='#1a1a2e',
            fg='#ffffff'
        )
        nav_label.pack(anchor='w', pady=(0, 10))
        
        # Navigation grid
        grid_frame = tk.Frame(nav_frame, bg='#1a1a2e')
        grid_frame.pack(fill='x')
        
        nav_items = [
            ("🔨 Crushing", self._navigate_to_crushing, "#2E86AB"),
            ("⚙️ Grinding", self._navigate_to_grinding, "#27AE60"),
            ("📊 Sieve Analysis", self._navigate_to_sieve, "#F18F01"),
            ("🧪 Beneficiation", self._navigate_to_beneficiation, "#A23B72"),
            ("📈 Results", self._navigate_to_results, "#E74C3C"),
            ("📄 Reports", self._navigate_to_reports, "#3498DB"),
        ]
        
        for idx, (text, command, color) in enumerate(nav_items):
            btn = tk.Button(
                grid_frame,
                text=text,
                command=command,
                bg=color,
                fg='white',
                font=('Segoe UI', 10, 'bold'),
                relief='flat',
                padx=20,
                pady=12,
                cursor='hand2'
            )
            btn.grid(row=0, column=idx, padx=6, pady=5, sticky="ew")
            
            # Hover effect
            btn.bind('<Enter>', lambda e, b=btn, c=color: b.configure(bg=self._darken_color(c)))
            btn.bind('<Leave>', lambda e, b=btn, c=color: b.configure(bg=c))
        
        for i in range(len(nav_items)):
            grid_frame.grid_columnconfigure(i, weight=1)
    
    def _darken_color(self, hex_color):
        """Darken a hex color."""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        r = max(0, int(r * 0.8))
        g = max(0, int(g * 0.8))
        b = max(0, int(b * 0.8))
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def _create_data_summary(self):
        """Create data summary section."""
        summary_frame = tk.Frame(self.content_frame, bg='#1a1a2e')
        summary_frame.pack(fill='both', expand=True)
        
        summary_label = tk.Label(
            summary_frame,
            text="📋 Data Summary",
            font=('Segoe UI', 14, 'bold'),
            bg='#1a1a2e',
            fg='#ffffff'
        )
        summary_label.pack(anchor='w', pady=(0, 10))
        
        # Summary text with scrollbar
        text_frame = tk.Frame(summary_frame, bg='#1a1a2e')
        text_frame.pack(fill='both', expand=True)
        
        self.summary_text = tk.Text(
            text_frame,
            wrap=tk.WORD,
            font=('Segoe UI', 9),
            bg='#252540',
            fg='#c8c8d3',
            relief='flat',
            highlightthickness=1,
            highlightcolor='#3a3a5a',
            padx=15,
            pady=15
        )
        self.summary_text.pack(side='left', fill='both', expand=True)
        
        scrollbar = tk.Scrollbar(
            text_frame,
            orient='vertical',
            command=self.summary_text.yview,
            bg='#1a1a2e',
            troughcolor='#1a1a2e'
        )
        scrollbar.pack(side='right', fill='y')
        self.summary_text.configure(yscrollcommand=scrollbar.set)
        
        self.summary_text.insert('1.0', "Loading data...")
        self.summary_text.configure(state='disabled')
    
    def _create_status_bar(self):
        """Create status bar."""
        status_frame = tk.Frame(self.frame, bg='#252540', height=35)
        status_frame.pack(side='bottom', fill='x')
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            status_frame,
            text="✅ Ready",
            font=('Segoe UI', 9),
            bg='#252540',
            fg='#a8a8b3',
            padx=15
        )
        self.status_label.pack(side='left')
        
        # Progress indicator
        self.progress = tk.Frame(status_frame, bg='#1a1a2e', width=200, height=4)
        self.progress.pack(side='right', padx=15)
        self.progress.pack_propagate(False)
        
        self.progress_bar = tk.Frame(self.progress, bg='#2ecc71', width=0, height=4)
        self.progress_bar.pack(side='left')
    
    def _load_data(self):
        """Load data and update dashboard."""
        try:
            self._update_progress(20)
            self.status_label.config(text="⏳ Loading dataset...")
            
            self.df = load_dataset()
            self._update_progress(60)
            
            if self.df is not None and not self.df.empty:
                self._update_kpis()
                self._update_summary()
                self.kpi_vars['dataset_status'].set("✅ Loaded")
                self.status_label.config(text="✅ Dataset loaded successfully")
                self.update_label.config(text=f"⏱ Last update: {pd.Timestamp.now().strftime('%H:%M:%S')}")
                self._update_progress(100)
            else:
                self.kpi_vars['dataset_status'].set("❌ Not Loaded")
                self.status_label.config(text="⚠️ No data available")
                self._update_progress(0)
                
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            self.status_label.config(text=error_msg)
            self.kpi_vars['dataset_status'].set("❌ Error")
            self._update_progress(0)
            messagebox.showerror("Data Load Error", error_msg)
    
    def _update_progress(self, value):
        """Update progress bar."""
        width = (value / 100) * 200
        self.progress_bar.config(width=width)
        self.frame.update()
    
    def _update_kpis(self):
        """Update KPI values."""
        if self.df is None or self.df.empty:
            return
        
        try:
            if 'sample_id' in self.df.columns:
                self.kpi_vars['samples'].set(str(self.df['sample_id'].nunique()))
            
            if 'material_type' in self.df.columns:
                self.kpi_vars['material_types'].set(str(self.df['material_type'].nunique()))
            
            if 'feed_mass' in self.df.columns:
                total = self.df['feed_mass'].sum()
                self.kpi_vars['feed_mass'].set(f"{total/1000:.2f} kg")
            
            if 'beneficiation_method' in self.df.columns:
                tests = self.df['beneficiation_method'].notna().sum()
                self.kpi_vars['beneficiation_tests'].set(str(tests))
            
            if 'sieve_size' in self.df.columns:
                records = self.df['sieve_size'].notna().sum()
                self.kpi_vars['sieve_records'].set(str(records))
            
            try:
                ben_df = calculate_beneficiation(self.df)
                if not ben_df.empty:
                    if 'recovery_percent' in ben_df.columns:
                        avg = ben_df['recovery_percent'].mean()
                        self.kpi_vars['avg_recovery'].set(f"{avg:.1f}%")
                    if 'concentrate_yield_percent' in ben_df.columns:
                        avg = ben_df['concentrate_yield_percent'].mean()
                        self.kpi_vars['avg_yield'].set(f"{avg:.1f}%")
            except:
                pass
                
        except Exception as e:
            logger.error(f"Error updating KPIs: {str(e)}")
    
    def _update_summary(self):
        """Update data summary."""
        if self.df is None or self.df.empty:
            return
        
        try:
            parts = []
            parts.append("═" * 70)
            parts.append(f"  {'LEBAS Dataset Summary':^66}")
            parts.append("═" * 70)
            parts.append("")
            parts.append(f"  📊 Total Rows: {len(self.df)}")
            parts.append(f"  📋 Columns: {len(self.df.columns)}")
            parts.append("")
            
            if 'sample_id' in self.df.columns:
                parts.append(f"  🏷️ Unique Samples: {self.df['sample_id'].nunique()}")
            
            if 'material_type' in self.df.columns:
                materials = ', '.join(self.df['material_type'].unique())
                parts.append(f"  🪨 Material Types: {materials}")
            
            if 'replicate' in self.df.columns:
                parts.append(f"  🔄 Replicates: {self.df['replicate'].nunique()}")
            parts.append("")
            
            if 'feed_mass' in self.df.columns:
                total = self.df['feed_mass'].sum()
                avg = self.df['feed_mass'].mean()
                parts.append(f"  ⚖️ Total Feed Mass: {total:.2f} g")
                parts.append(f"  📊 Average Feed Mass: {avg:.2f} g")
            
            if 'feed_grade' in self.df.columns:
                avg = self.df['feed_grade'].mean()
                parts.append(f"  📈 Average Feed Grade: {avg:.1f}%")
            
            if 'concentrate_grade' in self.df.columns:
                avg = self.df['concentrate_grade'].mean()
                parts.append(f"  📈 Average Concentrate Grade: {avg:.1f}%")
            
            if 'sieve_size' in self.df.columns:
                sizes = self.df['sieve_size'].unique()
                parts.append(f"  🔬 Sieve Sizes: {len(sizes)} unique")
                parts.append(f"  📏 Sieve Range: {min(sizes):.0f} - {max(sizes):.0f} µm")
            
            parts.append("")
            parts.append("═" * 70)
            parts.append(f"  ✅ Dataset Status: Loaded Successfully")
            parts.append("═" * 70)
            
            self.summary_text.configure(state='normal')
            self.summary_text.delete('1.0', tk.END)
            self.summary_text.insert('1.0', '\n'.join(parts))
            self.summary_text.configure(state='disabled')
            
        except Exception as e:
            self.summary_text.configure(state='normal')
            self.summary_text.delete('1.0', tk.END)
            self.summary_text.insert('1.0', f"❌ Error generating summary:\n{str(e)}")
            self.summary_text.configure(state='disabled')
    
    def _navigate_to_crushing(self):
        if self.controller:
            self.controller.show_frame("CrushingView")
    
    def _navigate_to_grinding(self):
        if self.controller:
            self.controller.show_frame("GrindingView")
    
    def _navigate_to_sieve(self):
        if self.controller:
            self.controller.show_frame("SieveView")
    
    def _navigate_to_beneficiation(self):
        if self.controller:
            self.controller.show_frame("BeneficiationView")
    
    def _navigate_to_results(self):
        if self.controller:
            self.controller.show_frame("ResultsView")
    
    def _navigate_to_reports(self):
        if self.controller:
            self.controller.show_frame("ReportsView")
    
    def refresh(self):
        self._load_data()
    
    def get_frame(self):
        return self.frame