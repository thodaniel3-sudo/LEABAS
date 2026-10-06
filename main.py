"""
LEBAS - Laboratory Extraction and Beneficiation Analysis System
Main Application Entry Point - AUTO GRAPH POPUP VERSION
"""

import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
from pathlib import Path
import traceback
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config

# Import all UI views
from ui.dashboard import DashboardView
from ui.crushing_view import CrushingView
from ui.grinding_view import GrindingView
from ui.sieve_view import SieveView
from ui.beneficiation_view import BeneficiationView
from ui.results_view import ResultsView
from ui.reports_view import ReportsView

# Import graph module
from charts.graph import GraphGenerator, GraphWindow, show_graph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.PROJECT_ROOT / 'lebas.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModernNavButton(tk.Canvas):
    """Modern navigation button with hover effects."""
    
    def __init__(self, parent, text, icon="", command=None, **kwargs):
        self.command = command
        self.is_hovered = False
        self.is_active = False
        
        super().__init__(parent, highlightthickness=0, **kwargs)
        self.configure(bg='#1a1a2e', bd=0, height=50)
        
        self.text = text
        self.icon = icon
        
        # Create text
        self.text_id = self.create_text(
            15, 25, 
            text=f"{icon} {text}", 
            font=('Segoe UI', 10),
            fill='#a8a8b3',
            anchor='w'
        )
        
        # Bind events
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
        self.bind('<Button-1>', self._on_click)
        
        # Border line
        self.border_id = self.create_line(0, 48, 200, 48, fill='#2a2a4a', width=1)
        
    def _on_enter(self, event):
        self.is_hovered = True
        self.itemconfig(self.text_id, fill='#ffffff')
        self.configure(bg='#2a2a4a')
        
    def _on_leave(self, event):
        self.is_hovered = False
        if not self.is_active:
            self.itemconfig(self.text_id, fill='#a8a8b3')
            self.configure(bg='#1a1a2e')
        
    def _on_click(self, event):
        if self.command:
            self.command()
    
    def set_active(self, active=True):
        self.is_active = active
        if active:
            self.itemconfig(self.text_id, fill='#00d2ff')
            self.configure(bg='#2a2a4a')
            self.border = self.create_rectangle(0, 0, 5, 50, fill='#00d2ff', outline='')
        else:
            self.itemconfig(self.text_id, fill='#a8a8b3')
            self.configure(bg='#1a1a2e')
            if hasattr(self, 'border'):
                self.delete(self.border)


class LEBASApplication:
    """Main application class with professional UI and auto graph popup."""
    
    def __init__(self, root):
        self.root = root
        self.root.title(Config.APP_NAME)
        self.root.geometry("1400x850")
        self.root.minsize(1200, 750)
        self.root.configure(bg='#0f0f1a')
        
        # Set application icon
        self._set_icon()
        
        # Views dictionary
        self.views = {}
        self.current_view = None
        
        # Create main container
        self.main_container = tk.Frame(root, bg='#0f0f1a')
        self.main_container.pack(fill='both', expand=True)
        
        # Create sidebar
        self._create_sidebar()
        
        # Create content area
        self.content_frame = tk.Frame(self.main_container, bg='#1a1a2e')
        self.content_frame.pack(side='right', fill='both', expand=True)
        
        # Configure content frame grid
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)
        
        # Create menu (simplified)
        self._create_menu()
        
        # Initialize views
        self._initialize_views()
        
        # Show dashboard by default
        self.show_frame("DashboardView")
        
        # Set up window close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        logger.info(f"{Config.APP_NAME} version {Config.APP_VERSION} started")
        
        # LOAD DATASET AND SHOW GRAPHS AUTOMATICALLY
        self.root.after(1000, self._auto_load_and_show_graphs)  # Wait 1 second for UI to load
        
        self._check_dataset()
    
    def _set_icon(self):
        try:
            icon_path = Config.ASSETS_DIR / 'icon.ico'
            if icon_path.exists():
                self.root.iconbitmap(str(icon_path))
        except:
            pass
    
    def _create_menu(self):
        """Create simple menu bar."""
        menubar = tk.Menu(self.root)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Dataset", command=self._load_dataset)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_closing)
        
        # Graph menu with direct access to graphs
        graphs_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="📊 Graphs", menu=graphs_menu)
        
        # Direct graph commands (no submenus)
        graphs_menu.add_command(label="🔨 Crushing - Yield", 
                               command=lambda: self._show_crushing_graph('yield'))
        graphs_menu.add_command(label="🔨 Crushing - Reduction Ratio", 
                               command=lambda: self._show_crushing_graph('reduction'))
        graphs_menu.add_separator()
        graphs_menu.add_command(label="⚙️ Grinding - Yield", 
                               command=lambda: self._show_grinding_graph('yield'))
        graphs_menu.add_command(label="⚙️ Grinding - P80 vs Time", 
                               command=lambda: self._show_grinding_graph('p80'))
        graphs_menu.add_separator()
        graphs_menu.add_command(label="📊 Sieve - PSD Curve", 
                               command=lambda: self._show_sieve_graph('psd'))
        graphs_menu.add_command(label="📊 Sieve - PSD Comparison", 
                               command=lambda: self._show_sieve_graph('comparison'))
        graphs_menu.add_separator()
        graphs_menu.add_command(label="🧪 Beneficiation - Recovery", 
                               command=lambda: self._show_beneficiation_graph('recovery'))
        graphs_menu.add_command(label="🧪 Beneficiation - Grade vs Recovery", 
                               command=lambda: self._show_beneficiation_graph('grade_vs_recovery'))
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
        
        self.root.config(menu=menubar)
        logger.info("Menu created successfully")
    
    def _show_about(self):
        messagebox.showinfo(
            "About LEBAS",
            f"{Config.APP_NAME}\nVersion: {Config.APP_VERSION}\n\n"
            f"Laboratory Extraction and Beneficiation Analysis System"
        )
    
    def _auto_load_and_show_graphs(self):
        """Automatically load dataset and show graphs."""
        try:
            from services.data_loader import load_dataset
            from calculations.crushing import calculate_crushing_metrics
            from calculations.grinding import calculate_grinding_metrics
            from calculations.sieve_analysis import calculate_sieve_analysis
            from calculations.beneficiation import calculate_beneficiation
            
            # Load dataset
            df = load_dataset()
            
            if df is not None and not df.empty:
                logger.info(f"Dataset loaded: {len(df)} rows")
                
                # Calculate all metrics
                crushing_df = calculate_crushing_metrics(df)
                grinding_df = calculate_grinding_metrics(df)
                sieve_df = calculate_sieve_analysis(df)
                ben_df = calculate_beneficiation(df)
                
                generator = GraphGenerator()
                
                # Open ALL graphs automatically in new windows
                graph_windows = []
                
                # 1. Crushing Yield
                if not crushing_df.empty:
                    fig1 = generator.crushing_yield_by_material(crushing_df)
                    gw1 = GraphWindow(self.root, "Crushing Yield by Material", fig1)
                    graph_windows.append(gw1)
                    logger.info("Opened Crushing Yield graph")
                
                # 2. Reduction Ratio Distribution
                if not crushing_df.empty:
                    fig2 = generator.reduction_ratio_distribution(crushing_df)
                    gw2 = GraphWindow(self.root, "Reduction Ratio Distribution", fig2)
                    graph_windows.append(gw2)
                    logger.info("Opened Reduction Ratio graph")
                
                # 3. Grinding Yield by Media Ratio
                if not grinding_df.empty:
                    fig3 = generator.grinding_yield_by_media_ratio(grinding_df)
                    gw3 = GraphWindow(self.root, "Grinding Yield by Media Ratio", fig3)
                    graph_windows.append(gw3)
                    logger.info("Opened Grinding Yield graph")
                
                # 4. Particle Size Distribution
                if not sieve_df.empty:
                    fig4 = generator.particle_size_distribution(sieve_df)
                    gw4 = GraphWindow(self.root, "Particle Size Distribution", fig4)
                    graph_windows.append(gw4)
                    logger.info("Opened PSD graph")
                
                # 5. Recovery by Material
                if not ben_df.empty:
                    fig5 = generator.recovery_by_material(ben_df)
                    gw5 = GraphWindow(self.root, "Recovery by Material", fig5)
                    graph_windows.append(gw5)
                    logger.info("Opened Recovery graph")
                
                # 6. Grade vs Recovery
                if not ben_df.empty:
                    fig6 = generator.grade_vs_recovery(ben_df)
                    gw6 = GraphWindow(self.root, "Grade vs Recovery", fig6)
                    graph_windows.append(gw6)
                    logger.info("Opened Grade vs Recovery graph")
                
                # 7. Enrichment Comparison
                if not ben_df.empty:
                    fig7 = generator.enrichment_comparison(ben_df)
                    gw7 = GraphWindow(self.root, "Enrichment Comparison", fig7)
                    graph_windows.append(gw7)
                    logger.info("Opened Enrichment graph")
                
                # 8. Mass Balance Closure
                if not crushing_df.empty:
                    from calculations.material_balance import calculate_material_balance
                    balance_df = calculate_material_balance(df)
                    if not balance_df.empty:
                        fig8 = generator.mass_balance_closure(balance_df)
                        gw8 = GraphWindow(self.root, "Mass Balance Closure", fig8)
                        graph_windows.append(gw8)
                        logger.info("Opened Mass Balance graph")
                
                # 9. PSD Comparison
                if not sieve_df.empty:
                    fig9 = generator.psd_comparison(sieve_df)
                    gw9 = GraphWindow(self.root, "PSD Comparison", fig9)
                    graph_windows.append(gw9)
                    logger.info("Opened PSD Comparison graph")
                
                # 10. Separation Efficiency
                if not ben_df.empty and 'separation_efficiency_percent' in ben_df.columns:
                    from charts.beneficiation import plot_separation_efficiency
                    # Use the beneficiation chart function
                    fig10 = generator.recovery_vs_yield_tradeoff(ben_df)
                    gw10 = GraphWindow(self.root, "Recovery vs Yield Trade-off", fig10)
                    graph_windows.append(gw10)
                    logger.info("Opened Recovery vs Yield graph")
                
                # Update status
                self.dataset_status.config(
                    text=f"📊 {len(df)} rows, {len(graph_windows)} graphs opened",
                    fg='#2ecc71'
                )
                
                # Show message
                messagebox.showinfo(
                    "Graphs Opened",
                    f"✅ {len(graph_windows)} graphs have been opened automatically!\n\n"
                    f"Graphs opened:\n"
                    f"• Crushing Yield\n"
                    f"• Reduction Ratio Distribution\n"
                    f"• Grinding Yield by Media Ratio\n"
                    f"• Particle Size Distribution\n"
                    f"• PSD Comparison\n"
                    f"• Recovery by Material\n"
                    f"• Grade vs Recovery\n"
                    f"• Enrichment Comparison\n"
                    f"• Mass Balance Closure\n"
                    f"• Recovery vs Yield Trade-off\n\n"
                    f"Close graph windows when done."
                )
                
            else:
                logger.warning("No data available for graphs")
                self.dataset_status.config(text="📂 No Data Available", fg='#e74c3c')
                
        except Exception as e:
            logger.error(f"Auto graph generation failed: {str(e)}")
            self.dataset_status.config(text=f"❌ Error: {str(e)[:30]}...", fg='#e74c3c')
    
    def _load_dataset(self):
        """Load dataset manually."""
        from services.data_loader import load_dataset
        try:
            df = load_dataset()
            if df is not None and not df.empty:
                messagebox.showinfo("Success", f"Dataset loaded successfully!\n{len(df)} rows")
                self._auto_load_and_show_graphs()
            else:
                messagebox.showwarning("No Data", "Dataset is empty or could not be loaded.")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load dataset:\n{str(e)}")
    
    def _create_sidebar(self):
        """Create professional sidebar navigation."""
        sidebar = tk.Frame(self.main_container, bg='#1a1a2e', width=220)
        sidebar.pack(side='left', fill='y')
        sidebar.pack_propagate(False)
        
        # Logo/Title
        logo_frame = tk.Frame(sidebar, bg='#1a1a2e', height=80)
        logo_frame.pack(fill='x', pady=(20, 10))
        logo_frame.pack_propagate(False)
        
        logo_label = tk.Label(
            logo_frame,
            text="🔬 LEBAS",
            font=('Segoe UI', 20, 'bold'),
            bg='#1a1a2e',
            fg='#00d2ff'
        )
        logo_label.pack(pady=10)
        
        subtitle = tk.Label(
            logo_frame,
            text="Laboratory Extraction & Beneficiation",
            font=('Segoe UI', 8),
            bg='#1a1a2e',
            fg='#6c6c80'
        )
        subtitle.pack()
        
        # Separator
        sep = tk.Frame(sidebar, bg='#2a2a4a', height=1)
        sep.pack(fill='x', padx=20, pady=10)
        
        # Navigation buttons
        self.nav_buttons = {}
        nav_items = [
            ("🏠", "Dashboard", "DashboardView"),
            ("🔨", "Crushing", "CrushingView"),
            ("⚙️", "Grinding", "GrindingView"),
            ("📊", "Sieve Analysis", "SieveView"),
            ("🧪", "Beneficiation", "BeneficiationView"),
            ("📈", "Results", "ResultsView"),
            ("📄", "Reports", "ReportsView"),
        ]
        
        for icon, text, view_name in nav_items:
            btn = ModernNavButton(
                sidebar,
                text=text,
                icon=icon,
                command=lambda v=view_name: self.show_frame(v),
                width=200
            )
            btn.pack(pady=2, padx=10)
            self.nav_buttons[view_name] = btn
        
        # Footer
        footer = tk.Frame(sidebar, bg='#1a1a2e')
        footer.pack(side='bottom', fill='x', pady=15)
        
        version_label = tk.Label(
            footer,
            text=f"v{Config.APP_VERSION}",
            font=('Segoe UI', 8),
            bg='#1a1a2e',
            fg='#4a4a5e'
        )
        version_label.pack()
        
        self.dataset_status = tk.Label(
            footer,
            text="📂 Not Loaded",
            font=('Segoe UI', 8),
            bg='#1a1a2e',
            fg='#e74c3c'
        )
        self.dataset_status.pack()
    
    def _initialize_views(self):
        """Initialize all view classes."""
        try:
            self.views = {
                "DashboardView": DashboardView(self.content_frame, self),
                "CrushingView": CrushingView(self.content_frame, self),
                "GrindingView": GrindingView(self.content_frame, self),
                "SieveView": SieveView(self.content_frame, self),
                "BeneficiationView": BeneficiationView(self.content_frame, self),
                "ResultsView": ResultsView(self.content_frame, self),
                "ReportsView": ReportsView(self.content_frame, self),
            }
            
            for view in self.views.values():
                view.get_frame().grid_remove()
                
            logger.info("All views initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing views: {str(e)}")
            messagebox.showerror("Initialization Error", f"Failed to initialize views:\n\n{str(e)}")
            sys.exit(1)
    
    def show_frame(self, view_name: str):
        """Show a specific frame and update navigation."""
        try:
            if view_name not in self.views:
                return
            
            if self.current_view and self.current_view in self.views:
                self.views[self.current_view].get_frame().grid_remove()
            
            view = self.views[view_name]
            view.get_frame().grid(row=0, column=0, sticky="nsew")
            
            for name, btn in self.nav_buttons.items():
                btn.set_active(name == view_name)
            
            if hasattr(view, 'refresh'):
                view.refresh()
            
            self.current_view = view_name
            self._update_dataset_status()
            
            logger.info(f"Showing view: {view_name}")
            
        except Exception as e:
            logger.error(f"Error showing frame {view_name}: {str(e)}")
            messagebox.showerror("Navigation Error", f"Failed to load view:\n\n{str(e)}")
    
    def _update_dataset_status(self):
        """Update dataset status indicator."""
        try:
            from services.data_loader import data_loader
            if data_loader._is_loaded and data_loader._dataframe is not None:
                df = data_loader._dataframe
                samples = df['sample_id'].nunique() if 'sample_id' in df.columns else 0
                self.dataset_status.config(
                    text=f"📊 {len(df)} rows, {samples} samples",
                    fg='#2ecc71'
                )
            else:
                self.dataset_status.config(text="📂 Not Loaded", fg='#e74c3c')
        except:
            self.dataset_status.config(text="📂 Error", fg='#e74c3c')
    
    def _check_dataset(self):
        """Check if dataset exists."""
        try:
            from services.data_loader import data_loader
            if Config.DATASET_PATH.exists():
                try:
                    data_loader.load_dataset()
                    self._update_dataset_status()
                    logger.info("Dataset loaded successfully")
                except Exception as e:
                    logger.warning(f"Could not load dataset: {str(e)}")
            else:
                self.dataset_status.config(text="📂 Not Found", fg='#e74c3c')
                logger.warning(f"Dataset not found at: {Config.DATASET_PATH}")
        except Exception as e:
            logger.error(f"Error checking dataset: {str(e)}")
    
    def _on_closing(self):
        try:
            logger.info(f"{Config.APP_NAME} shutting down")
            self.root.quit()
            self.root.destroy()
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            self.root.destroy()
    
    # =========================================================================
    # GRAPH SHOWING METHODS (for manual access via menu)
    # =========================================================================
    
    def _show_crushing_graph(self, graph_type):
        from services.data_loader import load_dataset
        from calculations.crushing import calculate_crushing_metrics
        
        try:
            df = load_dataset()
            if df is None or df.empty:
                messagebox.showwarning("No Data", "Please load dataset first.")
                return
            
            crushing_df = calculate_crushing_metrics(df)
            if crushing_df.empty:
                messagebox.showwarning("No Data", "No crushing data available.")
                return
            
            generator = GraphGenerator()
            if graph_type == 'yield':
                fig = generator.crushing_yield_by_material(crushing_df)
                title = "Crushing Yield by Material"
            elif graph_type == 'reduction':
                fig = generator.reduction_ratio_distribution(crushing_df)
                title = "Reduction Ratio Distribution"
            else:
                return
            
            GraphWindow(self.root, title, fig)
            
        except Exception as e:
            logger.error(f"Graph error: {str(e)}")
            messagebox.showerror("Graph Error", f"Failed to generate graph:\n{str(e)}")
    
    def _show_grinding_graph(self, graph_type):
        from services.data_loader import load_dataset
        from calculations.grinding import calculate_grinding_metrics
        from calculations.sieve_analysis import calculate_particle_size_distribution
        
        try:
            df = load_dataset()
            if df is None or df.empty:
                messagebox.showwarning("No Data", "Please load dataset first.")
                return
            
            grinding_df = calculate_grinding_metrics(df)
            psd_df = calculate_particle_size_distribution(df)
            
            if grinding_df.empty:
                messagebox.showwarning("No Data", "No grinding data available.")
                return
            
            generator = GraphGenerator()
            if graph_type == 'yield':
                fig = generator.grinding_yield_by_media_ratio(grinding_df)
                title = "Grinding Yield by Media Ratio"
            elif graph_type == 'p80':
                fig = generator.p80_vs_grinding_time(grinding_df, psd_df)
                title = "P80 vs Grinding Time"
            else:
                return
            
            GraphWindow(self.root, title, fig)
            
        except Exception as e:
            logger.error(f"Graph error: {str(e)}")
            messagebox.showerror("Graph Error", f"Failed to generate graph:\n{str(e)}")
    
    def _show_sieve_graph(self, graph_type):
        from services.data_loader import load_dataset
        from calculations.sieve_analysis import calculate_sieve_analysis
        
        try:
            df = load_dataset()
            if df is None or df.empty:
                messagebox.showwarning("No Data", "Please load dataset first.")
                return
            
            sieve_df = calculate_sieve_analysis(df)
            if sieve_df.empty:
                messagebox.showwarning("No Data", "No sieve data available.")
                return
            
            generator = GraphGenerator()
            if graph_type == 'psd':
                fig = generator.particle_size_distribution(sieve_df)
                title = "Particle Size Distribution"
            elif graph_type == 'comparison':
                fig = generator.psd_comparison(sieve_df)
                title = "PSD Comparison"
            else:
                return
            
            GraphWindow(self.root, title, fig)
            
        except Exception as e:
            logger.error(f"Graph error: {str(e)}")
            messagebox.showerror("Graph Error", f"Failed to generate graph:\n{str(e)}")
    
    def _show_beneficiation_graph(self, graph_type):
        from services.data_loader import load_dataset
        from calculations.beneficiation import calculate_beneficiation
        
        try:
            df = load_dataset()
            if df is None or df.empty:
                messagebox.showwarning("No Data", "Please load dataset first.")
                return
            
            ben_df = calculate_beneficiation(df)
            if ben_df.empty:
                messagebox.showwarning("No Data", "No beneficiation data available.")
                return
            
            generator = GraphGenerator()
            if graph_type == 'recovery':
                fig = generator.recovery_by_material(ben_df)
                title = "Recovery by Material"
            elif graph_type == 'grade_vs_recovery':
                fig = generator.grade_vs_recovery(ben_df)
                title = "Grade vs Recovery"
            else:
                return
            
            GraphWindow(self.root, title, fig)
            
        except Exception as e:
            logger.error(f"Graph error: {str(e)}")
            messagebox.showerror("Graph Error", f"Failed to generate graph:\n{str(e)}")
    
    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.root.destroy()
        except Exception as e:
            logger.error(f"Application error: {str(e)}")
            logger.error(traceback.format_exc())
            self.root.destroy()


def main():
    try:
        root = tk.Tk()
        try:
            from ctypes import windll
            windll.shutil.SetProcessDPIAware()
        except:
            pass
        
        app = LEBASApplication(root)
        app.run()
        
    except Exception as e:
        error_msg = f"Fatal Error: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("LEBAS Fatal Error", f"An unexpected error occurred:\n\n{str(e)}")
            root.destroy()
        except:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()