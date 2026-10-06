"""
Graph Diagnostic Tool for LEBAS
Run this script to diagnose why graphs are not showing.
"""

import sys
import os
import traceback

print("=" * 70)
print("LEBAS GRAPH DIAGNOSTIC TOOL")
print("=" * 70)
print()

# Step 1: Check Python version
print("1. CHECKING PYTHON VERSION...")
print(f"   Python version: {sys.version}")
print(f"   Python executable: {sys.executable}")
print()

# Step 2: Check imports
print("2. CHECKING IMPORTS...")

try:
    import matplotlib
    print(f"   ✅ matplotlib version: {matplotlib.__version__}")
    print(f"   matplotlib backend: {matplotlib.get_backend()}")
except ImportError as e:
    print(f"   ❌ matplotlib not installed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"   ✅ numpy version: {np.__version__}")
except ImportError as e:
    print(f"   ❌ numpy not installed: {e}")

try:
    import pandas as pd
    print(f"   ✅ pandas version: {pd.__version__}")
except ImportError as e:
    print(f"   ❌ pandas not installed: {e}")

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    print("   ✅ FigureCanvasTkAgg imported successfully")
except ImportError as e:
    print(f"   ❌ FigureCanvasTkAgg import failed: {e}")

print()

# Step 3: Check Tkinter
print("3. CHECKING TKINTER...")
try:
    import tkinter as tk
    print("   ✅ tkinter imported successfully")
    print(f"   Tkinter version: {tk.TkVersion}")
except ImportError as e:
    print(f"   ❌ tkinter not installed: {e}")
    sys.exit(1)

print()

# Step 4: Check project structure
print("4. CHECKING PROJECT STRUCTURE...")
project_root = os.path.dirname(os.path.abspath(__file__))
print(f"   Project root: {project_root}")

# Check for config.py
config_path = os.path.join(project_root, 'config.py')
if os.path.exists(config_path):
    print("   ✅ config.py found")
else:
    print(f"   ❌ config.py not found at {config_path}")

# Check for charts directory
charts_dir = os.path.join(project_root, 'charts')
if os.path.exists(charts_dir):
    print("   ✅ charts directory found")
    
    # Check for graph.py
    graph_path = os.path.join(charts_dir, 'graph.py')
    if os.path.exists(graph_path):
        print("   ✅ charts/graph.py found")
    else:
        print(f"   ❌ charts/graph.py not found at {graph_path}")
else:
    print(f"   ❌ charts directory not found at {charts_dir}")

print()

# Step 5: Try importing graph module
print("5. TRYING TO IMPORT GRAPH MODULE...")
try:
    # Add project root to path
    sys.path.insert(0, project_root)
    
    from charts.graph import GraphGenerator, GraphWindow, show_graph
    print("   ✅ GraphGenerator imported successfully")
    print("   ✅ GraphWindow imported successfully")
    print("   ✅ show_graph imported successfully")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    traceback.print_exc()

print()

# Step 6: Test graph generation
print("6. TESTING GRAPH GENERATION...")
try:
    from charts.graph import GraphGenerator
    generator = GraphGenerator()
    print("   ✅ GraphGenerator instantiated successfully")
    
    # Create a test figure
    print("   Creating test figure...")
    fig = generator.crushing_yield_by_material(pd.DataFrame())
    print(f"   ✅ Test figure created: {type(fig)}")
    
    # Test with sample data
    print("   Testing with sample data...")
    test_df = pd.DataFrame({
        'sample_id': ['TEST-001', 'TEST-002'],
        'material_type': ['limestone', 'limestone'],
        'crushing_yield': [85.0, 90.0]
    })
    fig2 = generator.crushing_yield_by_material(test_df)
    print(f"   ✅ Sample data figure created: {type(fig2)}")
    
except Exception as e:
    print(f"   ❌ Graph generation failed: {e}")
    traceback.print_exc()

print()

# Step 7: Check dataset
print("7. CHECKING DATASET...")
dataset_path = os.path.join(project_root, 'dataset.csv')
if os.path.exists(dataset_path):
    print(f"   ✅ dataset.csv found at: {dataset_path}")
    try:
        import pandas as pd
        df = pd.read_csv(dataset_path)
        print(f"   ✅ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"   Columns: {list(df.columns)[:10]}...")
        
        # Check for required columns
        required_cols = ['sample_id', 'sieve_size', 'mass_retained']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"   ⚠️ Missing columns: {missing}")
        else:
            print("   ✅ All required columns present")
            
        # Check for crushing data
        if 'crushing_yield' in df.columns:
            print(f"   ✅ crushing_yield column present")
        else:
            print("   ⚠️ crushing_yield column not found (may be calculated)")
            
        # Check for beneficiation data
        if 'recovery_percent' in df.columns:
            print(f"   ✅ recovery_percent column present")
        else:
            print("   ⚠️ recovery_percent column not found (may be calculated)")
            
    except Exception as e:
        print(f"   ❌ Failed to load dataset: {e}")
else:
    print(f"   ❌ dataset.csv not found at: {dataset_path}")

print()

# Step 8: Test Tkinter window creation
print("8. TESTING TKINTER WINDOW CREATION...")
try:
    root = tk.Tk()
    root.title("Graph Test Window")
    root.geometry("400x300")
    root.configure(bg='#1a1a2e')
    
    label = tk.Label(
        root,
        text="Graph Test Window",
        font=('Segoe UI', 16, 'bold'),
        bg='#1a1a2e',
        fg='white'
    )
    label.pack(pady=50)
    
    def test_graph():
        print("   Creating test graph in window...")
        try:
            from charts.graph import GraphGenerator, GraphWindow
            generator = GraphGenerator()
            
            # Create test data
            test_df = pd.DataFrame({
                'sample_id': ['TEST-001', 'TEST-002', 'TEST-003'],
                'material_type': ['limestone', 'limestone', 'limestone'],
                'crushing_yield': [85.0, 90.0, 88.0]
            })
            
            fig = generator.crushing_yield_by_material(test_df)
            GraphWindow(root, "Test Graph", fig)
            print("   ✅ Test graph window opened!")
        except Exception as e:
            print(f"   ❌ Failed to create test graph: {e}")
            traceback.print_exc()
    
    btn = tk.Button(
        root,
        text="Open Test Graph",
        command=test_graph,
        bg='#2E86AB',
        fg='white',
        font=('Segoe UI', 12, 'bold'),
        relief='flat',
        padx=20,
        pady=10,
        cursor='hand2'
    )
    btn.pack(pady=20)
    
    print("   ✅ Tkinter window created successfully")
    print("   Click the 'Open Test Graph' button to test graph display")
    print()
    print("=" * 70)
    print("Diagnostic complete. The test window should open.")
    print("Close the window when done.")
    print("=" * 70)
    
    root.mainloop()
    
except Exception as e:
    print(f"   ❌ Failed to create Tkinter window: {e}")
    traceback.print_exc()

print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)