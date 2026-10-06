"""
LEBAS - Build Script for creating standalone executable
Run this script to create a .exe file in the root folder
"""

import os
import sys
import shutil
from pathlib import Path

# Install required packages if not present
def install_packages():
    """Install required packages for building."""
    import subprocess
    
    packages = [
        'pyinstaller',
        'matplotlib',
        'pandas',
        'numpy',
        'openpyxl',
        'reportlab',
        'scipy'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Installed {package}")
        except Exception as e:
            print(f"⚠️ Failed to install {package}: {e}")

def build_exe():
    """Build the executable."""
    import subprocess
    
    print("=" * 70)
    print("LEBAS - Building Standalone Executable")
    print("=" * 70)
    print()
    
    # Step 1: Install packages
    print("Step 1: Installing required packages...")
    print("-" * 50)
    install_packages()
    print()
    
    # Step 2: Create PyInstaller command
    print("Step 2: Building executable with PyInstaller...")
    print("-" * 50)
    
    # Get the project root
    project_root = Path(__file__).parent.absolute()
    
    # PyInstaller command
    cmd = [
        'pyinstaller',
        '--onefile',  # Single executable file
        '--windowed',  # No console window (GUI only)
        '--name', 'LEBAS',  # Name of the executable
        '--icon', str(project_root / 'assets' / 'icon.ico') if (project_root / 'assets' / 'icon.ico').exists() else '',
        '--add-data', f'{str(project_root / "dataset.csv")};.',  # Include dataset
        '--add-data', f'{str(project_root / "config.py")};.',  # Include config
        '--add-data', f'{str(project_root / "calculations")};calculations',  # Include calculations
        '--add-data', f'{str(project_root / "ui")};ui',  # Include UI
        '--add-data', f'{str(project_root / "charts")};charts',  # Include charts
        '--add-data', f'{str(project_root / "services")};services',  # Include services
        '--add-data', f'{str(project_root / "assets")};assets',  # Include assets
        '--add-data', f'{str(project_root / "reports")};reports',  # Include reports
        '--add-data', f'{str(project_root / "exports")};exports',  # Include exports
        '--hidden-import', 'pandas',
        '--hidden-import', 'numpy',
        '--hidden-import', 'matplotlib',
        '--hidden-import', 'matplotlib.backends.backend_tkagg',
        '--hidden-import', 'openpyxl',
        '--hidden-import', 'reportlab',
        '--hidden-import', 'scipy',
        '--hidden-import', 'tkinter',
        '--hidden-import', 'tkinter.ttk',
        '--hidden-import', 'tkinter.messagebox',
        '--hidden-import', 'tkinter.filedialog',
        '--hidden-import', 'tkinter.simpledialog',
        '--collect-all', 'matplotlib',
        '--collect-all', 'numpy',
        '--collect-all', 'pandas',
        '--collect-data', 'matplotlib',
        '--collect-data', 'openpyxl',
        '--collect-data', 'reportlab',
        '--runtime-tmpdir', 'tmp',
        'main.py'
    ]
    
    # Remove empty arguments
    cmd = [arg for arg in cmd if arg]
    
    # Run PyInstaller
    try:
        subprocess.run(cmd, check=True)
        print()
        print("✅ Executable built successfully!")
        print()
        
        # Step 3: Copy executable to root folder
        print("Step 3: Copying executable to root folder...")
        print("-" * 50)
        
        dist_path = project_root / 'dist' / 'LEBAS.exe'
        root_exe_path = project_root / 'LEBAS.exe'
        
        if dist_path.exists():
            shutil.copy2(dist_path, root_exe_path)
            print(f"✅ Executable copied to: {root_exe_path}")
            
            # Step 4: Clean up build files (optional)
            print()
            print("Step 4: Cleaning up build files...")
            print("-" * 50)
            
            build_dir = project_root / 'build'
            dist_dir = project_root / 'dist'
            spec_file = project_root / 'LEBAS.spec'
            
            # Ask user if they want to clean up
            cleanup = input("Clean up build files? (y/n): ").lower().strip()
            if cleanup == 'y':
                if build_dir.exists():
                    shutil.rmtree(build_dir)
                    print("✅ Removed build directory")
                if dist_dir.exists():
                    shutil.rmtree(dist_dir)
                    print("✅ Removed dist directory")
                if spec_file.exists():
                    spec_file.unlink()
                    print("✅ Removed spec file")
                print()
            
        else:
            print("❌ Executable not found in dist folder")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        return
    
    # Final message
    print("=" * 70)
    print("✅ BUILD COMPLETE!")
    print("=" * 70)
    print()
    print(f"Your executable is located at: {root_exe_path}")
    print()
    print("You can now run LEBAS.exe directly from the root folder.")
    print("Note: The first run may take a few seconds to start.")
    print("=" * 70)


if __name__ == "__main__":
    build_exe()