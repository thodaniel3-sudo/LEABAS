"""
Simple build script for LEBAS - One click build
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("LEBAS - One Click Build")
    print("=" * 70)
    print()
    
    # Get the project root
    project_root = Path(__file__).parent.absolute()
    os.chdir(project_root)
    
    # Step 1: Install PyInstaller
    print("Installing PyInstaller...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyinstaller'], check=False)
    print()
    
    # Step 2: Build
    print("Building executable...")
    print("-" * 50)
    
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--onefile',
        '--windowed',
        '--name', 'LEBAS',
        '--add-data', f'dataset.csv;.',
        '--add-data', f'config.py;.',
        '--add-data', f'calculations;calculations',
        '--add-data', f'ui;ui',
        '--add-data', f'charts;charts',
        '--add-data', f'services;services',
        '--add-data', f'assets;assets',
        '--add-data', f'reports;reports',
        '--add-data', f'exports;exports',
        '--hidden-import', 'pandas',
        '--hidden-import', 'numpy',
        '--hidden-import', 'matplotlib',
        '--hidden-import', 'matplotlib.backends.backend_tkagg',
        '--hidden-import', 'openpyxl',
        '--hidden-import', 'reportlab',
        '--hidden-import', 'scipy',
        '--collect-all', 'matplotlib',
        '--collect-all', 'numpy',
        '--collect-all', 'pandas',
        'main.py'
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print()
        print("✅ Build successful!")
        
        # Copy to root
        import shutil
        dist_exe = project_root / 'dist' / 'LEBAS.exe'
        root_exe = project_root / 'LEBAS.exe'
        if dist_exe.exists():
            shutil.copy2(dist_exe, root_exe)
            print(f"✅ Executable copied to: {root_exe}")
            print()
            print("=" * 70)
            print("BUILD COMPLETE!")
            print("=" * 70)
            print()
            print(f"Run: {root_exe}")
    else:
        print()
        print("❌ Build failed. Check the error messages above.")
    
    print()
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()