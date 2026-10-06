@echo off
echo ============================================================
echo LEBAS - Building Standalone Executable
echo ============================================================
echo.

echo Installing required packages...
pip install pyinstaller pandas numpy matplotlib openpyxl reportlab scipy

echo.
echo Building executable...
pyinstaller --onefile --windowed --name LEBAS --add-data "dataset.csv;." --add-data "config.py;." --add-data "calculations;calculations" --add-data "ui;ui" --add-data "charts;charts" --add-data "services;services" --add-data "assets;assets" --add-data "reports;reports" --add-data "exports;exports" --hidden-import pandas --hidden-import numpy --hidden-import matplotlib --hidden-import matplotlib.backends.backend_tkagg --hidden-import openpyxl --hidden-import reportlab --hidden-import scipy --collect-all matplotlib --collect-all numpy --collect-all pandas main.py

echo.
echo Copying executable to root folder...
copy dist\LEBAS.exe LEBAS.exe

echo.
echo ============================================================
echo BUILD COMPLETE!
echo ============================================================
echo.
echo Your executable is located at: LEBAS.exe
echo You can now run LEBAS.exe directly.
pause