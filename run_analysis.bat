@echo off
echo ================================================================================
echo BigMart Sales Prediction Analysis Pipeline
echo ================================================================================
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Running comprehensive analysis...
echo This will take approximately 20-40 minutes depending on your hardware.
echo.
echo Progress will be displayed in the console.
echo Results will be saved as CSV files and PNG visualizations.
echo.

python bigmart_analysis.py

echo.
echo ================================================================================
echo Analysis Complete!
echo ================================================================================
echo.
echo Generated files:
echo.
echo CSV Files (current directory):
echo   - model_performance_summary.csv
echo   - statistical_comparison.csv
echo   - performance_by_outlet_type.csv
echo   - performance_by_category.csv
echo   - best_model.pkl
echo.
echo Visualization Files (Visualizations\ folder):
echo   - 11 PNG visualization files
echo.
echo Check the output above for model performance results.
echo See README.md for detailed information on interpreting results.
echo See process.txt for methodology and reasoning.
echo.
pause

