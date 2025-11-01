# BigMart Sales Prediction Analysis Pipeline
# PowerShell execution script

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "BigMart Sales Prediction Analysis Pipeline" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

Write-Host ""
Write-Host "Running comprehensive analysis..." -ForegroundColor Yellow
Write-Host "This will take approximately 20-40 minutes depending on your hardware." -ForegroundColor Gray
Write-Host ""
Write-Host "Progress will be displayed in the console." -ForegroundColor Gray
Write-Host "Results will be saved as CSV files and PNG visualizations." -ForegroundColor Gray
Write-Host ""

python bigmart_analysis.py

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "Analysis Complete!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Generated files:" -ForegroundColor Yellow
Write-Host ""
Write-Host "CSV Files (current directory):" -ForegroundColor Cyan
Write-Host "  - model_performance_summary.csv" -ForegroundColor White
Write-Host "  - statistical_comparison.csv" -ForegroundColor White
Write-Host "  - performance_by_outlet_type.csv" -ForegroundColor White
Write-Host "  - performance_by_category.csv" -ForegroundColor White
Write-Host "  - best_model.pkl" -ForegroundColor White
Write-Host ""
Write-Host "Visualization Files (Visualizations\ folder):" -ForegroundColor Cyan
Write-Host "  - 11 PNG visualization files" -ForegroundColor White
Write-Host ""
Write-Host "Check the output above for model performance results." -ForegroundColor Gray
Write-Host "See README.md for detailed information on interpreting results." -ForegroundColor Gray
Write-Host "See process.txt for methodology and reasoning." -ForegroundColor Gray
Write-Host ""

