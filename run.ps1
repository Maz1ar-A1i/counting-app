# Run script for PowerShell
# This ensures the virtual environment is activated

Write-Host "Activating virtual environment..." -ForegroundColor Green
& ".\venv\Scripts\Activate.ps1"

Write-Host "Starting application..." -ForegroundColor Green
python main.py

