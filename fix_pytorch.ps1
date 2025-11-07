# PyTorch CPU-Only Installation Fix Script for PowerShell
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PyTorch CPU-Only Installation Fix" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This script will fix PyTorch DLL loading errors" -ForegroundColor Yellow
Write-Host "by reinstalling PyTorch with CPU-only support." -ForegroundColor Yellow
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& ".\venv\Scripts\Activate.ps1"

# Uninstall existing PyTorch packages
Write-Host ""
Write-Host "Step 1: Uninstalling existing PyTorch packages..." -ForegroundColor Green
pip uninstall -y torch torchvision torchaudio

# Install CPU-only PyTorch
Write-Host ""
Write-Host "Step 2: Installing PyTorch CPU-only version..." -ForegroundColor Green
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Verify installation
Write-Host ""
Write-Host "Step 3: Verifying installation..." -ForegroundColor Green
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now run your application with: python main.py" -ForegroundColor Yellow

