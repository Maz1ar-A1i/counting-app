@echo off
echo ========================================
echo PyTorch CPU-Only Installation Fix
echo ========================================
echo.
echo This script will fix PyTorch DLL loading errors
echo by reinstalling PyTorch with CPU-only support.
echo.
pause

echo.
echo Step 1: Uninstalling existing PyTorch packages...
call venv\Scripts\activate.bat
pip uninstall -y torch torchvision torchaudio

echo.
echo Step 2: Installing PyTorch CPU-only version...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo Step 3: Verifying installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
pause

