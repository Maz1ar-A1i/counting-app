@echo off
echo ========================================
echo GPU Setup for Object Detection System
echo ========================================
echo.

echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] nvidia-smi not found. NVIDIA drivers may not be installed.
    echo Please install NVIDIA drivers first: https://www.nvidia.com/Download/index.aspx
    echo.
    pause
    exit /b 1
)

echo [OK] NVIDIA GPU detected!
echo.

echo Installing PyTorch with CUDA 12.1 support...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if %errorlevel% neq 0 (
    echo.
    echo [WARNING] CUDA 12.1 installation failed, trying CUDA 11.8...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)

echo.
echo Installing other dependencies...
pip install -r requirements.txt

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Testing GPU setup...
python test_gpu.py

echo.
echo Setup finished! You can now run the project with GPU acceleration.
pause

