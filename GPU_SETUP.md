# GPU Setup Guide for Object Detection System

This guide will help you set up and run the project on GPU (NVIDIA CUDA) for faster inference.

## Prerequisites

- **NVIDIA GPU** with CUDA support (Compute Capability 3.5+)
- **Windows 10/11** (or Linux)
- **Python 3.8+**

---

## Step 1: Check Your GPU

### Check if you have an NVIDIA GPU:

1. **Windows:**
   - Press `Win + X` → Device Manager → Display adapters
   - Look for "NVIDIA" in the list

2. **Check GPU details:**
   ```powershell
   nvidia-smi
   ```
   If this command works, you have NVIDIA drivers installed.

### If `nvidia-smi` doesn't work:
- Download and install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx
- Install the latest Game Ready or Studio drivers for your GPU

---

## Step 2: Install CUDA Toolkit

### Check CUDA Version Required:
- PyTorch currently supports **CUDA 11.8** or **CUDA 12.1**
- Check your GPU's CUDA Compute Capability: https://developer.nvidia.com/cuda-gpus

### Install CUDA Toolkit:

1. **Download CUDA Toolkit:**
   - Visit: https://developer.nvidia.com/cuda-downloads
   - Select your OS (Windows/Linux)
   - Download **CUDA 11.8** or **CUDA 12.1** (recommended: 12.1)

2. **Install CUDA:**
   - Run the installer
   - Choose "Express Installation"
   - Follow the installation wizard
   - **Note:** Installation may take 10-15 minutes

3. **Verify CUDA Installation:**
   ```powershell
   nvcc --version
   ```
   Should show CUDA version (e.g., "release 12.1")

---

## Step 3: Install PyTorch with CUDA Support

### Option A: CUDA 12.1 (Recommended - Latest)

```powershell
# Activate your virtual environment first (if using one)
# Then install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Option B: CUDA 11.8 (If CUDA 12.1 doesn't work)

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Option C: CPU Only (Fallback - No GPU)

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## Step 4: Verify GPU Setup

### Test PyTorch GPU Access:

Create a test file `test_gpu.py`:

```python
import torch

print("=" * 50)
print("GPU Setup Test")
print("=" * 50)

# Check if CUDA is available
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Test GPU computation
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("✅ GPU computation test: SUCCESS")
else:
    print("❌ CUDA not available. Check your installation.")
    print("   - Make sure NVIDIA drivers are installed")
    print("   - Make sure CUDA toolkit is installed")
    print("   - Make sure PyTorch was installed with CUDA support")

print("=" * 50)
```

Run it:
```powershell
python test_gpu.py
```

**Expected Output (GPU working):**
```
CUDA Available: True
CUDA Version: 12.1
GPU Device: NVIDIA GeForce RTX 3060
GPU Memory: 12.00 GB
✅ GPU computation test: SUCCESS
```

---

## Step 5: Install Project Dependencies

### Install all project requirements:

```powershell
# Navigate to project directory
cd E:\Counting

# Install dependencies (excluding PyTorch - already installed)
pip install -r requirements.txt
```

**Note:** If you see errors about PyTorch, ignore them - you already installed it in Step 3.

---

## Step 6: Run the Project

### Start the Backend Server:

```powershell
# Option 1: Use the batch file
.\run_backend.bat

# Option 2: Run manually
cd backend
python server.py
```

### Check Console Output:

When the server starts, you should see:

```
[INFO] Using GPU: NVIDIA GeForce RTX 3060
[OK] Using GPU (FP16)
[OK] Model loaded successfully
```

**If you see "Using CPU" instead:**
- GPU is not being detected
- Check Step 4 to verify GPU setup
- Make sure PyTorch was installed with CUDA support

---

## Step 7: Verify GPU Usage During Runtime

### Monitor GPU Usage:

1. **Open a new terminal and run:**
   ```powershell
   nvidia-smi -l 1
   ```
   This shows GPU usage updating every second.

2. **Start the camera in your web app**

3. **Watch GPU usage:**
   - **GPU Utilization** should increase (20-80% depending on model)
   - **Memory Usage** should show some allocation
   - If both stay at 0%, GPU is not being used

---

## Troubleshooting

### Problem: `torch.cuda.is_available()` returns `False`

**Solutions:**
1. **Check NVIDIA drivers:**
   ```powershell
   nvidia-smi
   ```
   If this fails, install/update NVIDIA drivers.

2. **Check CUDA installation:**
   ```powershell
   nvcc --version
   ```
   If this fails, reinstall CUDA toolkit.

3. **Reinstall PyTorch with correct CUDA version:**
   ```powershell
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Check Python environment:**
   - Make sure you're using the same Python where PyTorch is installed
   - If using virtual environment, activate it first

### Problem: "CUDA out of memory" error

**Solutions:**
1. **Reduce batch size** (already done in code - processes one frame at a time)
2. **Use smaller model:**
   - Current: `yolov8n.pt` (nano - smallest)
   - If still issues, reduce `img_size` in `backend/server.py` (line 69)

3. **Close other GPU applications:**
   - Close games, video editors, etc.

### Problem: GPU detected but slow performance

**Solutions:**
1. **Check GPU utilization:**
   ```powershell
   nvidia-smi
   ```
   If utilization is low, GPU might not be fully utilized.

2. **Enable FP16 (Half Precision):**
   - Already enabled in code (line 215 in `modules/detector.py`)
   - This should double the speed

3. **Check if model is actually on GPU:**
   - Look for "Using GPU (FP16)" in console output

---

## Performance Comparison

### Expected FPS (on RTX 3060 / similar mid-range GPU):

- **CPU:** 5-15 FPS
- **GPU (FP32):** 30-50 FPS
- **GPU (FP16):** 50-80 FPS ← **Current setup**

### Expected FPS (on RTX 4090 / high-end GPU):

- **GPU (FP16):** 100-150+ FPS

---

## Quick Reference Commands

```powershell
# Check GPU
nvidia-smi

# Check CUDA
nvcc --version

# Test PyTorch GPU
python test_gpu.py

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Monitor GPU usage
nvidia-smi -l 1
```

---

## Additional Notes

1. **The code automatically detects GPU** - no code changes needed!
2. **If GPU is not available, it falls back to CPU** automatically
3. **FP16 (Half Precision) is enabled** for 2x speed boost on GPU
4. **Model fusion is enabled** for additional speed improvement

---

## Need Help?

If you're still having issues:

1. Run `test_gpu.py` and share the output
2. Share output of `nvidia-smi`
3. Share the console output when starting the backend server

---

**Last Updated:** 2024
**Tested on:** Windows 10/11, CUDA 12.1, PyTorch 2.1+

