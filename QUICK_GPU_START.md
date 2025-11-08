# Quick GPU Setup Guide

## 🚀 Fast Setup (3 Steps)

### Step 1: Install PyTorch with CUDA

**Option A: Automatic Setup (Recommended)**
```powershell
.\install_gpu.bat
```

**Option B: Manual Installation**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 2: Test GPU Setup
```powershell
python test_gpu.py
```

**Expected Output:**
```
✅ CUDA Available: True
✅ GPU Device: NVIDIA GeForce RTX 3060
✅ GPU Memory: 12.00 GB
🎉 GPU is ready to use!
```

### Step 3: Run the Project
```powershell
.\run_backend.bat
```

**Look for this in console:**
```
============================================================
[GPU] CUDA Available - GPU Acceleration Enabled
[GPU] Device: NVIDIA GeForce RTX 3060
[GPU] Memory: 12.00 GB
[GPU] CUDA Version: 12.1
============================================================
[OK] Using GPU (FP16) - Optimized for speed
```

---

## ✅ That's It!

Your project is now running on GPU with:
- **FP16 (Half Precision)** - 2x faster inference
- **cuDNN Optimizations** - Additional speed boost
- **Model Fusion** - Optimized model structure
- **Automatic GPU Memory Management** - No memory leaks

---

## 🐛 Troubleshooting

### If you see "Using CPU" instead of "Using GPU":

1. **Check if GPU is detected:**
   ```powershell
   nvidia-smi
   ```

2. **Reinstall PyTorch with CUDA:**
   ```powershell
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Run test again:**
   ```powershell
   python test_gpu.py
   ```

---

## 📊 Performance

- **CPU:** 5-15 FPS
- **GPU (FP16):** 50-80+ FPS ⚡

---

For detailed instructions, see `GPU_SETUP.md`

