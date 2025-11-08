"""
Quick GPU test script to verify CUDA and PyTorch GPU setup.
Run this before starting the main application.
"""

import torch
import sys

print("=" * 60)
print("GPU Setup Verification")
print("=" * 60)

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"\n✅ CUDA Available: {cuda_available}")

if cuda_available:
    print(f"\n📊 GPU Information:")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
    
    props = torch.cuda.get_device_properties(0)
    print(f"   GPU Memory: {props.total_memory / 1024**3:.2f} GB")
    print(f"   Compute Capability: {props.major}.{props.minor}")
    
    print(f"\n🧪 Testing GPU Computation...")
    try:
        # Test GPU computation
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("   ✅ GPU computation test: SUCCESS")
        
        # Test memory allocation
        del x, y, z
        torch.cuda.empty_cache()
        print("   ✅ GPU memory management: SUCCESS")
        
        print(f"\n🎉 GPU is ready to use!")
        print(f"   Your project will automatically use GPU for faster inference.")
        sys.exit(0)
        
    except Exception as e:
        print(f"   ❌ GPU computation test FAILED: {e}")
        print(f"\n⚠️  GPU is detected but not working properly.")
        sys.exit(1)
else:
    print(f"\n❌ CUDA not available. Your project will use CPU (slower).")
    print(f"\n📋 Troubleshooting Steps:")
    print(f"   1. Check if you have an NVIDIA GPU:")
    print(f"      → Run: nvidia-smi")
    print(f"   2. Install/update NVIDIA drivers:")
    print(f"      → https://www.nvidia.com/Download/index.aspx")
    print(f"   3. Install CUDA Toolkit:")
    print(f"      → https://developer.nvidia.com/cuda-downloads")
    print(f"   4. Reinstall PyTorch with CUDA:")
    print(f"      → pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print(f"\n💡 The project will still work on CPU, but will be slower.")
    sys.exit(0)

print("=" * 60)

