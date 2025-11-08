"""
GPU Setup Script for Object Detection System
This script will install PyTorch with CUDA support for GPU acceleration.
"""

import subprocess
import sys
import platform

def check_nvidia_gpu():
    """Check if NVIDIA GPU is available."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected!")
            print(result.stdout.split('\n')[8])  # Show GPU info
            return True
        else:
            print("❌ nvidia-smi not found. NVIDIA drivers may not be installed.")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found. NVIDIA drivers may not be installed.")
        return False

def install_pytorch_cuda():
    """Install PyTorch with CUDA support."""
    print("\n" + "=" * 60)
    print("Installing PyTorch with CUDA Support")
    print("=" * 60)
    
    # Try CUDA 12.1 first (latest)
    print("\n[1/2] Attempting to install PyTorch with CUDA 12.1...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            'torch', 'torchvision', 'torchaudio',
            '--index-url', 'https://download.pytorch.org/whl/cu121'
        ])
        print("✅ PyTorch with CUDA 12.1 installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  CUDA 12.1 installation failed, trying CUDA 11.8...")
    
    # Fallback to CUDA 11.8
    print("\n[2/2] Attempting to install PyTorch with CUDA 11.8...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            'torch', 'torchvision', 'torchaudio',
            '--index-url', 'https://download.pytorch.org/whl/cu118'
        ])
        print("✅ PyTorch with CUDA 11.8 installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install PyTorch with CUDA support.")
        print("   Installing CPU-only version as fallback...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                'torch', 'torchvision', 'torchaudio',
                '--index-url', 'https://download.pytorch.org/whl/cpu'
            ])
            print("✅ PyTorch (CPU) installed as fallback.")
            return False
        except:
            print("❌ Failed to install PyTorch.")
            return False

def verify_gpu_setup():
    """Verify GPU setup after installation."""
    print("\n" + "=" * 60)
    print("Verifying GPU Setup")
    print("=" * 60)
    
    try:
        import torch
        print(f"\n✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"✅ CUDA Version: {torch.version.cuda}")
            print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            
            # Test GPU computation
            print("\n🧪 Testing GPU computation...")
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            del x, y, z
            torch.cuda.empty_cache()
            print("✅ GPU test: SUCCESS")
            
            print("\n🎉 GPU setup complete! Your project will use GPU acceleration.")
            return True
        else:
            print("\n⚠️  CUDA not available. The project will use CPU (slower).")
            print("   Make sure:")
            print("   1. NVIDIA drivers are installed (run: nvidia-smi)")
            print("   2. CUDA toolkit is installed")
            print("   3. PyTorch was installed with CUDA support")
            return False
    except ImportError:
        print("❌ PyTorch not installed. Please run the installation again.")
        return False
    except Exception as e:
        print(f"❌ Error verifying GPU: {e}")
        return False

def main():
    """Main setup function."""
    print("=" * 60)
    print("GPU Setup for Object Detection System")
    print("=" * 60)
    
    # Check for NVIDIA GPU
    print("\n[Step 1] Checking for NVIDIA GPU...")
    has_gpu = check_nvidia_gpu()
    
    if not has_gpu:
        print("\n⚠️  No NVIDIA GPU detected.")
        response = input("Continue with CPU-only installation? (y/n): ")
        if response.lower() != 'y':
            print("Installation cancelled.")
            return
    
    # Install PyTorch with CUDA
    print("\n[Step 2] Installing PyTorch...")
    cuda_installed = install_pytorch_cuda()
    
    # Verify setup
    print("\n[Step 3] Verifying installation...")
    verify_gpu_setup()
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run: python test_gpu.py (to verify GPU)")
    print("2. Run: python backend/server.py (to start the server)")
    print("\n")

if __name__ == '__main__':
    main()

