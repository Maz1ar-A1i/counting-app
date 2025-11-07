"""
Diagnostic script to check PyTorch installation and dependencies.
"""
import sys
import os

print("=" * 60)
print("PyTorch Installation Diagnostic")
print("=" * 60)
print()

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path[:3]}")
print()

# Check for torch
print("Checking PyTorch installation...")
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    print(f"✓ PyTorch location: {torch.__file__}")
    
    # Check for DLL
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    c10_dll = os.path.join(torch_lib, "c10.dll")
    if os.path.exists(c10_dll):
        print(f"✓ c10.dll found at: {c10_dll}")
        print(f"  File size: {os.path.getsize(c10_dll) / (1024*1024):.2f} MB")
    else:
        print(f"✗ c10.dll NOT found at: {c10_dll}")
    
    # Try to create a tensor
    try:
        x = torch.tensor([1.0, 2.0, 3.0])
        print(f"✓ Tensor creation successful: {x}")
    except Exception as e:
        print(f"✗ Tensor creation failed: {e}")
        
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("Checking Ultralytics...")
try:
    from ultralytics import YOLO
    print("✓ Ultralytics imported successfully")
except Exception as e:
    print(f"✗ Ultralytics import failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("Checking detector module...")
try:
    from modules.detector import ObjectDetector
    print("✓ Detector module imported successfully")
except Exception as e:
    print(f"✗ Detector module import failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("Diagnostic complete")
print("=" * 60)

