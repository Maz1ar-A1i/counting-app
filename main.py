

import sys
import os
import torch

# Add modules directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.threaded_cam import ThreadedVideoCapture
from modules.detector import ObjectDetector
from modules.ui import ObjectDetectionGUI


def cleanup_resources(camera, detector):
 
    print("\n" + "="*60)
    print("CLEANING UP RESOURCES")
    print("="*60)
    
    # Stop camera
    if camera:
        print("Stopping camera...")
        camera.stop()
    
    # Clear GPU memory if using CUDA
    if detector and detector.device == 'cuda':
        print("Clearing GPU memory...")
        torch.cuda.empty_cache()
    
    print("[OK] Cleanup complete")
    print("="*60 + "\n")


def main():
    """Main application entry point."""
    print("="*60)
    print("Real-Time Object Detection & Counting System")
    print("="*60)
    print("\nInitializing components...")
    
    # Auto-select GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"[INFO] CUDA available: {torch.cuda.get_device_name(0)}")
        print("[OK] Using GPU")
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
    else:
        print("[INFO] CUDA not available, using CPU")
    
    # Initialize components
    camera = None
    detector = None
    gui = None
    
    try:
        # Initialize camera (not started yet)
        print("\n1. Initializing camera module...")
        camera = ThreadedVideoCapture(source=0, width=640, height=360)
        print("[OK] Camera module ready")
        
        # Initialize detector
        print("\n2. Initializing detector...")
        model_path = 'yolov8n.pt'  # Default model
        if os.path.exists('yolov8n.pt'):
            model_path = 'yolov8n.pt'
        elif os.path.exists('yolov8s.pt'):
            model_path = 'yolov8s.pt'
        
        detector = ObjectDetector(
            model_path=model_path,
            confidence=0.5,
            iou=0.45,
            device=device,
            img_size=640
        )
        
        if not detector.load_model():
            print("[ERROR] Failed to load model. Exiting.")
            return
        
        print("[OK] Detector ready")
        
        # Initialize GUI
        print("\n3. Initializing GUI...")
        
        def on_quit():
            """Callback for GUI quit."""
            cleanup_resources(camera, detector)
        
        gui = ObjectDetectionGUI(detector, camera, on_quit)
        print("[OK] GUI ready")
        
        print("\n" + "="*60)
        print("SYSTEM READY")
        print("="*60)
        print("\nInstructions:")
        print("  • Click 'Start Camera' to begin")
        print("  • Press 'L' to enable line drawing mode")
        print("  • Draw a line by clicking and dragging, or clicking twice")
        print("  • Adjust confidence and NMS IoU sliders as needed")
        print("  • Press 'Q' to quit")
        print("="*60 + "\n")
        
        # Run GUI
        gui.run()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Application error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        cleanup_resources(camera, detector)
        print("Application closed. Goodbye!")


if __name__ == "__main__":
    main()
