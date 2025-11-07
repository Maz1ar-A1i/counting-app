# 🚀 Complete Setup Guide - Object Detection & Counting System

## Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Windows, Linux, or macOS
- **Hardware**: 
  - External USB camera (or built-in webcam)
  - Minimum 8GB RAM
  - GPU (optional but recommended for better performance)

---

## Step 1: Environment Setup

### Create Virtual Environment

```bash
# Create a new virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

### Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install opencv-python==4.8.1.78
pip install ultralytics==8.0.196
pip install numpy==1.24.3
pip install deep-sort-realtime==1.3.2
pip install PyQt5==5.15.9
pip install matplotlib==3.7.2
pip install pillow==10.0.0
```

**For GPU Support (NVIDIA only):**
```bash
# Install CUDA-enabled PyTorch (if you have NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Step 2: Project Structure Setup

Create the following directory structure:

```
object_detection_system/
├── main.py
├── config.py
├── requirements.txt
├── models/
│   └── (YOLOv8 model will be downloaded here automatically)
├── modules/
│   ├── __init__.py
│   ├── camera.py
│   ├── detector.py
│   ├── tracker.py
│   ├── counter.py
│   └── ui.py
└── utils/
    ├── __init__.py
    └── helpers.py
```

### Create Empty `__init__.py` Files

```bash
# Create module init files
touch modules/__init__.py
touch utils/__init__.py

# On Windows, use:
# type nul > modules\__init__.py
# type nul > utils\__init__.py
```

---

## Step 3: Camera Setup

### USB Camera Configuration

1. **Connect your external USB camera** to your computer
2. **Identify camera index:**

```python
# test_camera.py - Run this to find your camera index
import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✓ Camera found at index {i}")
        ret, frame = cap.read()
        if ret:
            print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
        cap.release()
    else:
        print(f"✗ No camera at index {i}")
```

3. **Update `config.py`:**
   - Set `CAMERA_CONFIG['source']` to your camera index (usually 0 or 1)
   - For external camera, typically use `1`
   - For built-in webcam, use `0`

### IP Camera Configuration

If using an IP camera:

```python
# In config.py
CAMERA_CONFIG = {
    'source': 'rtsp://username:password@ip_address:port/stream',
    'width': 1280,
    'height': 720,
    'fps': 30
}
```

---

## Step 4: Model Download

The YOLOv8 model will be downloaded automatically on first run. However, you can pre-download it:

```python
# download_model.py
from ultralytics import YOLO

# This will download the model to the default location
model = YOLO('yolov8n.pt')  # Nano (fastest)
# model = YOLO('yolov8s.pt')  # Small (balanced)
# model = YOLO('yolov8m.pt')  # Medium (more accurate)

print("✓ Model downloaded successfully!")
```

**Model Comparison:**

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| yolov8n.pt | 6MB | Fastest | Good | Real-time, low-end hardware |
| yolov8s.pt | 22MB | Fast | Better | Balanced performance |
| yolov8m.pt | 52MB | Moderate | Best | High accuracy needed |

---

## Step 5: Running the System

### Basic Run

```bash
python main.py
```

### Expected Output

```
============================================================
🚀 INITIALIZING OBJECT DETECTION SYSTEM
============================================================

1. Starting camera...
Attempting to connect to camera at index 1...
✓ Camera initialized successfully
  Resolution: 1280x720
  Target FPS: 30

2. Loading detection model...
Loading YOLO model: yolov8n.pt...
✓ Model loaded successfully on cpu

3. Initializing tracker and counter...
✓ DeepSORT tracker initialized
✓ Object counter initialized
✓ All systems ready!
============================================================

✓ System is running!
   - Objects will be detected and tracked automatically
   - Toggle 'Line Crossing' to enable line-based counting
   - Press 'Reset' to clear all counts
   - Close window to exit
```

---

## Step 6: Using the System

### Main Interface Components

1. **Video Feed Panel (Left)**
   - Real-time camera feed
   - Bounding boxes around detected objects
   - Object IDs and class labels
   - Virtual counting line (when enabled)

2. **Dashboard Panel (Right)**
   - Total object count
   - Category-wise breakdown
   - Real-time FPS display
   - Control buttons

### Control Buttons

- **🎯 Enable/Disable Line Crossing**: Toggle between free counting and line-based counting
- **🔄 Reset Counts**: Clear all counts and restart tracking

### Line-Crossing Mode

When enabled:
1. A yellow horizontal line appears across the video feed
2. Only objects crossing this line are counted
3. Each object is counted only once (prevents double-counting)

When disabled:
- All detected objects are counted immediately
- Useful for stationary object counting

---

## Step 7: Customization

### Modify Detection Classes

Edit `config.py`:

```python
# Only detect specific classes
DETECTION_CLASSES = {
    0: 'person',
    2: 'car',
    16: 'dog',
    17: 'cat'
}
```

### Adjust Virtual Line Position

```python
# In config.py
LINE_CONFIG = {
    'enabled': False,
    'line_start': (100, 400),  # Left point (x, y)
    'line_end': (1100, 400),   # Right point (x, y)
    'line_color': (0, 255, 255),
    'line_thickness': 3
}
```

### Change Camera Resolution

```python
CAMERA_CONFIG = {
    'source': 1,
    'width': 1920,  # Full HD
    'height': 1080,
    'fps': 30
}
```

### Adjust Detection Sensitivity

```python
MODEL_CONFIG = {
    'confidence_threshold': 0.3,  # Lower = more detections (0.0-1.0)
    'iou_threshold': 0.45,
    'device': 'cpu'  # or 'cuda' for GPU
}
```

---

## Step 8: Performance Optimization

### For Better FPS

1. **Use GPU acceleration:**
   ```python
   MODEL_CONFIG['device'] = 'cuda'
   ```

2. **Lower camera resolution:**
   ```python
   CAMERA_CONFIG['width'] = 640
   CAMERA_CONFIG['height'] = 480
   ```

3. **Use smaller model:**
   ```python
   MODEL_CONFIG['model_path'] = 'yolov8n.pt'  # Fastest
   ```

4. **Reduce tracking complexity:**
   ```python
   TRACKING_CONFIG['max_age'] = 20  # Lower value
   ```

### For Better Accuracy

1. **Use larger model:**
   ```python
   MODEL_CONFIG['model_path'] = 'yolov8m.pt'
   ```

2. **Higher confidence threshold:**
   ```python
   MODEL_CONFIG['confidence_threshold'] = 0.6
   ```

3. **Higher camera resolution:**
   ```python
   CAMERA_CONFIG['width'] = 1920
   CAMERA_CONFIG['height'] = 1080
   ```

---

## Troubleshooting

### Camera Not Found

**Problem:** "Failed to open camera source"

**Solutions:**
1. Check if camera is properly connected
2. Try different camera indices (0, 1, 2)
3. Check camera permissions
4. On Linux: `sudo usermod -a -G video $USER` (logout required)

### Low FPS

**Problem:** System running slowly

**Solutions:**
1. Lower camera resolution
2. Use YOLOv8n (nano) model
3. Enable GPU if available
4. Close other applications

### Model Download Issues

**Problem:** Model not downloading

**Solutions:**
1. Check internet connection
2. Manually download from: https://github.com/ultralytics/assets/releases
3. Place model in project root directory

### Import Errors

**Problem:** ModuleNotFoundError

**Solutions:**
```bash
# Reinstall all dependencies
pip install --force-reinstall -r requirements.txt

# Check Python version
python --version  # Should be 3.8+
```

---

## Advanced Features

### Export Statistics

Add to `main.py`:

```python
from utils.helpers import save_statistics, export_counts_to_csv

# In process_frame method
if self.frame_count % 100 == 0:  # Every 100 frames
    save_statistics(counts)
    export_counts_to_csv(counts)
```

### Custom Object Classes

To detect custom objects, you need to fine-tune the model:

```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolov8n.pt')

# Train on custom dataset
model.train(
    data='path/to/dataset.yaml',
    epochs=100,
    imgsz=640
)

# Use custom model
MODEL_CONFIG['model_path'] = 'runs/detect/train/weights/best.pt'
```

---

## System Requirements Summary

### Minimum
- CPU: Intel i5 or equivalent
- RAM: 8GB
- Storage: 5GB free space
- Camera: Any USB camera

### Recommended
- CPU: Intel i7 or equivalent
- RAM: 16GB
- GPU: NVIDIA GTX 1060 or better
- Storage: 10GB SSD
- Camera: 1080p USB 3.0 camera

---

## Getting Help

If you encounter issues:

1. Check the console output for error messages
2. Verify all dependencies are installed correctly
3. Test camera separately using `test_camera.py`
4. Check configuration in `config.py`
5. Review the troubleshooting section above

---

## Next Steps

- Experiment with different camera angles
- Test various lighting conditions
- Fine-tune detection thresholds
- Add custom object classes
- Implement data logging and analytics

**Happy Detecting! 🎯**