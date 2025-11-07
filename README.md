# Real-Time Object Detection & Counting System

A production-ready system for real-time object detection, tracking, and counting using YOLO (Ultralytics) with line-crossing detection capabilities.

## Features

- **Real-time Detection**: Detects multiple object types (people, animals, cars, laptops, shoes, etc.) from webcam or IP camera
- **Object Tracking**: Uses IoU-based tracking to maintain stable object IDs and prevent double-counting
- **Line-Cross Counting**: Interactive line drawing with mouse to count objects crossing a virtual line
- **Direction Detection**: Count objects crossing in specific directions (both, up→down, down→up)
- **Interactive GUI**: Modern Tkinter interface with real-time controls and statistics
- **Performance Optimized**: Multi-threaded frame capture, frame resizing, and adaptive frame skipping
- **GPU/CPU Support**: Works with both CPU and GPU (CUDA) for optimal performance

## System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, Linux, or macOS
- **RAM**: 4GB minimum (8GB recommended)
- **GPU**: Optional but recommended for better performance (NVIDIA GPU with CUDA support)

## Installation

### Step 1: Install Python

Download and install Python 3.8+ from [python.org](https://www.python.org/downloads/). Make sure to check "Add Python to PATH" during installation.

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### Step 3: Install PyTorch

Choose the appropriate PyTorch installation based on your system:

**For CPU-only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**For GPU (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For GPU (CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify PyTorch installation:**
```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Download YOLO Model (Optional)

The system will automatically download `yolov8n.pt` (nano model) on first run. For better accuracy, you can manually download larger models:

- `yolov8n.pt` - Nano (fastest, lowest accuracy)
- `yolov8s.pt` - Small (balanced)
- `yolov8m.pt` - Medium (better accuracy)
- `yolov8l.pt` - Large (high accuracy)
- `yolov8x.pt` - XLarge (best accuracy, slowest)

Place the model file in the project root directory.

## Usage

### Basic Usage

1. **Start the application:**
   ```bash
   python main.py
   ```

2. **Start Camera:**
   - Click "Start Camera" button
   - Enter camera source (0 for default webcam, 1 for USB camera, or IP camera URL)

3. **Adjust Settings:**
   - **Confidence Slider**: Lower = more detections, Higher = fewer but more confident
   - **NMS IoU Slider**: Controls non-maximum suppression (lower = stricter)

4. **Line Counting (Optional):**
   - Press **L** to enable line drawing mode
   - Click and drag on video to draw a line, or click twice (start point, end point)
   - Objects crossing the line will be counted
   - Use "Reset Line" button to clear the line

5. **Quit:**
   - Press **Q** or close the window

### Keyboard Shortcuts

- **L**: Toggle line drawing mode
- **Q**: Quit application

### Camera Sources

- **Webcam Index**: Enter `0` for default webcam, `1` for second camera, etc.
- **IP Camera**: Enter URL like `rtsp://username:password@ip:port/stream` or `http://ip:port/video`

## Performance Tips

### Improving FPS

1. **Use Smaller Model**: `yolov8n.pt` is fastest but less accurate
2. **Reduce Input Size**: Edit `main.py` to change `width=640, height=360` to smaller values
3. **Enable GPU**: Use CUDA-enabled PyTorch for 3-5x speedup
4. **Lower Confidence**: Reduce confidence threshold to process fewer detections
5. **Frame Skipping**: System automatically skips frames if processing is slow

### Improving Accuracy

1. **Use Larger Model**: `yolov8m.pt` or `yolov8x.pt` for better accuracy
2. **Increase Input Size**: Use `img_size=1280` in detector initialization
3. **Adjust Confidence**: Increase confidence threshold to filter false positives
4. **Fine-tune Model**: Train on custom dataset (see Custom Training section)

## Custom Training

### Preparing Custom Dataset

1. **Collect Images**: Gather images of objects you want to detect
2. **Label Images**: Use tools like [LabelImg](https://github.com/tzutalin/labelImg) to create YOLO format annotations
3. **Organize Dataset**:
   ```
   dataset/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── val/
   │   ├── images/
   │   └── labels/
   └── data.yaml
   ```

4. **Create data.yaml**:
   ```yaml
   path: ./dataset
   train: train/images
   val: val/images
   
   names:
     0: person
     1: car
     2: laptop
     # ... your classes
   ```

### Training Command

```bash
# Fine-tune from pre-trained model
yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=100 imgsz=640

# Or use Python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='dataset/data.yaml', epochs=100, imgsz=640)
```

### Using Custom Model

After training, use your custom model:

```python
# In main.py, change:
detector = ObjectDetector(
    model_path='runs/detect/train/weights/best.pt',  # Your trained model
    ...
)
```

## Project Structure

```
Counting/
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── modules/
│   ├── __init__.py
│   ├── threaded_cam.py    # Threaded video capture
│   ├── detector.py        # YOLO detection + tracking + line-cross logic
│   └── ui.py             # Tkinter GUI
└── yolov8n.pt            # YOLO model (auto-downloaded)
```

## Troubleshooting

### Camera Not Opening

- **Check camera permissions**: Ensure camera access is allowed
- **Try different index**: Use 0, 1, 2, etc. for different cameras
- **Check camera in use**: Close other applications using the camera
- **IP Camera**: Verify URL format and network connectivity

### Low FPS

- **Use GPU**: Install CUDA-enabled PyTorch
- **Reduce resolution**: Lower `width` and `height` in `main.py`
- **Use smaller model**: Switch to `yolov8n.pt`
- **Close other applications**: Free up system resources

### Out of Memory Errors

- **Reduce batch size**: Not applicable (processing one frame at a time)
- **Use CPU**: Switch to CPU-only PyTorch if GPU memory is limited
- **Reduce input size**: Lower `img_size` parameter

### Model Download Issues

- **Manual download**: Download model from [Ultralytics](https://github.com/ultralytics/assets/releases)
- **Check internet**: Ensure internet connection for auto-download
- **Firewall**: Check if firewall is blocking downloads

## Advanced Configuration

### Changing Detection Classes

Edit `modules/detector.py` and modify the `class_names` dictionary:

```python
self.class_names = {
    0: 'person',
    2: 'car',
    63: 'laptop',
    # Add/remove classes as needed
}
```

### Adjusting Tracking Parameters

Edit `modules/detector.py` in `SimpleTracker.__init__()`:

```python
self.tracker = SimpleTracker(
    max_age=30,        # Frames to keep track without detection
    min_hits=3,        # Minimum detections before confirming track
    iou_threshold=0.3  # IoU threshold for matching
)
```

### Changing Frame Size

Edit `main.py`:

```python
camera = ThreadedVideoCapture(
    source=0,
    width=1280,   # Change width
    height=720   # Change height
)
```

## License

This project is open source. YOLO models are provided by Ultralytics under AGPL-3.0 license.

## Credits

- **YOLO**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **OpenCV**: Computer vision library
- **PyTorch**: Deep learning framework

## Support

For issues and questions:
1. Check the Troubleshooting section
2. Review error messages in console
3. Verify all dependencies are installed correctly
4. Check camera and system permissions

## Quick Start Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] PyTorch installed (CPU or GPU)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Camera connected and accessible
- [ ] Run `python main.py`
- [ ] Click "Start Camera"
- [ ] Adjust settings as needed
- [ ] Press L to draw counting line (optional)

Enjoy your object detection and counting system!

