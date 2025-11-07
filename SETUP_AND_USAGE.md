# Object Detection & Counting System - Setup & Usage Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Windows 10/11 (or Linux/Mac with minor adjustments)
- Webcam or USB camera
- 4GB+ RAM recommended

### Installation

1. **Activate your virtual environment:**
   ```powershell
   # Windows PowerShell
   .\venv\Scripts\Activate.ps1
   
   # Windows CMD
   venv\Scripts\activate.bat
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Install dependencies (if not already installed):**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install PyTorch CPU-only (if not already installed):**
   ```bash
   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
   pip install intel-openmp mkl
   pip install "numpy<2.0"
   ```

### Running the Application

```bash
python main.py
```

## 📖 Detailed Usage

### Starting the Application

1. **Launch the application:**
   ```bash
   python main.py
   ```

2. **Start the camera:**
   - Click the **"Start Camera"** button in the top control panel
   - The camera will initialize and begin capturing frames
   - The video feed will appear in the main window

3. **Detection begins automatically:**
   - Objects are detected and tracked in real-time
   - Bounding boxes and labels appear on detected objects
   - Counts update in the right-side dashboard panel

### UI Controls

#### Top Control Panel

- **Start/Stop Camera Button:**
  - Green "Start Camera" - Click to begin video capture
  - Red "Stop Camera" - Click to stop video capture and release resources

- **Model Selector:**
  - Dropdown menu to select YOLO model:
    - `yolov8n.pt` - Nano (fastest, lower accuracy)
    - `yolov8s.pt` - Small (balanced)
    - `yolov8m.pt` - Medium (better accuracy)
    - `yolov8l.pt` - Large (high accuracy)
    - `yolov8x.pt` - XLarge (best accuracy, slowest)
  - **Note:** Model can only be changed when camera is stopped

- **Load Model Button:**
  - Click to browse and load a custom YOLO model file (.pt)
  - Useful for loading custom-trained models

#### Bottom Control Panel

- **Line Counting Toggle:**
  - Click to enable/disable line-crossing counting mode
  - When enabled, only objects crossing the virtual line are counted
  - The counting line appears as a yellow line across the video

- **Reset Counts Button:**
  - Clears all object counts and resets tracking
  - Useful for starting a new counting session

- **Detection Settings:**
  - **Confidence Threshold:** Adjust detection confidence (0.1 - 0.99)
    - Lower values = more detections (may include false positives)
    - Higher values = fewer but more confident detections
    - Recommended: 0.4 - 0.6
  
  - **IoU Threshold:** Adjust Non-Maximum Suppression threshold (0.1 - 0.99)
    - Lower values = stricter overlap filtering
    - Higher values = allow more overlapping boxes
    - Recommended: 0.4 - 0.5

#### Dashboard Panel (Right Side)

- **Total Objects:** Shows cumulative count of all detected objects
- **Counts by Category:** Breakdown of counts by object class (e.g., Person: 5, Car: 2)
- **FPS Display:** Current frames per second (performance indicator)

### Keyboard Shortcuts

- **Q Key:** Quit the application (cleanly closes camera and exits)
- **Window Close (X):** Also triggers clean shutdown

### Performance Optimization Tips

1. **If video is laggy:**
   - Use a smaller model (`yolov8n.pt` or `yolov8s.pt`)
   - Reduce image size in `config.py` (change `img_size` from 640 to 416)
   - Lower camera resolution in `config.py` (reduce `width` and `height`)

2. **If detection accuracy is poor:**
   - Use a larger model (`yolov8m.pt` or `yolov8l.pt`)
   - Adjust confidence threshold (try 0.3 - 0.4 for more detections)
   - Increase image size in `config.py` (change `img_size` to 832 or 1280)

3. **If FPS is too low (<10):**
   - The system automatically skips frames to maintain responsiveness
   - Consider using GPU acceleration (change `device` to `'cuda'` in `config.py`)
   - Reduce image processing size

## ⚙️ Configuration

### Camera Settings (`config.py`)

```python
CAMERA_CONFIG = {
    'source': 0,      # 0 = default webcam, 1 = USB camera, 'rtsp://...' = IP camera
    'width': 1280,    # Camera resolution width
    'height': 720,    # Camera resolution height
    'fps': 30         # Target frames per second
}
```

### Model Settings (`config.py`)

```python
MODEL_CONFIG = {
    'model_path': 'yolov8n.pt',  # Model file path
    'confidence_threshold': 0.5,  # Detection confidence (0.1-0.99)
    'iou_threshold': 0.45,         # NMS threshold (0.1-0.99)
    'device': 'cpu',               # 'cpu' or 'cuda' for GPU
    'img_size': 640                # Input image size (416, 640, 832, 1280)
}
```

### Detection Classes (`config.py`)

Edit `DETECTION_CLASSES` dictionary to customize which objects to detect:
- Keys are COCO dataset class IDs
- Values are class names
- Remove classes you don't need for better performance

## 🔧 Troubleshooting

### Camera Not Starting

1. **Check camera permissions:**
   - Windows: Settings > Privacy > Camera
   - Ensure camera access is enabled

2. **Try different camera index:**
   - Change `source` in `config.py` from 0 to 1 (or vice versa)
   - For USB cameras, try indices 0, 1, 2, etc.

3. **Check camera is not in use:**
   - Close other applications using the camera (Zoom, Teams, etc.)

### Low FPS / Laggy Video

1. **Reduce processing load:**
   - Use smaller model (`yolov8n.pt`)
   - Lower image size (`img_size: 416`)
   - Lower camera resolution

2. **Check system resources:**
   - Close other applications
   - Check CPU usage in Task Manager

3. **Enable GPU (if available):**
   - Install CUDA-enabled PyTorch
   - Change `device: 'cuda'` in `config.py`

### Poor Detection Accuracy

1. **Adjust thresholds:**
   - Lower confidence threshold (try 0.3-0.4)
   - Adjust IoU threshold (try 0.4-0.5)

2. **Use better model:**
   - Upgrade from `yolov8n.pt` to `yolov8m.pt` or `yolov8l.pt`

3. **Improve lighting:**
   - Ensure good lighting conditions
   - Avoid backlighting

### Application Won't Exit

1. **Use Q key:**
   - Press Q to trigger clean shutdown

2. **Force close:**
   - Close window using X button
   - If stuck, use Task Manager to end process

## 📝 Customization

### Adding Custom Classes

Edit `DETECTION_CLASSES` in `config.py`:
```python
DETECTION_CLASSES = {
    0: 'person',
    2: 'car',
    # Add more COCO classes as needed
}
```

### Changing Colors

Edit `CLASS_COLORS` in `config.py`:
```python
CLASS_COLORS = {
    'person': (255, 0, 0),    # BGR format
    'car': (0, 255, 0),
    'default': (0, 255, 255)
}
```

### Line Counting Position

Edit `LINE_CONFIG` in `config.py`:
```python
LINE_CONFIG = {
    'line_start': (200, 360),  # Start point (x, y)
    'line_end': (1080, 360),   # End point (x, y)
}
```

## 🎯 Best Practices

1. **Start with default settings** and adjust based on your needs
2. **Monitor FPS** - aim for 20+ FPS for smooth operation
3. **Use appropriate model size** - balance between speed and accuracy
4. **Clean exit** - always use Q key or window close button
5. **Test camera first** - ensure camera works before running detection

## 📚 Additional Resources

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [PyQt5 Documentation](https://www.riverbankcomputing.com/static/Docs/PyQt5/)

## 🐛 Reporting Issues

If you encounter issues:
1. Check the console output for error messages
2. Verify all dependencies are installed correctly
3. Ensure camera permissions are granted
4. Check that PyTorch is properly installed (CPU or GPU version)

---

**Enjoy using the Object Detection & Counting System!** 🎉

