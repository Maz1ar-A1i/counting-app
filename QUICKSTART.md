# Quick Start Guide

## Installation (5 minutes)

1. **Install Python 3.8+** from [python.org](https://www.python.org/downloads/)

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or: source venv/bin/activate  # Linux/Mac
   ```

3. **Install PyTorch (CPU):**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

```bash
python main.py
```

## Basic Usage

1. Click **"Start Camera"** button
2. Enter camera source (0 for webcam, or IP camera URL)
3. Objects will be detected and counted automatically
4. Press **L** to draw a counting line
5. Click and drag on video to draw line
6. Objects crossing the line will be counted
7. Press **Q** to quit

## Keyboard Shortcuts

- **L** - Toggle line drawing mode
- **Q** - Quit application

## Troubleshooting

**Camera not working?**
- Try different camera index (0, 1, 2...)
- Check if camera is used by another app
- Verify camera permissions

**Low FPS?**
- Use GPU: Install CUDA PyTorch
- Reduce resolution in `main.py` (width/height)
- Use smaller model (yolov8n.pt)

**Model download issues?**
- Check internet connection
- Download manually from Ultralytics GitHub releases

For detailed information, see README.md

