# config.py
"""
System configuration file for object detection and counting system.
"""

# Camera Configuration
CAMERA_CONFIG = {
    'source': 0,  # 0 for default webcam, 1 for external USB camera, or 'rtsp://...' for IP camera
    'width': 1280,
    'height': 720,
    'fps': 30
}

# Model Configuration
MODEL_CONFIG = {
    'model_path': 'yolov8n.pt',  # Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large), yolov8x.pt (xlarge)
    'confidence_threshold': 0.5,  # Lower = more detections, Higher = fewer but more confident
    'iou_threshold': 0.45,  # Non-maximum suppression threshold (lower = stricter)
    'device': 'cpu',  # 'cpu' or 'cuda' for GPU
    'img_size': 640  # Input image size (smaller = faster, larger = more accurate)
}

# Tracking Configuration
TRACKING_CONFIG = {
    'max_age': 30,  # Maximum frames to keep track without detection
    'min_hits': 3,  # Minimum consecutive detections before tracking
    'iou_threshold': 0.3
}

# Line Crossing Configuration
LINE_CONFIG = {
    'enabled': False,  # Toggle via UI
    'line_start': (200, 360),  # (x, y) - will be set via UI
    'line_end': (1080, 360),   # (x, y) - will be set via UI
    'line_color': (0, 255, 255),  # Yellow in BGR
    'line_thickness': 3
}

# UI Configuration
UI_CONFIG = {
    'window_title': 'Real-Time Object Detection & Counting System',
    'dashboard_width': 400,
    'font_scale': 0.6,
    'font_thickness': 2,
    'bbox_thickness': 2,
    'show_fps': True,
    'show_confidence': True
}

# Classes to detect and count (COCO dataset classes)
# You can customize this list based on your needs
DETECTION_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    6: 'train',
    7: 'truck',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    24: 'backpack',
    26: 'handbag',
    28: 'suitcase',
    39: 'bottle',
    41: 'cup',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone'
}

# Color palette for different classes (BGR format)
CLASS_COLORS = {
    'person': (255, 0, 0),      # Blue
    'car': (0, 255, 0),          # Green
    'dog': (0, 0, 255),          # Red
    'cat': (255, 0, 255),        # Magenta
    'laptop': (255, 255, 0),     # Cyan
    'default': (0, 255, 255)     # Yellow
}