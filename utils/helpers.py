# utils/helpers.py
"""
Utility functions for the object detection system.
"""

import cv2
import numpy as np
import json
from datetime import datetime

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        float: IoU score
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def line_intersection(line1_start, line1_end, line2_start, line2_end):
    """
    Check if two line segments intersect.
    
    Args:
        line1_start: (x, y) start point of line 1
        line1_end: (x, y) end point of line 1
        line2_start: (x, y) start point of line 2
        line2_end: (x, y) end point of line 2
        
    Returns:
        bool: True if lines intersect
    """
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    A, B = line1_start, line1_end
    C, D = line2_start, line2_end
    
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def save_statistics(counts, filename="statistics.json"):
    """
    Save counting statistics to a JSON file.
    
    Args:
        counts: Dictionary of counts by class
        filename: Output filename
    """
    data = {
        'timestamp': datetime.now().isoformat(),
        'counts': counts,
        'total': sum(counts.values())
    }
    
    try:
        # Load existing data
        try:
            with open(filename, 'r') as f:
                history = json.load(f)
        except FileNotFoundError:
            history = []
        
        # Append new data
        history.append(data)
        
        # Save
        with open(filename, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"✓ Statistics saved to {filename}")
        return True
    except Exception as e:
        print(f"✗ Error saving statistics: {e}")
        return False

def load_statistics(filename="statistics.json"):
    """
    Load statistics from JSON file.
    
    Args:
        filename: Input filename
        
    Returns:
        list: List of statistics entries
    """
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"Error loading statistics: {e}")
        return []

def resize_frame(frame, target_width=None, target_height=None):
    """
    Resize frame while maintaining aspect ratio.
    
    Args:
        frame: Input frame
        target_width: Target width (optional)
        target_height: Target height (optional)
        
    Returns:
        np.ndarray: Resized frame
    """
    height, width = frame.shape[:2]
    
    if target_width is None and target_height is None:
        return frame
    
    if target_width is not None:
        ratio = target_width / width
        new_height = int(height * ratio)
        return cv2.resize(frame, (target_width, new_height))
    
    if target_height is not None:
        ratio = target_height / height
        new_width = int(width * ratio)
        return cv2.resize(frame, (new_width, target_height))

def draw_polygon(frame, points, color=(0, 255, 0), thickness=2):
    """
    Draw a polygon on the frame.
    
    Args:
        frame: Input frame
        points: List of (x, y) points
        color: BGR color tuple
        thickness: Line thickness
    """
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, color, thickness)

def check_point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon.
    
    Args:
        point: (x, y) tuple
        polygon: List of (x, y) points
        
    Returns:
        bool: True if point is inside polygon
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def create_heatmap(detections_history, frame_shape):
    """
    Create a heatmap visualization from detection history.
    
    Args:
        detections_history: List of detection center points
        frame_shape: (height, width) of frame
        
    Returns:
        np.ndarray: Heatmap image
    """
    height, width = frame_shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    for point in detections_history:
        x, y = point
        if 0 <= x < width and 0 <= y < height:
            # Add gaussian blob around point
            cv2.circle(heatmap, (x, y), 30, 1.0, -1)
    
    # Normalize and apply colormap
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap_color
    
    return np.zeros((height, width, 3), dtype=np.uint8)

def export_counts_to_csv(counts, filename="object_counts.csv"):
    """
    Export counts to CSV file.
    
    Args:
        counts: Dictionary of counts by class
        filename: Output filename
    """
    import csv
    
    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Class', 'Count'])
            for class_name, count in sorted(counts.items()):
                writer.writerow([class_name, count])
        
        print(f"✓ Counts exported to {filename}")
        return True
    except Exception as e:
        print(f"✗ Error exporting to CSV: {e}")
        return False

def calculate_fps(frame_times):
    """
    Calculate average FPS from frame timestamps.
    
    Args:
        frame_times: List of timestamps
        
    Returns:
        float: Average FPS
    """
    if len(frame_times) < 2:
        return 0.0
    
    time_diffs = np.diff(frame_times)
    avg_time = np.mean(time_diffs)
    
    return 1.0 / avg_time if avg_time > 0 else 0.0