# modules/detector.py
"""
YOLO object detection module with ByteTrack tracking and line-cross counting logic.
Handles detection, tracking, and line-crossing detection in a unified pipeline.
"""

import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import math


class SimpleTracker:
    """
    Simple object tracker using IoU matching (similar to SORT).
    Tracks objects across frames and maintains stable IDs.
    """
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Initialize tracker.
        
        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum detections before track is confirmed
            iou_threshold: IoU threshold for matching detections to tracks
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks = []  # List of active tracks
        self.frame_count = 0
        self.next_id = 1
        
    def _iou(self, box1, box2):
        """Calculate IoU between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def update(self, detections):
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections, each with 'bbox' [x1, y1, x2, y2]
        
        Returns:
            List of tracked objects with 'track_id', 'bbox', and other detection info
        """
        self.frame_count += 1
        
        # Update existing tracks
        for track in self.tracks:
            track['age'] += 1
            track['time_since_update'] += 1
        
        # Match detections to tracks
        matched_tracks = []
        matched_detections = []
        
        if len(self.tracks) > 0 and len(detections) > 0:
            # Calculate IoU matrix
            iou_matrix = np.zeros((len(self.tracks), len(detections)))
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    iou_matrix[i, j] = self._iou(track['bbox'], det['bbox'])
            
            # Greedy matching
            while True:
                max_iou = np.max(iou_matrix)
                if max_iou < self.iou_threshold:
                    break
                
                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                matched_tracks.append(i)
                matched_detections.append(j)
                
                # Remove matched row and column
                iou_matrix[i, :] = -1
                iou_matrix[:, j] = -1
        
        # Update matched tracks
        for i, j in zip(matched_tracks, matched_detections):
            track = self.tracks[i]
            det = detections[j]
            track['bbox'] = det['bbox']
            track['class_id'] = det['class_id']
            track['class_name'] = det['class_name']
            track['confidence'] = det['confidence']
            track['center'] = det['center']
            track['time_since_update'] = 0
            track['hits'] += 1
        
        # Create new tracks for unmatched detections
        unmatched_detections = [j for j in range(len(detections)) if j not in matched_detections]
        for j in unmatched_detections:
            det = detections[j]
            new_track = {
                'track_id': self.next_id,
                'bbox': det['bbox'],
                'class_id': det['class_id'],
                'class_name': det['class_name'],
                'confidence': det['confidence'],
                'center': det['center'],
                'age': 1,
                'time_since_update': 0,
                'hits': 1
            }
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t['time_since_update'] < self.max_age]
        
        # Return confirmed tracks only
        confirmed_tracks = []
        for track in self.tracks:
            if track['hits'] >= self.min_hits:
                confirmed_tracks.append({
                    'track_id': track['track_id'],
                    'bbox': track['bbox'],
                    'class_id': track['class_id'],
                    'class_name': track['class_name'],
                    'confidence': track['confidence'],
                    'center': track['center']
                })
        
        return confirmed_tracks
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.frame_count = 0
        self.next_id = 1


class ObjectDetector:
    """
    YOLO-based object detector with integrated tracking and line-cross counting.
    """
    
    def __init__(self, model_path='yolov8n.pt', confidence=0.5, iou=0.45, device='cpu', img_size=640):
        """
        Initialize detector.
        
        Args:
            model_path: Path to YOLO model weights
            confidence: Confidence threshold
            iou: IoU threshold for NMS
            device: 'cpu' or 'cuda'
            img_size: Input image size
        """
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.img_size = img_size
        self.model = None
        self.use_half = False  # Enable FP16 on CUDA for faster inference
        
        # Initialize tracker
        self.tracker = SimpleTracker(max_age=30, min_hits=3, iou_threshold=0.3)
        
        # Line-crossing state
        self.line_start = None
        self.line_end = None
        self.line_crossing_enabled = False
        self.crossing_direction = 'both'  # 'both', 'up_down', 'down_up'
        
        # Track object trajectories for line crossing
        self.track_history = defaultdict(lambda: deque(maxlen=10))  # {track_id: deque of (x, y)}
        self.crossed_tracks = set()  # Track IDs that have already crossed
        
        # Detection classes (COCO dataset)
        self.class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus',
            6: 'train', 7: 'truck', 14: 'bird', 15: 'cat', 16: 'dog',
            24: 'backpack', 26: 'handbag', 28: 'suitcase', 39: 'bottle',
            41: 'cup', 56: 'chair', 57: 'couch', 58: 'potted plant',
            59: 'bed', 60: 'dining table', 62: 'tv', 63: 'laptop',
            64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone'
        }
        
    def load_model(self):
        """Load YOLO model."""
        try:
            print(f"Loading YOLO model: {self.model_path}...")
            self.model = YOLO(self.model_path)

            if self.device == 'cuda':
                try:
                    self.model.to('cuda')
                    self.use_half = True
                    # Fuse model for slightly faster inference when supported
                    try:
                        self.model.fuse()
                    except Exception:
                        pass
                    print("[OK] Using GPU (FP16)")
                except Exception:
                    print("[WARNING] CUDA not available, using CPU")
                    self.device = 'cpu'
                    self.use_half = False
            else:
                print("[OK] Using CPU")

            # Warm up model (match size)
            warm_h = self.img_size
            warm_w = self.img_size
            dummy = np.zeros((warm_h, warm_w, 3), dtype=np.uint8)
            _ = self.model(dummy, verbose=False)
            
            print(f"[OK] Model loaded successfully")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return False
    
    def set_confidence(self, confidence):
        """Update confidence threshold."""
        self.confidence = max(0.1, min(0.95, confidence))
    
    def set_iou(self, iou):
        """Update IoU threshold."""
        self.iou = max(0.1, min(0.9, iou))
    
    def detect(self, frame):
        """
        Perform object detection on frame.
        
        Args:
            frame: Input frame (numpy array)
        
        Returns:
            List of detections with bbox, class_id, class_name, confidence, center
        """
        if self.model is None:
            return []
        
        try:
            # Run YOLO inference
            results = self.model(
                frame,
                conf=self.confidence,
                iou=self.iou,
                imgsz=self.img_size,
                verbose=False,
                half=self.use_half,
                agnostic_nms=False,
                max_det=200,
                classes=list(self.class_names.keys())
            )
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                
                boxes_xyxy = boxes.xyxy.cpu().numpy()
                boxes_conf = boxes.conf.cpu().numpy()
                boxes_cls = boxes.cls.cpu().numpy()
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes_xyxy[i]
                    class_id = int(boxes_cls[i])
                    confidence = float(boxes_conf[i])
                    
                    # Filter by class (only detect classes in our list)
                    if class_id not in self.class_names:
                        continue
                    
                    class_name = self.class_names[class_id]
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'center': (center_x, center_y)
                    })
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def _line_segment_intersection(self, p1, p2, p3, p4):
        """
        Check if line segment p1-p2 intersects line segment p3-p4.
        Returns intersection point if they intersect, None otherwise.
        
        Args:
            p1, p2: Endpoints of first line segment
            p3, p4: Endpoints of second line segment
        
        Returns:
            (x, y) tuple if intersection exists, None otherwise
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # Calculate denominators
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None  # Lines are parallel
        
        # Calculate intersection point
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        # Check if intersection is within both segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (int(x), int(y))
        
        return None
    
    def _point_to_line_distance(self, point, line_start, line_end):
        """
        Calculate signed distance from point to line.
        Positive = on one side, negative = on other side.
        """
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Vector from line_start to line_end
        dx = x2 - x1
        dy = y2 - y1
        
        # Vector from line_start to point
        px_dx = px - x1
        px_dy = py - y1
        
        # Cross product to determine side
        cross = dx * px_dy - dy * px_dx
        return cross
    
    def process_frame(self, frame):
        """
        Process frame: detect, track, and check line crossings.
        
        Args:
            frame: Input frame
        
        Returns:
            dict with:
                - tracked_objects: List of tracked objects
                - line_crosses: List of (track_id, class_name, intersection_point) for new crosses
                - counts: Dict of counts by class
        """
        # Detect objects
        detections = self.detect(frame)
        
        # Update tracker
        tracked_objects = self.tracker.update(detections)
        
        # Update track history and check line crossings
        line_crosses = []
        counts = defaultdict(int)
        
        if self.line_crossing_enabled and self.line_start is not None and self.line_end is not None:
            for obj in tracked_objects:
                track_id = obj['track_id']
                center = obj['center']
                
                # Update track history
                self.track_history[track_id].append(center)
                
                # Check for line crossing if we have history
                if len(self.track_history[track_id]) >= 2 and track_id not in self.crossed_tracks:
                    prev_center = self.track_history[track_id][-2]
                    curr_center = self.track_history[track_id][-1]
                    
                    # Check if trajectory crosses the line
                    intersection = self._line_segment_intersection(
                        prev_center, curr_center,
                        self.line_start, self.line_end
                    )
                    
                    if intersection is not None:
                        # Check direction if needed
                        if self.crossing_direction != 'both':
                            prev_dist = self._point_to_line_distance(prev_center, self.line_start, self.line_end)
                            curr_dist = self._point_to_line_distance(curr_center, self.line_start, self.line_end)
                            
                            if self.crossing_direction == 'up_down':
                                # Only count if crossing from negative to positive
                                if prev_dist >= 0 or curr_dist < 0:
                                    continue
                            elif self.crossing_direction == 'down_up':
                                # Only count if crossing from positive to negative
                                if prev_dist <= 0 or curr_dist > 0:
                                    continue
                        
                        # Mark as crossed and count
                        self.crossed_tracks.add(track_id)
                        line_crosses.append((track_id, obj['class_name'], intersection))
                        counts[obj['class_name']] += 1
        else:
            # Free counting mode: count all tracked objects once
            for obj in tracked_objects:
                track_id = obj['track_id']
                center = obj['center']
                
                # Update track history
                self.track_history[track_id].append(center)
                
                # Count each track ID only once
                if track_id not in self.crossed_tracks:
                    self.crossed_tracks.add(track_id)
                    counts[obj['class_name']] += 1
        
        return {
            'tracked_objects': tracked_objects,
            'line_crosses': line_crosses,
            'counts': dict(counts)
        }
    
    def set_line(self, start_point, end_point):
        """Set counting line coordinates."""
        self.line_start = start_point
        self.line_end = end_point
        # Reset crossed tracks when line changes
        self.crossed_tracks.clear()
        print(f"Line set: {start_point} -> {end_point}")
    
    def reset_line(self):
        """Reset line and clear crossing history."""
        self.line_start = None
        self.line_end = None
        self.crossed_tracks.clear()
        self.track_history.clear()
        print("Line reset")
    
    def toggle_line_crossing(self):
        """Toggle line-crossing mode."""
        self.line_crossing_enabled = not self.line_crossing_enabled
        if not self.line_crossing_enabled:
            self.crossed_tracks.clear()
        return self.line_crossing_enabled
    
    def set_crossing_direction(self, direction):
        """Set crossing direction: 'both', 'up_down', or 'down_up'."""
        self.crossing_direction = direction
    
    def reset_counts(self):
        """Reset all counts and tracking."""
        self.crossed_tracks.clear()
        self.track_history.clear()
        self.tracker.reset()
    
    def get_line_coords(self):
        """Get current line coordinates."""
        return self.line_start, self.line_end
