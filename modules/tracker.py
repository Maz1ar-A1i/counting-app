# modules/tracker.py
"""
Object tracking module using DeepSORT.
"""

from deep_sort_realtime.deepsort_tracker import DeepSort
from config import TRACKING_CONFIG

class ObjectTracker:
    def __init__(self):
        """Initialize DeepSORT tracker."""
        self.tracker = DeepSort(
            max_age=TRACKING_CONFIG['max_age'],
            n_init=TRACKING_CONFIG['min_hits'],
            nms_max_overlap=TRACKING_CONFIG['iou_threshold'],
            max_cosine_distance=0.3,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=False,
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )
        
        self.tracked_objects = {}  # Dictionary to store tracked object info
        print("[OK] DeepSORT tracker initialized")
    
    def update(self, detections, frame):
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dictionaries
            frame: Current frame (for feature extraction)
            
        Returns:
            list: Tracked objects with format:
                  [{'track_id': int, 'bbox': [x1, y1, x2, y2], 'class_id': int, 
                    'class_name': str, 'confidence': float, 'center': (cx, cy)}]
        """
        if not detections:
            # Update tracker with empty detections to handle disappeared objects
            self.tracker.update_tracks([], frame=frame)
            return []
        
        # Prepare detections in DeepSORT format: ([x1, y1, w, h], confidence, class_name)
        deepsort_detections = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w = x2 - x1
            h = y2 - y1
            
            deepsort_detections.append((
                [x1, y1, w, h],
                det['confidence'],
                det['class_name']
            ))
        
        # Update tracker
        tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)
        
        tracked_objects = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()  # Get bounding box in [left, top, right, bottom] format
            
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            
            # Calculate center
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Get class information from detection class
            class_name = track.get_det_class() if track.get_det_class() else 'unknown'
            
            # Find corresponding detection to get class_id and confidence
            class_id = None
            confidence = 0.0
            
            for det in detections:
                det_x1, det_y1, det_x2, det_y2 = det['bbox']
                # Check if this detection matches the track (simple IoU check)
                if (abs(det_x1 - x1) < 50 and abs(det_y1 - y1) < 50 and 
                    abs(det_x2 - x2) < 50 and abs(det_y2 - y2) < 50):
                    class_id = det['class_id']
                    confidence = det['confidence']
                    break
            
            tracked_obj = {
                'track_id': track_id,
                'bbox': [x1, y1, x2, y2],
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'center': (center_x, center_y)
            }
            
            # Store or update tracked object info
            if track_id not in self.tracked_objects:
                self.tracked_objects[track_id] = {
                    'class_name': class_name,
                    'first_seen': True,
                    'trajectory': [center_y]  # Store y-coordinate history for line crossing
                }
            else:
                self.tracked_objects[track_id]['first_seen'] = False
                self.tracked_objects[track_id]['trajectory'].append(center_y)
                
                # Keep only last 10 positions
                if len(self.tracked_objects[track_id]['trajectory']) > 10:
                    self.tracked_objects[track_id]['trajectory'].pop(0)
            
            tracked_objects.append(tracked_obj)
        
        return tracked_objects
    
    def get_object_info(self, track_id):
        """Get stored information about a tracked object."""
        return self.tracked_objects.get(track_id, None)
    
    def reset(self):
        """Reset tracker and clear all tracked objects."""
        self.tracker = DeepSort(
            max_age=TRACKING_CONFIG['max_age'],
            n_init=TRACKING_CONFIG['min_hits'],
            nms_max_overlap=TRACKING_CONFIG['iou_threshold']
        )
        self.tracked_objects = {}
        print("Tracker reset")