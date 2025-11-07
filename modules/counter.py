# modules/counter.py
"""
Object counting module with line-crossing detection.
"""

import numpy as np
from collections import defaultdict
from config import LINE_CONFIG

class ObjectCounter:
    def __init__(self):
        """Initialize object counter."""
        self.counts = defaultdict(int)  # Count by class name
        self.total_count = 0
        self.counted_ids = set()  # Track which objects have been counted
        
        # Line crossing configuration
        self.line_crossing_enabled = LINE_CONFIG['enabled']
        self.line_start = LINE_CONFIG['line_start']
        self.line_end = LINE_CONFIG['line_end']
        
        # Track object positions for line crossing
        self.object_positions = {}  # {track_id: {'prev_y': y, 'crossed': bool}}
        
        print("[OK] Object counter initialized")
    
    def set_line(self, start_point, end_point):
        """
        Set virtual line for line-crossing detection.
        
        Args:
            start_point: (x, y) tuple for line start
            end_point: (x, y) tuple for line end
        """
        self.line_start = start_point
        self.line_end = end_point
        print(f"Line set: {start_point} to {end_point}")
    
    def toggle_line_crossing(self):
        """Toggle line-crossing mode on/off."""
        self.line_crossing_enabled = not self.line_crossing_enabled
        status = "enabled" if self.line_crossing_enabled else "disabled"
        print(f"Line-crossing mode {status}")
        return self.line_crossing_enabled
    
    def check_line_crossing(self, track_id, center_y, class_name):
        """
        Check if an object has crossed the virtual line.
        
        Args:
            track_id: Unique ID of tracked object
            center_y: Current y-coordinate of object center
            class_name: Class name of the object
            
        Returns:
            bool: True if object just crossed the line
        """
        # Get line y-coordinate (assuming horizontal line)
        line_y = self.line_start[1]
        
        # Initialize position tracking for new objects
        if track_id not in self.object_positions:
            self.object_positions[track_id] = {
                'prev_y': center_y,
                'crossed': False,
                'class_name': class_name
            }
            return False
        
        prev_y = self.object_positions[track_id]['prev_y']
        already_crossed = self.object_positions[track_id]['crossed']
        
        # Check if object crossed the line (from top to bottom or bottom to top)
        crossed_down = prev_y < line_y and center_y >= line_y
        crossed_up = prev_y > line_y and center_y <= line_y
        
        line_crossed = (crossed_down or crossed_up) and not already_crossed
        
        if line_crossed:
            self.object_positions[track_id]['crossed'] = True
            print(f"[OK] Object {track_id} ({class_name}) crossed the line!")
        
        # Update previous position
        self.object_positions[track_id]['prev_y'] = center_y
        
        return line_crossed
    
    def update(self, tracked_objects):
        """
        Update counts based on tracked objects.
        
        Args:
            tracked_objects: List of tracked object dictionaries
        """
        for obj in tracked_objects:
            track_id = obj['track_id']
            class_name = obj['class_name']
            center_x, center_y = obj['center']
            
            # Skip if already counted
            if track_id in self.counted_ids:
                continue
            
            should_count = False
            
            if self.line_crossing_enabled:
                # Only count if object crosses the line
                if self.check_line_crossing(track_id, center_y, class_name):
                    should_count = True
            else:
                # Count all detected objects (once per unique ID)
                should_count = True
            
            if should_count:
                self.counts[class_name] += 1
                self.total_count += 1
                self.counted_ids.add(track_id)
                print(f"Counted: {class_name} (ID: {track_id}) - Total {class_name}: {self.counts[class_name]}")
    
    def get_counts(self):
        """
        Get current counts by category.
        
        Returns:
            dict: Dictionary of counts by class name
        """
        return dict(self.counts)
    
    def get_total_count(self):
        """Get total count across all categories."""
        return self.total_count
    
    def reset_counts(self):
        """Reset all counts."""
        self.counts = defaultdict(int)
        self.total_count = 0
        self.counted_ids = set()
        self.object_positions = {}
        print("Counts reset")
    
    def get_line_coords(self):
        """Get current line coordinates."""
        return self.line_start, self.line_end
    
    def is_line_crossing_enabled(self):
        """Check if line-crossing mode is enabled."""
        return self.line_crossing_enabled