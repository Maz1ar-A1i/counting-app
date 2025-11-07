# modules/camera.py
"""
Optimized video capture module with multi-threading for smooth real-time performance.
"""

import cv2
import time
import threading
from queue import Queue, Empty
from config import CAMERA_CONFIG

class CameraStream:
    """
    Threaded camera stream for non-blocking frame capture.
    Uses a separate thread to continuously read frames, preventing lag.
    """
    
    def __init__(self, source=None):
        """
        Initialize camera stream.
        
        Args:
            source: Camera source (int for USB camera, str for IP camera)
        """
        self.source = source if source is not None else CAMERA_CONFIG['source']
        self.cap = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=2)  # Keep only latest 2 frames
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()
        self.last_frame = None
        self.capture_thread = None
        self.lock = threading.Lock()
        
    def start(self):
        """Start the camera stream and capture thread."""
        try:
            # Try to open camera
            if isinstance(self.source, int):
                print(f"Attempting to connect to camera at index {self.source}...")
                self.cap = cv2.VideoCapture(self.source)
                
                # If source doesn't work, try alternative
                if not self.cap.isOpened() and self.source == 1:
                    print("Camera at index 1 not found, trying index 0...")
                    self.source = 0
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.source)
            else:
                # IP camera stream
                print(f"Connecting to IP camera: {self.source}")
                self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                raise Exception(f"Failed to open camera source: {self.source}")
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG['height'])
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG['fps'])
            
            # Optimize camera buffer (reduce latency)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Get actual camera properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            print(f"[OK] Camera initialized successfully")
            print(f"  Resolution: {actual_width}x{actual_height}")
            print(f"  Target FPS: {actual_fps}")
            
            # Start capture thread
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.capture_thread.start()
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Error initializing camera: {e}")
            return False
    
    def _capture_frames(self):
        """Internal method to continuously capture frames in a separate thread."""
        while self.is_running:
            ret, frame = self.cap.read()
            
            if ret:
                # Update FPS calculation
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    if elapsed > 0:
                        self.fps = self.frame_count / elapsed
                
                # Add frame to queue (discard old frames if queue is full)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    try:
                        self.frame_queue.get_nowait()  # Remove oldest frame
                        self.frame_queue.put(frame)     # Add new frame
                    except Empty:
                        pass
                
                with self.lock:
                    self.last_frame = frame
            else:
                time.sleep(0.01)  # Small delay if read fails
    
    def read(self):
        """
        Read the latest frame from the queue (non-blocking).
        
        Returns:
            tuple: (success, frame)
        """
        if not self.is_running:
            return False, None
        
        try:
            # Get latest frame (non-blocking)
            frame = self.frame_queue.get_nowait()
            return True, frame
        except Empty:
            # Return last known frame if queue is empty
            with self.lock:
                if self.last_frame is not None:
                    return True, self.last_frame.copy()
            return False, None
    
    def get_fps(self):
        """Get current FPS."""
        return self.fps
    
    def get_frame_size(self):
        """Get frame dimensions."""
        if self.cap is not None:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        return None, None
    
    def stop(self):
        """Release camera resources and stop capture thread."""
        self.is_running = False
        
        # Wait for thread to finish
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2.0)
        
        # Release camera
        if self.cap is not None:
            self.cap.release()
            print("[OK] Camera released")
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
    
    def __del__(self):
        """Destructor to ensure camera is released."""
        self.stop()
