# modules/threaded_cam.py
"""
Threaded video capture module with bounded queue for smooth real-time performance.
Uses imutils VideoStream plus an additional queue to always deliver the latest frame.
Improved Windows camera handling to avoid MSMF errors.
"""

import cv2
import threading
import time
import sys
import platform
from queue import Queue, Empty

from imutils.video import VideoStream


class ThreadedVideoCapture:
    """
    Threaded video capture class that continuously reads frames in a background thread.
    Uses a bounded queue to store the latest frames, automatically discarding old ones.
    """
    
    def __init__(self, source=0, width=640, height=360, queue_size=2):
        """
        Initialize threaded video capture.
        
        Args:
            source: Camera source (int for webcam index, str for IP camera URL)
            width: Target frame width (will be resized)
            height: Target frame height (will be resized)
            queue_size: Maximum number of frames to keep in queue (default: 2)
        """
        self.source = source
        self.target_width = width
        self.target_height = height
        self.queue_size = queue_size
        
        self.vs = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=queue_size)
        self.lock = threading.Lock()
        
        # FPS tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.last_frame = None
        
        # Thread management
        self.capture_thread = None
        
        # Windows-specific: Use DirectShow backend to avoid MSMF issues
        self.backend = None
        if platform.system() == 'Windows':
            # Try DirectShow first (more reliable on Windows)
            self.backend = cv2.CAP_DSHOW
        else:
            self.backend = cv2.CAP_ANY
    
    def _init_videostream(self, source):
        """Internal helper to initialize VideoStream with optional resolution."""
        try:
            if isinstance(source, int):
                # On Windows, use DirectShow backend to avoid MSMF errors
                if platform.system() == 'Windows':
                    # Use OpenCV directly with DirectShow backend for better Windows compatibility
                    cap = cv2.VideoCapture(source, self.backend)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        return cap
                    else:
                        # Fallback to imutils
                        return VideoStream(src=source, resolution=(self.target_width, self.target_height)).start()
                else:
                    return VideoStream(src=source, resolution=(self.target_width, self.target_height)).start()
            else:
                # Check if it's an RTSP stream
                if isinstance(source, str) and source.startswith('rtsp://'):
                    print(f"[INFO] Detected RTSP stream, configuring for RTSP...")
                    # RTSP stream - use OpenCV directly with RTSP optimizations
                    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                    
                    # RTSP-specific settings for better stability
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer to reduce latency
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                    
                    # Set timeout for RTSP connection (in milliseconds)
                    # Note: OpenCV doesn't directly support timeout, but we'll handle it in the read loop
                    
                    if cap.isOpened():
                        print("[OK] RTSP stream opened successfully")
                        return cap
                    else:
                        print("[WARNING] Failed to open RTSP stream with OpenCV, trying imutils...")
                        # Fallback to imutils
                        return VideoStream(src=source).start()
                else:
                    # IP camera (HTTP/MJPEG) - use imutils
                    print(f"[INFO] Detected IP camera stream...")
                    return VideoStream(src=source).start()
        except Exception as e:
            print(f"[WARNING] VideoStream init error: {e}, trying fallback...")
            # Fallback to standard OpenCV
            if isinstance(source, int):
                cap = cv2.VideoCapture(source)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            elif isinstance(source, str) and source.startswith('rtsp://'):
                # RTSP fallback
                print("[INFO] Trying RTSP fallback with OpenCV...")
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap if cap.isOpened() else None
            return None
    
    def start(self):
        """
        Start the camera and begin frame capture thread.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.is_running:
                self.stop()
            
            # Open camera
            print(f"Connecting to camera source: {self.source}...")
            self.vs = self._init_videostream(self.source)
            
            if self.vs is None:
                raise Exception("Failed to initialize video stream")
            
            # Check if it's OpenCV VideoCapture or imutils VideoStream
            is_opencv_cap = hasattr(self.vs, 'read') and not hasattr(self.vs, 'stop')
            
            if is_opencv_cap:
                # Standard OpenCV VideoCapture
                if not self.vs.isOpened():
                    raise Exception(f"Failed to open camera source: {self.source}")
                print("[OK] Camera opened with OpenCV DirectShow backend")
            else:
                # imutils VideoStream
                time.sleep(1.0)
                frame = self.vs.read()
                if frame is None:
                    # Try fallback
                    if isinstance(self.source, int) and self.source != 0:
                        print(f"Camera {self.source} not found, trying index 0...")
                        if hasattr(self.vs, 'stop'):
                            self.vs.stop()
                        self.source = 0
                        self.vs = self._init_videostream(self.source)
                        time.sleep(1.0)
                        frame = self.vs.read()
                    
                    if frame is None:
                        raise Exception(f"Failed to open camera source: {self.source}")
                print("[OK] VideoStream initialized via imutils")
            
            # Start capture thread
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to start camera: {e}")
            if self.vs is not None:
                try:
                    if hasattr(self.vs, 'stop'):
                        self.vs.stop()
                    else:
                        self.vs.release()
                except:
                    pass
                self.vs = None
            return False
    
    def _capture_loop(self):
        """Internal method: continuously capture frames in background thread."""
        is_opencv_cap = hasattr(self.vs, 'read') and not hasattr(self.vs, 'stop')
        consecutive_errors = 0
        max_errors = 10
        is_rtsp = isinstance(self.source, str) and self.source.startswith('rtsp://')
        
        while self.is_running and self.vs is not None:
            try:
                if is_opencv_cap:
                    # Standard OpenCV VideoCapture (including RTSP)
                    ret, frame = self.vs.read()
                    if not ret or frame is None:
                        consecutive_errors += 1
                        if consecutive_errors > max_errors:
                            if is_rtsp:
                                print("[ERROR] RTSP stream connection lost. Too many consecutive read failures.")
                            else:
                                print("[ERROR] Too many consecutive read failures, stopping capture")
                            break
                        # For RTSP, wait a bit longer before retrying
                        time.sleep(0.1 if is_rtsp else 0.01)
                        continue
                    consecutive_errors = 0
                else:
                    # imutils VideoStream
                    frame = self.vs.read()
                    if frame is None:
                        consecutive_errors += 1
                        if consecutive_errors > max_errors:
                            print("[ERROR] Too many consecutive read failures, stopping capture")
                            break
                        time.sleep(0.01)
                        continue
                    consecutive_errors = 0
                
                if frame is not None:
                    # Resize frame for performance/consistency
                    if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
                        frame = cv2.resize(frame, (self.target_width, self.target_height))
                    
                    # Update FPS
                    self.frame_count += 1
                    elapsed = time.time() - self.start_time
                    if elapsed > 0:
                        self.fps = self.frame_count / elapsed
                    
                    # Store latest frame
                    with self.lock:
                        self.last_frame = frame.copy()
                    
                    # Add to queue (discard oldest if full)
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except Empty:
                            pass
                    
                    try:
                        self.frame_queue.put_nowait(frame)
                    except Exception:
                        pass
            except Exception as e:
                # Suppress MSMF warnings - they're common on Windows and don't affect functionality
                if 'MSMF' not in str(e) and 'grabFrame' not in str(e):
                    print(f"[WARNING] Capture error: {e}")
                consecutive_errors += 1
                if consecutive_errors > max_errors:
                    break
                time.sleep(0.01)
        
        print("[INFO] Capture loop ended")
    
    def read(self):
        """
        Read the latest frame from the queue (non-blocking).
        
        Returns:
            tuple: (success: bool, frame: numpy array or None)
        """
        if not self.is_running:
            return False, None
        
        try:
            frame = self.frame_queue.get_nowait()
            return True, frame
        except Empty:
            with self.lock:
                if self.last_frame is not None:
                    return True, self.last_frame.copy()
            return False, None
    
    def get_fps(self):
        """Get current FPS."""
        return self.fps
    
    def get_frame_size(self):
        """Get current frame dimensions."""
        return self.target_width, self.target_height
    
    def stop(self):
        """Stop capture thread and release camera resources."""
        if not self.is_running and self.vs is None:
            return
        
        print("Stopping camera...")
        self.is_running = False
        
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2.0)
            self.capture_thread = None
        
        if self.vs is not None:
            try:
                if hasattr(self.vs, 'stop'):
                    self.vs.stop()
                else:
                    self.vs.release()
            except Exception as e:
                print(f"[WARNING] Error releasing camera: {e}")
            self.vs = None
            print("[OK] VideoStream released")
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
    
    def __del__(self):
        """Destructor: ensure camera is released."""
        self.stop()
