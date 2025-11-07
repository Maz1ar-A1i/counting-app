# modules/threaded_cam.py
"""
Threaded video capture module with bounded queue for smooth real-time performance.
Uses imutils VideoStream plus an additional queue to always deliver the latest frame.
"""

import cv2
import threading
import time
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

    def _init_videostream(self, source):
        """Internal helper to initialize VideoStream with optional resolution."""
        if isinstance(source, int):
            return VideoStream(src=source, resolution=(self.target_width, self.target_height)).start()
        return VideoStream(src=source).start()

    def start(self):
        """
        Start the camera and begin frame capture thread.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.is_running:
                self.stop()

            # Open camera via imutils VideoStream
            print(f"Connecting to camera source: {self.source}...")
            self.vs = self._init_videostream(self.source)

            # Warm-up
            time.sleep(1.0)
            frame = self.vs.read()

            # Validate first frame, try fallback if needed
            if frame is None and isinstance(self.source, int) and self.source != 0:
                print(f"Camera {self.source} not found, trying index 0...")
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
                self.vs.stop()
                self.vs = None
            return False

    def _capture_loop(self):
        """Internal method: continuously capture frames in background thread."""
        while self.is_running and self.vs is not None:
            frame = self.vs.read()

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
            else:
                time.sleep(0.01)  # Small delay on read failure

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
            self.vs.stop()
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

