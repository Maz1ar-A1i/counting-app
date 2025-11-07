# modules/ui.py
"""
Tkinter GUI module for real-time object detection and counting.
Provides interactive controls, video display, and line drawing functionality.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
from collections import deque


class ObjectDetectionGUI:
    """
    Main GUI class for object detection and counting system.
    """
    
    def __init__(self, detector, camera, on_quit_callback):
        """
        Initialize GUI.
        
        Args:
            detector: ObjectDetector instance
            camera: ThreadedVideoCapture instance
            on_quit_callback: Callback function to call on quit
        """
        self.detector = detector
        self.camera = camera
        self.on_quit_callback = on_quit_callback
        
        # GUI state
        self.root = tk.Tk()
        self.root.title("Real-Time Object Detection & Counting System")
        self.root.geometry("1400x900")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Video display
        self.video_label = None
        self.current_frame = None
        self.displayed_image_size = None  # (width, height) of displayed image
        self.display_offset = None  # (x, y) offset of displayed image
        
        # Line drawing state
        self.draw_line_mode = False
        self.line_start_point = None
        self.line_end_point = None
        self.drawing_line = False
        self.line_points = []  # For two-click mode
        
        # Statistics
        self.total_count = 0
        self.counts_by_class = {}
        self.current_fps = 0.0
        self.fps_history = deque(maxlen=30)
        
        # Processing state
        self.is_processing = False
        self.processing_thread = None
        
        # Initialize UI
        self._create_widgets()
        self._bind_events()
        
        # Start update loop
        self.update_frame()
    
    def _create_widgets(self):
        """Create and layout all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Left panel: Video and controls
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        main_frame.columnconfigure(0, weight=3)
        main_frame.rowconfigure(0, weight=1)
        
        # Top control panel
        control_top = ttk.Frame(left_panel)
        control_top.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Camera controls
        camera_frame = ttk.LabelFrame(control_top, text="Camera", padding="5")
        camera_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5)
        
        self.start_btn = ttk.Button(camera_frame, text="Start Camera", command=self.start_camera)
        self.start_btn.grid(row=0, column=0, padx=5)
        
        self.stop_btn = ttk.Button(camera_frame, text="Stop Camera", command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=5)
        
        # Camera source input
        ttk.Label(camera_frame, text="Source:").grid(row=0, column=2, padx=5)
        self.camera_source_var = tk.StringVar(value="0")
        camera_source_entry = ttk.Entry(camera_frame, textvariable=self.camera_source_var, width=15)
        camera_source_entry.grid(row=0, column=3, padx=5)
        ttk.Label(camera_frame, text="(0=webcam, 1=USB, or IP URL)").grid(row=0, column=4, padx=5)
        
        # Video display
        video_frame = ttk.Frame(left_panel)
        video_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        left_panel.rowconfigure(1, weight=1)
        left_panel.columnconfigure(0, weight=1)
        
        self.video_label = tk.Label(video_frame, text="Camera not started", bg="black", fg="white", font=("Arial", 16))
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events for line drawing
        self.video_label.bind("<Button-1>", self.on_mouse_click)
        self.video_label.bind("<B1-Motion>", self.on_mouse_drag)
        self.video_label.bind("<ButtonRelease-1>", self.on_mouse_release)
        
        # Bottom control panel
        control_bottom = ttk.Frame(left_panel)
        control_bottom.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Detection settings
        settings_frame = ttk.LabelFrame(control_bottom, text="Detection Settings", padding="5")
        settings_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5)
        
        # Confidence slider
        ttk.Label(settings_frame, text="Confidence:").grid(row=0, column=0, padx=5)
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = ttk.Scale(settings_frame, from_=0.1, to=0.95, variable=self.confidence_var, 
                                     orient=tk.HORIZONTAL, length=200, command=self.on_confidence_change)
        confidence_scale.grid(row=0, column=1, padx=5)
        self.confidence_label = ttk.Label(settings_frame, text="0.50")
        self.confidence_label.grid(row=0, column=2, padx=5)
        
        # NMS IoU slider
        ttk.Label(settings_frame, text="NMS IoU:").grid(row=1, column=0, padx=5)
        self.iou_var = tk.DoubleVar(value=0.45)
        iou_scale = ttk.Scale(settings_frame, from_=0.1, to=0.9, variable=self.iou_var,
                             orient=tk.HORIZONTAL, length=200, command=self.on_iou_change)
        iou_scale.grid(row=1, column=1, padx=5)
        self.iou_label = ttk.Label(settings_frame, text="0.45")
        self.iou_label.grid(row=1, column=2, padx=5)
        
        # Line counting controls
        line_frame = ttk.LabelFrame(control_bottom, text="Line Counting", padding="5")
        line_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        self.line_mode_var = tk.BooleanVar(value=False)
        line_toggle = ttk.Checkbutton(line_frame, text="Enable Line Counting", 
                                      variable=self.line_mode_var, command=self.toggle_line_mode)
        line_toggle.grid(row=0, column=0, padx=5)
        
        self.reset_line_btn = ttk.Button(line_frame, text="Reset Line", command=self.reset_line, state=tk.DISABLED)
        self.reset_line_btn.grid(row=0, column=1, padx=5)
        
        # Direction selector
        ttk.Label(line_frame, text="Direction:").grid(row=1, column=0, padx=5)
        self.direction_var = tk.StringVar(value="both")
        direction_combo = ttk.Combobox(line_frame, textvariable=self.direction_var, 
                                      values=["both", "up→down", "down→up"], state="readonly", width=10)
        direction_combo.grid(row=1, column=1, padx=5)
        direction_combo.bind("<<ComboboxSelected>>", self.on_direction_change)
        
        # Status label
        self.status_label = ttk.Label(control_bottom, text="Ready - Press 'Start Camera'", 
                                     font=("Arial", 10, "bold"))
        self.status_label.grid(row=0, column=2, padx=10)
        
        # Right panel: Dashboard
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        main_frame.columnconfigure(1, weight=1)
        right_panel.configure(width=350)
        
        # Dashboard title
        dashboard_title = ttk.Label(right_panel, text="Live Dashboard", font=("Arial", 16, "bold"))
        dashboard_title.pack(pady=10)
        
        # Total count
        total_frame = ttk.Frame(right_panel)
        total_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(total_frame, text="Total Objects:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        self.total_label = ttk.Label(total_frame, text="0", font=("Arial", 14, "bold"), foreground="blue")
        self.total_label.pack(side=tk.LEFT, padx=10)
        
        # FPS display
        fps_frame = ttk.Frame(right_panel)
        fps_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(fps_frame, text="FPS:", font=("Arial", 10)).pack(side=tk.LEFT)
        self.fps_label = ttk.Label(fps_frame, text="0.0", font=("Arial", 10, "bold"))
        self.fps_label.pack(side=tk.LEFT, padx=10)
        
        # Counts by class
        counts_frame = ttk.LabelFrame(right_panel, text="Counts by Category", padding="10")
        counts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Scrollable text widget for counts
        counts_scroll = tk.Scrollbar(counts_frame)
        counts_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.counts_text = tk.Text(counts_frame, height=20, yscrollcommand=counts_scroll.set, 
                                   font=("Arial", 10), wrap=tk.WORD)
        self.counts_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        counts_scroll.config(command=self.counts_text.yview)
        
        # Instructions
        instructions = ttk.LabelFrame(right_panel, text="Instructions", padding="10")
        instructions.pack(fill=tk.X, padx=10, pady=5)
        inst_text = """Keyboard Shortcuts:
• L - Toggle line drawing mode
• Q - Quit application

Line Drawing:
• Press L to enable
• Click and drag, or
• Click twice (start, end)"""
        ttk.Label(instructions, text=inst_text, font=("Arial", 9), justify=tk.LEFT).pack()
    
    def _bind_events(self):
        """Bind keyboard events."""
        self.root.bind('<KeyPress-l>', lambda e: self.toggle_draw_line_mode())
        self.root.bind('<KeyPress-L>', lambda e: self.toggle_draw_line_mode())
        self.root.bind('<KeyPress-q>', lambda e: self.on_closing())
        self.root.bind('<KeyPress-Q>', lambda e: self.on_closing())
        self.root.focus_set()
    
    def on_mouse_click(self, event):
        """Handle mouse click for line drawing."""
        if not self.draw_line_mode:
            return
        
        if not self.drawing_line:
            # First click - start drawing
            self.line_start_point = (event.x, event.y)
            self.drawing_line = True
            self.line_points = [(event.x, event.y)]
        else:
            # Second click - finish line
            self.line_end_point = (event.x, event.y)
            self.finish_line()
    
    def on_mouse_drag(self, event):
        """Handle mouse drag for line drawing."""
        if not self.draw_line_mode or not self.drawing_line:
            return
        
        # Update end point while dragging
        self.line_end_point = (event.x, event.y)
    
    def on_mouse_release(self, event):
        """Handle mouse release for line drawing."""
        if not self.draw_line_mode or not self.drawing_line:
            return
        
        # Finish line on release (drag mode)
        if self.line_start_point and self.line_end_point:
            self.finish_line()
    
    def finish_line(self):
        """Finish drawing line and set it in detector."""
        if self.line_start_point and self.line_end_point and self.current_frame is not None:
            frame_width = self.current_frame.shape[1]
            frame_height = self.current_frame.shape[0]
            
            # Get label size
            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()
            
            if label_width > 1 and label_height > 1 and self.displayed_image_size:
                # Calculate scale and offset
                disp_w, disp_h = self.displayed_image_size
                offset_x, offset_y = self.display_offset if self.display_offset else (0, 0)
                
                # Adjust click coordinates by offset
                click_x1 = self.line_start_point[0] - offset_x
                click_y1 = self.line_start_point[1] - offset_y
                click_x2 = self.line_end_point[0] - offset_x
                click_y2 = self.line_end_point[1] - offset_y
                
                # Scale to frame coordinates
                scale_x = frame_width / disp_w
                scale_y = frame_height / disp_h
                
                start_x = max(0, min(frame_width - 1, int(click_x1 * scale_x)))
                start_y = max(0, min(frame_height - 1, int(click_y1 * scale_y)))
                end_x = max(0, min(frame_width - 1, int(click_x2 * scale_x)))
                end_y = max(0, min(frame_height - 1, int(click_y2 * scale_y)))
                
                self.detector.set_line((start_x, start_y), (end_x, end_y))
                self.reset_line_btn.config(state=tk.NORMAL)
                self.status_label.config(text="Line set! Counting enabled.")
            elif self.current_frame is not None:
                # Fallback: simple scaling if display info not available
                frame_width = self.current_frame.shape[1]
                frame_height = self.current_frame.shape[0]
                label_width = max(1, self.video_label.winfo_width())
                label_height = max(1, self.video_label.winfo_height())
                
                scale_x = frame_width / label_width
                scale_y = frame_height / label_height
                
                start_x = int(self.line_start_point[0] * scale_x)
                start_y = int(self.line_start_point[1] * scale_y)
                end_x = int(self.line_end_point[0] * scale_x)
                end_y = int(self.line_end_point[1] * scale_y)
                
                self.detector.set_line((start_x, start_y), (end_x, end_y))
                self.reset_line_btn.config(state=tk.NORMAL)
                self.status_label.config(text="Line set! Counting enabled.")
            
            self.drawing_line = False
            self.line_start_point = None
            self.line_end_point = None
    
    def toggle_draw_line_mode(self):
        """Toggle line drawing mode (L key)."""
        self.draw_line_mode = not self.draw_line_mode
        if self.draw_line_mode:
            self.status_label.config(text="Line Drawing Mode: Click and drag, or click twice")
            self.video_label.config(cursor="crosshair")
        else:
            self.status_label.config(text="Line Drawing Mode OFF")
            self.video_label.config(cursor="")
            self.drawing_line = False
            self.line_start_point = None
            self.line_end_point = None
    
    def toggle_line_mode(self):
        """Toggle line counting mode."""
        enabled = self.detector.toggle_line_crossing()
        if enabled:
            self.status_label.config(text="Line counting enabled")
        else:
            self.status_label.config(text="Line counting disabled")
    
    def reset_line(self):
        """Reset counting line."""
        self.detector.reset_line()
        self.reset_line_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Line reset")
    
    def on_direction_change(self, event=None):
        """Handle direction change."""
        direction = self.direction_var.get()
        if direction == "up→down":
            self.detector.set_crossing_direction('up_down')
        elif direction == "down→up":
            self.detector.set_crossing_direction('down_up')
        else:
            self.detector.set_crossing_direction('both')
    
    def on_confidence_change(self, value):
        """Handle confidence threshold change."""
        conf = float(value)
        self.confidence_label.config(text=f"{conf:.2f}")
        self.detector.set_confidence(conf)
    
    def on_iou_change(self, value):
        """Handle IoU threshold change."""
        iou = float(value)
        self.iou_label.config(text=f"{iou:.2f}")
        self.detector.set_iou(iou)
    
    def start_camera(self):
        """Start camera stream."""
        try:
            source = self.camera_source_var.get()
            # Try to parse as int (webcam index) or use as string (IP URL)
            try:
                source = int(source)
            except ValueError:
                pass  # Keep as string for IP camera
            
            self.camera.source = source
            self.camera.target_width = 640
            self.camera.target_height = 360
            
            if self.camera.start():
                self.start_btn.config(state=tk.DISABLED)
                self.stop_btn.config(state=tk.NORMAL)
                self.status_label.config(text="Camera running", foreground="green")
                self.is_processing = True
            else:
                messagebox.showerror("Error", "Failed to start camera")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {e}")
    
    def stop_camera(self):
        """Stop camera stream."""
        self.is_processing = False
        self.camera.stop()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Camera stopped", foreground="red")
        self.video_label.config(image='', text="Camera stopped")
    
    def update_frame(self):
        """Update video frame display (called periodically)."""
        if self.is_processing:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                # Process frame
                result = self.detector.process_frame(frame)
                tracked_objects = result['tracked_objects']
                line_crosses = result['line_crosses']
                counts = result['counts']
                
                # Update counts
                for class_name, count in counts.items():
                    self.counts_by_class[class_name] = self.counts_by_class.get(class_name, 0) + count
                self.total_count = sum(self.counts_by_class.values())
                
                # Draw on frame
                frame = self._draw_detections(frame, tracked_objects, line_crosses)
                
                # Update FPS
                self.fps_history.append(time.time())
                if len(self.fps_history) > 1:
                    elapsed = self.fps_history[-1] - self.fps_history[0]
                    if elapsed > 0:
                        self.current_fps = (len(self.fps_history) - 1) / elapsed
                
                # Display frame
                self._display_frame(frame)
                self.current_frame = frame
        
        # Update dashboard
        self._update_dashboard()
        
        # Schedule next update
        self.root.after(33, self.update_frame)  # ~30 FPS
    
    def _draw_detections(self, frame, tracked_objects, line_crosses):
        """Draw bounding boxes, labels, and line on frame."""
        # Draw counting line
        line_start, line_end = self.detector.get_line_coords()
        if line_start and line_end:
            cv2.line(frame, line_start, line_end, (0, 255, 255), 3)
            # Draw line label
            mid_x = (line_start[0] + line_end[0]) // 2
            mid_y = (line_start[1] + line_end[1]) // 2
            cv2.putText(frame, "COUNTING LINE", (mid_x - 80, mid_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw line crossing markers
        for track_id, class_name, intersection in line_crosses:
            cv2.circle(frame, intersection, 8, (0, 255, 0), -1)
            cv2.putText(frame, f"{class_name}", (intersection[0] + 10, intersection[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw bounding boxes and labels
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj['bbox']
            track_id = obj['track_id']
            class_name = obj['class_name']
            confidence = obj['confidence']
            center = obj['center']
            
            # Color based on class
            color = (0, 255, 0)  # Green default
            if class_name == 'person':
                color = (255, 0, 0)  # Blue
            elif class_name == 'car':
                color = (0, 255, 0)  # Green
            elif class_name == 'dog' or class_name == 'cat':
                color = (0, 0, 255)  # Red
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"ID:{track_id} {class_name} {confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(frame, (x1, y1 - text_height - 5),
                         (x1 + text_width, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw center point
            cv2.circle(frame, center, 4, color, -1)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw mode indicator
        mode = "Line-Crossing Mode" if self.detector.line_crossing_enabled else "Free Count Mode"
        cv2.putText(frame, mode, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame
    
    def _display_frame(self, frame):
        """Display frame in GUI."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to fit label (maintain aspect ratio)
        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()
        
        if label_width > 1 and label_height > 1:
            # Calculate scaling
            frame_h, frame_w = frame_rgb.shape[:2]
            scale = min(label_width / frame_w, label_height / frame_h)
            new_w = int(frame_w * scale)
            new_h = int(frame_h * scale)
            
            frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
            
            # Store displayed image size and offset for coordinate mapping
            self.displayed_image_size = (new_w, new_h)
            offset_x = (label_width - new_w) // 2
            offset_y = (label_height - new_h) // 2
            self.display_offset = (offset_x, offset_y)
        else:
            frame_resized = frame_rgb
            self.displayed_image_size = (frame_rgb.shape[1], frame_rgb.shape[0])
            self.display_offset = (0, 0)
        
        # Convert to PhotoImage
        image = Image.fromarray(frame_resized)
        photo = ImageTk.PhotoImage(image=image)
        
        # Update label
        self.video_label.config(image=photo, text="")
        self.video_label.image = photo  # Keep a reference
    
    def _update_dashboard(self):
        """Update dashboard statistics."""
        # Update total count
        self.total_label.config(text=str(self.total_count))
        
        # Update FPS
        self.fps_label.config(text=f"{self.current_fps:.1f}")
        
        # Update counts by class
        self.counts_text.delete(1.0, tk.END)
        if self.counts_by_class:
            for class_name, count in sorted(self.counts_by_class.items(), key=lambda x: x[1], reverse=True):
                self.counts_text.insert(tk.END, f"{class_name.capitalize()}: {count}\n")
        else:
            self.counts_text.insert(tk.END, "No objects detected yet")
    
    def on_closing(self):
        """Handle window close event."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.is_processing = False
            self.camera.stop()
            if self.on_quit_callback:
                self.on_quit_callback()
            self.root.destroy()
    
    def run(self):
        """Start GUI main loop."""
        self.root.mainloop()
