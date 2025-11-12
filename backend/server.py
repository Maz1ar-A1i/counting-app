# backend/server.py
"""
Flask backend server for object detection and counting system.
Provides REST API and WebSocket for real-time video streaming.
"""

import cv2
import base64
import json
import threading
import time
import traceback
from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import sys
import os
import platform
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.threaded_cam import ThreadedVideoCapture
from modules.detector import ObjectDetector
from modules.area_classifier import AreaClassifier
import torch
import psutil

app = Flask(__name__, static_folder='../frontend/build', static_url_path='')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
camera = None
detector = None
is_processing = False
processing_thread = None
last_processed_frame = None
last_tracked_objects = []
last_line_crosses = []

# Statistics - accumulated counts
current_stats = {
    'fps': 0.0,
    'counts': {},
    'total_count': 0,
    'tracked_objects': []
}
accumulated_counts = {}  # Track cumulative counts

# Area-wise classification state (separate from model-based flow)
area_camera = None
area_classifier = None
is_area_processing = False
area_processing_thread = None
last_area_frame = None
area_stats = {
    'fps': 0.0,
    'counts': {},
    'total_regions': 0
}


def init_detector(force_reload=False):
    """
    Initialize detector with auto GPU detection and optimization.
    
    Args:
        force_reload: If True, force reload even if detector exists
    """
    global detector
    
    try:
        # Check GPU availability
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device == 'cuda':
            try:
                # Enable cuDNN optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                
                # Get GPU info
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                cuda_version = torch.version.cuda
                
                print("=" * 60)
                print("[GPU] CUDA Available - GPU Acceleration Enabled")
                print(f"[GPU] Device: {gpu_name}")
                print(f"[GPU] Memory: {gpu_memory:.2f} GB")
                print(f"[GPU] CUDA Version: {cuda_version}")
                print("=" * 60)
            except Exception as e:
                print(f"[WARNING] CUDA setup error: {e}")
                print("[WARNING] Falling back to CPU")
                device = 'cpu'
        else:
            print("=" * 60)
            print("[CPU] CUDA not available - Using CPU")
            print("[CPU] For GPU acceleration, install PyTorch with CUDA support")
            print("=" * 60)
        
        # Initialize detector with enhanced settings for complex scenes
        # Use yolov8m.pt or yolov8l.pt for better accuracy (downloads automatically if not present)
        # For best accuracy on complex scenes, use: model_path='yolov8m.pt' or 'yolov8l.pt'
        if detector is None or force_reload:
            detector = ObjectDetector(
                model_path='yolov8m.pt',  # Medium model for better accuracy (use 'yolov8n.pt' for speed)
                confidence=0.25,  # Lower threshold for better detection in complex scenes
                iou=0.45,  # NMS IoU threshold
                device=device,
                img_size=640,  # Can increase to 1280 for higher accuracy (slower)
                multi_scale=False  # Set to True for maximum accuracy (slower but better for complex scenes)
            )
        
        if not detector.load_model(force_reload=force_reload):
            print("[ERROR] Failed to load detector model")
            return False
        
        print("[OK] Detector initialized successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Detector initialization failed: {e}")
        traceback.print_exc()
        return False


def process_frames():
    """Process video frames in background thread."""
    global is_processing, camera, detector, current_stats, last_processed_frame
    global last_tracked_objects, last_line_crosses, accumulated_counts
    
    frame_skip = 0
    # Adapt inference frequency to device
    infer_every_n = 2
    try:
        if detector is not None and getattr(detector, 'device', 'cpu') != 'cuda':
            infer_every_n = 3
    except Exception:
        pass

    # Throttle video emit rate to reduce network/CPU load
    target_fps = 20.0
    min_emit_interval = 1.0 / target_fps
    last_emit_time = 0.0
    
    print("[INFO] Frame processing thread started")
    
    try:
        while is_processing and camera is not None:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            
            frame_skip += 1
            
            # Run detection every N frames for better performance
            if frame_skip % infer_every_n == 0 and detector is not None:
                try:
                    result = detector.process_frame(frame)
                    tracked_objects = result['tracked_objects']
                    line_crosses = result['line_crosses']
                    new_counts = result['counts']
                    
                    # Accumulate counts correctly (sum per-frame increments)
                    for class_name, count in new_counts.items():
                        accumulated_counts[class_name] = accumulated_counts.get(class_name, 0) + count
                    
                    # Update statistics with accumulated counts
                    current_stats['tracked_objects'] = tracked_objects
                    current_stats['counts'] = accumulated_counts.copy()
                    current_stats['total_count'] = sum(accumulated_counts.values())
                    current_stats['fps'] = min(camera.get_fps(), 60.0)  # Cap FPS at 60
                    
                    # Cache results for skipped frames
                    last_tracked_objects = tracked_objects
                    last_line_crosses = line_crosses
                except Exception as e:
                    print(f"[ERROR] Detection error: {e}")
                    traceback.print_exc()
                    # Use cached results on error
                    tracked_objects = last_tracked_objects if last_tracked_objects else []
                    line_crosses = last_line_crosses if last_line_crosses else []
            else:
                # Use cached results for skipped frames
                tracked_objects = last_tracked_objects if last_tracked_objects else []
                line_crosses = last_line_crosses if last_line_crosses else []
            
            # Draw detections on frame
            try:
                frame = draw_detections(frame, tracked_objects, line_crosses)
                last_processed_frame = frame.copy()
            except Exception as e:
                print(f"[ERROR] Drawing error: {e}")
                if last_processed_frame is not None:
                    frame = last_processed_frame.copy()
            
            # Encode frame to JPEG
            try:
                # Throttle emits to target FPS
                now = time.time()
                if (now - last_emit_time) >= min_emit_interval:
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')
                        
                        # Send frame via WebSocket
                        socketio.emit('video_frame', {
                            'frame': frame_b64,
                            'stats': current_stats
                        })
                        last_emit_time = now
            except Exception as e:
                print(f"[ERROR] Frame encoding error: {e}")
            
            # Send statistics update every 10 frames
            if frame_skip % 10 == 0:
                try:
                    socketio.emit('stats_update', current_stats)
                except Exception as e:
                    print(f"[ERROR] Stats update error: {e}")
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.01)
            
            # Periodic GPU memory cleanup (every 100 frames)
            if frame_skip % 100 == 0 and detector is not None and detector.device == 'cuda':
                try:
                    import torch
                    torch.cuda.empty_cache()
                except:
                    pass
    
    except Exception as e:
        print(f"[ERROR] Processing thread error: {e}")
        traceback.print_exc()
    finally:
        # Cleanup GPU memory on exit
        if detector is not None and detector.device == 'cuda':
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass
        print("[INFO] Frame processing thread stopped")
        is_processing = False


def draw_detections(frame, tracked_objects, line_crosses):
    """Draw bounding boxes, labels, and line on frame."""
    if detector is None:
        return frame
    
    try:
        # Draw counting line
        line_start, line_end = detector.get_line_coords()
        if line_start and line_end:
            cv2.line(frame, line_start, line_end, (0, 255, 255), 3)
            mid_x = (line_start[0] + line_end[0]) // 2
            mid_y = (line_start[1] + line_end[1]) // 2
            cv2.putText(frame, "COUNTING LINE", (mid_x - 80, mid_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw line crossing markers
        for track_id, class_name, intersection in line_crosses:
            cv2.circle(frame, intersection, 8, (0, 255, 0), -1)
            cv2.putText(frame, f"{class_name}", (intersection[0] + 10, intersection[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw bounding boxes
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj['bbox']
            track_id = obj['track_id']
            class_name = obj['class_name']
            confidence = obj['confidence']
            center = obj['center']
            
            # Color based on class
            color = (0, 255, 0)
            if class_name == 'person':
                color = (255, 0, 0)
            elif class_name == 'car':
                color = (0, 255, 0)
            elif class_name in ['dog', 'cat']:
                color = (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"ID:{track_id} {class_name} {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(frame, (x1, y1 - text_height - 5),
                         (x1 + text_width, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw center point
            cv2.circle(frame, center, 4, color, -1)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {current_stats['fps']:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw mode
        mode = "Line-Crossing Mode" if detector.line_crossing_enabled else "Free Count Mode"
        cv2.putText(frame, mode, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    except Exception as e:
        print(f"[ERROR] Drawing error: {e}")
    
    return frame


# API Routes
@app.route('/')
def index():
    """Serve React app."""
    try:
        return send_from_directory(app.static_folder, 'index.html')
    except:
        return jsonify({'message': 'React app not built. Run: cd frontend && npm run build'}), 404


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status."""
    return jsonify({
        'camera_running': is_processing,
        'camera_available': camera is not None,
        'detector_loaded': detector is not None,
        'device': detector.device if detector else 'unknown'
    })


@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """Start camera stream."""
    global camera, detector, is_processing, processing_thread
    
    print("\n[API] Camera start request received")
    
    if is_processing:
        print("[WARNING] Camera already running")
        return jsonify({'success': False, 'message': 'Camera already running'}), 400
    
    try:
        data = request.json or {}
        source = data.get('source', 0)
        
        # Parse source - handle string "0" or integer 0, or RTSP/IP camera URLs
        try:
            if isinstance(source, str):
                # Check if it's an RTSP or HTTP URL
                if source.startswith('rtsp://') or source.startswith('http://') or source.startswith('https://'):
                    # Keep as string for RTSP/IP camera URLs
                    print(f"[INFO] Detected camera URL: {source[:50]}...")  # Show first 50 chars for security
                    pass
                else:
                    # Try to convert to int, if fails keep as string
                    source = int(source)
        except ValueError:
            # Keep as string for IP camera URLs
            pass
        
        print(f"[INFO] Attempting to start camera with source: {source}")
        
        # Initialize detector if not already done
        if detector is None:
            print("[INFO] Initializing detector...")
            if not init_detector():
                error_msg = 'Failed to load detector model. Check console for details.'
                print(f"[ERROR] {error_msg}")
                return jsonify({'success': False, 'message': error_msg}), 500
        
        # Stop existing camera if any
        if camera is not None:
            print("[INFO] Stopping existing camera...")
            camera.stop()
            camera = None
            time.sleep(0.5)
        
        # Start camera
        print("[INFO] Creating camera instance...")
        try:
            camera = ThreadedVideoCapture(source=source, width=640, height=360)
        except Exception as e:
            error_msg = f'Failed to create camera instance: {str(e)}'
            print(f"[ERROR] {error_msg}")
            traceback.print_exc()
            return jsonify({
                'success': False, 
                'message': error_msg,
                'error_type': type(e).__name__
            }), 500
        
        print("[INFO] Starting camera stream...")
        try:
            if not camera.start():
                # Get more details about why it failed
                if isinstance(source, str) and source.startswith('rtsp://'):
                    error_msg = 'Failed to connect to RTSP stream. Possible reasons:\n' \
                              '- RTSP URL is incorrect or camera is offline\n' \
                              '- Network connectivity issues\n' \
                              '- Camera credentials are wrong\n' \
                              '- Firewall blocking port 554\n' \
                              '- Camera does not support the requested stream format'
                elif isinstance(source, int):
                    error_msg = f'Failed to start camera (index {source}). Possible reasons:\n' \
                              '- Camera is not connected\n' \
                              '- Camera is being used by another application\n' \
                              '- Camera index is incorrect'
                else:
                    error_msg = 'Failed to start camera. Check if camera is accessible.'
                
                print(f"[ERROR] {error_msg}")
                print(f"[ERROR] Camera source: {source if not isinstance(source, str) or not source.startswith('rtsp://') else source[:50] + '...'}")
                camera = None
                return jsonify({
                    'success': False, 
                    'message': error_msg,
                    'source_type': 'rtsp' if isinstance(source, str) and source.startswith('rtsp://') else 'local'
                }), 500
        except Exception as e:
            error_msg = f'Exception during camera start: {str(e)}'
            print(f"[ERROR] {error_msg}")
            traceback.print_exc()
            if camera:
                try:
                    camera.stop()
                except:
                    pass
                camera = None
            return jsonify({
                'success': False, 
                'message': error_msg,
                'error_type': type(e).__name__
            }), 500
        
        print("[OK] Camera started successfully")
        
        # Wait a bit for camera to stabilize
        time.sleep(0.5)
        
        # Reset processing state
        is_processing = True
        last_tracked_objects = []
        last_line_crosses = []
        
        # Start processing thread
        print("[INFO] Starting processing thread...")
        processing_thread = threading.Thread(target=process_frames, daemon=True)
        processing_thread.start()
        
        print("[OK] Processing thread started")
        
        return jsonify({'success': True, 'message': 'Camera started successfully'})
    
    except Exception as e:
        error_msg = f'Error starting camera: {str(e)}'
        error_type = type(e).__name__
        print(f"\n[ERROR] ========================================")
        print(f"[ERROR] Camera start failed!")
        print(f"[ERROR] Error Type: {error_type}")
        print(f"[ERROR] Error Message: {error_msg}")
        print(f"[ERROR] ========================================")
        print(f"\n[ERROR] Full traceback:")
        traceback.print_exc()
        print(f"[ERROR] ========================================\n")
        
        # Provide more specific error messages
        detailed_msg = error_msg
        if 'RTSP' in str(e) or 'rtsp' in str(e).lower():
            detailed_msg = f'RTSP connection failed: {error_msg}. Check if the camera is accessible and the URL is correct.'
        elif 'CUDA' in str(e) or 'cuda' in str(e).lower():
            detailed_msg = f'GPU error: {error_msg}. Trying to continue with CPU...'
        elif 'model' in str(e).lower() or 'yolo' in str(e).lower():
            detailed_msg = f'Model loading error: {error_msg}. Check if YOLO model file exists.'
        
        # Cleanup on error
        if camera:
            try:
                camera.stop()
            except Exception as cleanup_error:
                print(f"[WARNING] Error during camera cleanup: {cleanup_error}")
            camera = None
        is_processing = False
        
        return jsonify({
            'success': False, 
            'message': detailed_msg,
            'error_type': error_type,
            'error_details': str(e)
        }), 500


@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """Stop camera stream."""
    global camera, is_processing, processing_thread
    
    print("\n[API] Camera stop request received")
    
    if not is_processing and camera is None:
        return jsonify({'success': False, 'message': 'Camera not running'}), 400
    
    try:
        is_processing = False
        
        # Wait for processing thread to stop
        if processing_thread:
            print("[INFO] Waiting for processing thread to stop...")
            processing_thread.join(timeout=3.0)
            processing_thread = None
        
        # Stop camera
        if camera:
            print("[INFO] Stopping camera...")
            camera.stop()
            camera = None
            print("[OK] Camera stopped")
        
        # Reset accumulated counts
        accumulated_counts.clear()
        current_stats['counts'] = {}
        current_stats['total_count'] = 0
        
        return jsonify({'success': True, 'message': 'Camera stopped successfully'})
    
    except Exception as e:
        error_msg = f'Error stopping camera: {str(e)}'
        print(f"[ERROR] {error_msg}")
        return jsonify({'success': False, 'message': error_msg}), 500


@app.route('/api/detector/settings', methods=['POST'])
def update_settings():
    """Update detector settings."""
    global detector
    
    if detector is None:
        return jsonify({'success': False, 'message': 'Detector not initialized'}), 400
    
    try:
        data = request.json
        if 'confidence' in data:
            detector.set_confidence(data['confidence'])
            print(f"[INFO] Confidence updated to: {data['confidence']}")
        if 'iou' in data:
            detector.set_iou(data['iou'])
            print(f"[INFO] IoU updated to: {data['iou']}")
        if 'multi_scale' in data:
            detector.multi_scale = bool(data['multi_scale'])
            print(f"[INFO] Multi-scale inference: {'enabled' if detector.multi_scale else 'disabled'}")
        if 'img_size' in data:
            detector.img_size = int(data['img_size'])
            print(f"[INFO] Image size updated to: {detector.img_size}")
        
        return jsonify({'success': True})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/line/set', methods=['POST'])
def set_line():
    """Set counting line coordinates."""
    global detector
    
    if detector is None:
        return jsonify({'success': False, 'message': 'Detector not initialized'}), 400
    
    try:
        data = request.json
        if 'start' not in data or 'end' not in data:
            return jsonify({'success': False, 'message': 'Missing start or end coordinates'}), 400
        
        start = tuple(data['start'])
        end = tuple(data['end'])
        
        # Validate coordinates
        if len(start) != 2 or len(end) != 2:
            return jsonify({'success': False, 'message': 'Invalid coordinates format'}), 400
        
        # Ensure coordinates are within frame bounds
        start = (max(0, min(639, int(start[0]))), max(0, min(359, int(start[1]))))
        end = (max(0, min(639, int(end[0]))), max(0, min(359, int(end[1]))))
        
        detector.set_line(start, end)
        print(f"[INFO] Line set: {start} -> {end}")
        return jsonify({'success': True, 'message': f'Line set from {start} to {end}'})
    
    except Exception as e:
        print(f"[ERROR] Line set error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/line/reset', methods=['POST'])
def reset_line():
    """Reset counting line."""
    global detector
    
    if detector is None:
        return jsonify({'success': False, 'message': 'Detector not initialized'}), 400
    
    try:
        detector.reset_line()
        detector.reset_counts()
        # Reset accumulated counts
        accumulated_counts.clear()
        current_stats['counts'] = {}
        current_stats['total_count'] = 0
        print("[INFO] Line reset")
        return jsonify({'success': True})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/line/toggle', methods=['POST'])
def toggle_line_mode():
    """Toggle line counting mode."""
    global detector
    
    if detector is None:
        return jsonify({'success': False, 'message': 'Detector not initialized'}), 400
    
    try:
        enabled = detector.toggle_line_crossing()
        print(f"[INFO] Line counting mode: {'enabled' if enabled else 'disabled'}")
        return jsonify({'success': True, 'enabled': enabled})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get current statistics."""
    return jsonify(current_stats)


@app.route('/api/detector/reload', methods=['POST'])
def reload_detector():
    """Reload the detector model."""
    global detector
    
    try:
        print("[INFO] Reloading detector model...")
        
        # Release old model if exists
        if detector is not None:
            try:
                # Clear GPU memory if using CUDA
                if detector.device == 'cuda':
                    import torch
                    if detector.model is not None:
                        del detector.model
                    torch.cuda.empty_cache()
                    print("[INFO] Cleared GPU memory")
            except Exception as e:
                print(f"[WARNING] Error clearing old model: {e}")
        
        # Reinitialize detector with force reload
        if not init_detector(force_reload=True):
            return jsonify({
                'success': False,
                'message': 'Failed to reload detector model. Check console for details.'
            }), 500
        
        # Verify model loaded
        if detector is None or detector.model is None:
            return jsonify({
                'success': False,
                'message': 'Model reloaded but not properly initialized.'
            }), 500
        
        print("[OK] Detector model reloaded successfully")
        
        return jsonify({
            'success': True,
            'message': 'Model reloaded successfully',
            'model_path': detector.model_path,
            'device': detector.device,
            'img_size': detector.img_size,
            'confidence': detector.confidence
        })
    
    except Exception as e:
        error_msg = f'Error reloading detector: {str(e)}'
        print(f"[ERROR] {error_msg}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': error_msg
        }), 500


@app.route('/api/detect/image', methods=['POST'])
def detect_image():
    """Detect objects in uploaded image with enhanced settings for better accuracy."""
    global detector
    
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'}), 400
    
    try:
        # Initialize detector if not already done
        if detector is None:
            print("[INFO] Initializing detector for image detection...")
            if not init_detector():
                return jsonify({
                    'success': False,
                    'message': 'Failed to load detector model. Check console for details.'
                }), 500
        
        # Verify model is loaded
        if detector.model is None:
            print("[WARNING] Model not loaded, attempting to reload...")
            if not detector.load_model():
                return jsonify({
                    'success': False,
                    'message': 'Model failed to load. Try reloading the model.'
                }), 500
        
        # Read image
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'success': False, 'message': 'Invalid image file'}), 400
        
        # Get original image dimensions
        orig_height, orig_width = image.shape[:2]
        print(f"[INFO] Processing image: {orig_width}x{orig_height}")
        
        # For static images, use higher resolution and multi-scale for better accuracy
        # Save current settings
        original_img_size = detector.img_size
        original_multi_scale = detector.multi_scale
        original_confidence = detector.confidence
        
        # Use higher resolution for images (can be slower but more accurate)
        # Calculate optimal image size (max 1280, but scale based on original size)
        max_img_size = 1280
        if max(orig_width, orig_height) > max_img_size:
            detector.img_size = max_img_size
        else:
            # Use larger size for better detection, but cap at 1280
            detector.img_size = min(max(orig_width, orig_height), max_img_size)
        
        # Enable multi-scale for static images (better accuracy, acceptable for images)
        detector.multi_scale = True
        
        # Use slightly lower confidence for images to catch more objects
        detector.confidence = max(0.2, original_confidence - 0.05)
        
        print(f"[INFO] Image detection settings: img_size={detector.img_size}, multi_scale={detector.multi_scale}, confidence={detector.confidence}")
        
        # Perform detection with enhanced settings
        detections = detector.detect(image)
        
        # Restore original settings for video processing
        detector.img_size = original_img_size
        detector.multi_scale = original_multi_scale
        detector.confidence = original_confidence
        
        # Count objects by class
        counts = {}
        for det in detections:
            class_name = det['class_name']
            counts[class_name] = counts.get(class_name, 0) + 1
        
        total_count = len(detections)
        
        # Draw bounding boxes on image
        annotated_image = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Color based on class
            color = (0, 255, 0)  # Default green
            if class_name == 'person':
                color = (255, 0, 0)  # Blue
            elif class_name == 'car':
                color = (0, 255, 0)  # Green
            elif class_name in ['dog', 'cat']:
                color = (0, 0, 255)  # Red
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(annotated_image, (x1, y1 - text_height - 5),
                         (x1 + text_width, y1), color, -1)
            cv2.putText(annotated_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw center point
            center = det['center']
            cv2.circle(annotated_image, center, 4, color, -1)
        
        # Encode annotated image to base64
        ret, buffer = cv2.imencode('.jpg', annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ret:
            annotated_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        else:
            annotated_b64 = None
        
        print(f"[INFO] Image detection completed: {total_count} objects detected")
        
        return jsonify({
            'success': True,
            'counts': counts,
            'total_count': total_count,
            'detections': detections,
            'annotated_image': annotated_b64
        })
    
    except Exception as e:
        error_msg = f'Error processing image: {str(e)}'
        print(f"[ERROR] {error_msg}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': error_msg
        }), 500


# WebSocket Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print("[INFO] WebSocket client connected")
    emit('connected', {'message': 'Connected to server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print("[INFO] WebSocket client disconnected")


# ======================
# Area-wise classification (separate feature)
# ======================
def process_area_frames():
    """Process frames for area-wise classification in a separate thread."""
    global is_area_processing, area_camera, area_classifier, area_stats, last_area_frame

    frame_index = 0
    # Throttle emits for area to ~15 FPS to be safe
    target_fps = 15.0
    min_emit_interval = 1.0 / target_fps
    last_emit_time = 0.0

    print("[INFO] Area processing thread started")
    try:
        while is_area_processing and area_camera is not None:
            ret, frame = area_camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_index += 1
            try:
                result = area_classifier.process(frame)
                area_stats['fps'] = result.get('fps', 0.0)
                area_stats['counts'] = result.get('counts', {})
                area_stats['total_regions'] = result.get('total_regions', 0)
                annotated = result.get('frame', frame)
                last_area_frame = annotated.copy()

                # Throttle emit
                now = time.time()
                if (now - last_emit_time) >= min_emit_interval:
                    ret_j, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ret_j:
                        b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                        socketio.emit('area_frame', {
                            'frame': b64,
                            'stats': area_stats
                        })
                        last_emit_time = now

                if frame_index % 10 == 0:
                    socketio.emit('area_stats', area_stats)
            except Exception as e:
                print(f"[ERROR] Area classification error: {e}")
                traceback.print_exc()
                if last_area_frame is not None:
                    try:
                        ret_j, buffer = cv2.imencode('.jpg', last_area_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        if ret_j:
                            b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                            socketio.emit('area_frame', {
                                'frame': b64,
                                'stats': area_stats
                            })
                    except Exception:
                        pass

            time.sleep(0.005)
    except Exception as e:
        print(f"[ERROR] Area processing thread error: {e}")
        traceback.print_exc()
    finally:
        print("[INFO] Area processing thread stopped")
        is_area_processing = False


@app.route('/api/area/start', methods=['POST'])
def area_start():
    """Start area-wise classification with separate camera/context."""
    global area_camera, area_classifier, is_area_processing, area_processing_thread, area_stats

    if is_area_processing:
        return jsonify({'success': False, 'message': 'Area processing already running'}), 400

    try:
        data = request.json or {}
        source = data.get('source', 0)
        min_area = int(data.get('min_area', 500))
        area_bins = data.get('area_bins')  # optional dict

        # Parse source
        try:
            if isinstance(source, str):
                if source.startswith(('rtsp://', 'http://', 'https://')):
                    pass
                else:
                    source = int(source)
        except ValueError:
            pass

        # Stop existing
        if area_camera is not None:
            try:
                area_camera.stop()
            except Exception:
                pass
            area_camera = None
            time.sleep(0.3)

        # Create classifier and camera
        area_classifier = AreaClassifier(min_area=min_area, area_bins=area_bins)
        area_camera = ThreadedVideoCapture(source=source, width=640, height=360)

        if not area_camera.start():
            area_camera = None
            return jsonify({'success': False, 'message': 'Failed to start area camera'}), 500

        # Reset stats
        area_stats = {'fps': 0.0, 'counts': {}, 'total_regions': 0}

        # Start thread
        is_area_processing = True
        area_processing_thread = threading.Thread(target=process_area_frames, daemon=True)
        area_processing_thread.start()

        return jsonify({'success': True, 'message': 'Area classification started'})
    except Exception as e:
        print(f"[ERROR] Area start failed: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/area/stop', methods=['POST'])
def area_stop():
    """Stop area-wise classification."""
    global area_camera, is_area_processing, area_processing_thread

    try:
        is_area_processing = False
        if area_processing_thread:
            area_processing_thread.join(timeout=2.0)
            area_processing_thread = None
        if area_camera:
            area_camera.stop()
            area_camera = None
        return jsonify({'success': True, 'message': 'Area classification stopped'})
    except Exception as e:
        print(f"[ERROR] Area stop failed: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/area/settings', methods=['POST'])
def area_settings():
    """Update area-wise classification settings without affecting models."""
    global area_classifier

    if area_classifier is None:
        return jsonify({'success': False, 'message': 'Area classifier not initialized'}), 400

    try:
        data = request.json or {}
        min_area = data.get('min_area')
        area_bins = data.get('area_bins')
        area_classifier.update_settings(min_area=min_area, area_bins=area_bins)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ======================
# System metrics & recommendations
# ======================
def _detect_disk_kind() -> str:
    """
    Best-effort disk type detection.
    Returns: 'ssd' | 'hdd' | 'unknown'
    """
    try:
        if platform.system() == 'Linux':
            # Check primary device rotational flag
            root = '/'
            stat = os.stat(root)
            major_minor = f"{os.major(stat.st_dev)}:{os.minor(stat.st_dev)}"
            # Find matching block device
            by_dev = '/sys/dev/block'
            for name in os.listdir(by_dev):
                if name == major_minor:
                    dev_link = os.path.realpath(os.path.join(by_dev, name))
                    block = dev_link.split('/')[-1]
                    rot_path = f"/sys/block/{block}/queue/rotational"
                    if os.path.exists(rot_path):
                        with open(rot_path, 'r') as f:
                            val = f.read().strip()
                            return 'hdd' if val == '1' else 'ssd'
            return 'unknown'
        # Windows/macOS fallback
        return 'unknown'
    except Exception:
        return 'unknown'


@app.route('/api/system/metrics', methods=['GET'])
def system_metrics():
    """Return live system metrics: CPU/GPU, RAM, Disk."""
    try:
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.2)
        cpu_freq = psutil.cpu_freq()
        cpu = {
            'percent': cpu_percent,
            'cores_physical': psutil.cpu_count(logical=False),
            'cores_logical': psutil.cpu_count(logical=True),
            'freq_current_mhz': cpu_freq.current if cpu_freq else None,
            'freq_max_mhz': cpu_freq.max if cpu_freq else None,
            'processor': platform.processor(),
            'brand': platform.uname().machine
        }

        # RAM
        vm = psutil.virtual_memory()
        ram = {
            'total_gb': round(vm.total / (1024 ** 3), 2),
            'available_gb': round(vm.available / (1024 ** 3), 2),
            'used_gb': round(vm.used / (1024 ** 3), 2),
            'percent': vm.percent
        }

        # Disk (root partition)
        total, used, free = shutil.disk_usage(os.path.abspath(os.sep))
        disk = {
            'total_gb': round(total / (1024 ** 3), 2),
            'used_gb': round(used / (1024 ** 3), 2),
            'free_gb': round(free / (1024 ** 3), 2),
            'percent': round(used / total * 100, 1),
            'kind': _detect_disk_kind()
        }

        # GPU (Torch CUDA)
        gpu = None
        if torch.cuda.is_available():
            try:
                idx = 0
                props = torch.cuda.get_device_properties(idx)
                mem_total = props.total_memory / (1024 ** 3)
                mem_alloc = torch.cuda.memory_allocated(idx) / (1024 ** 3)
                mem_reserved = torch.cuda.memory_reserved(idx) / (1024 ** 3)
                gpu = {
                    'name': torch.cuda.get_device_name(idx),
                    'total_mem_gb': round(mem_total, 2),
                    'allocated_gb': round(mem_alloc, 2),
                    'reserved_gb': round(mem_reserved, 2),
                    'cuda_version': torch.version.cuda,
                    'sm_count': getattr(props, 'multi_processor_count', None)
                }
            except Exception as e:
                gpu = {'error': str(e)}

        # App-specific status
        streams = {
            'detector_running': is_processing,
            'area_running': is_area_processing
        }

        return jsonify({
            'success': True,
            'cpu': cpu,
            'ram': ram,
            'disk': disk,
            'gpu': gpu,
            'streams': streams
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/system/recommendations', methods=['GET'])
def system_recommendations():
    """Return recommended and minimum resources for smooth operation."""
    # Baseline from typical YOLOv8 workloads; adjust to your scene complexity
    rec = {
        'minimum': {
            'cpu': '4 cores @ 2.5+ GHz',
            'ram': '8 GB',
            'disk': '10 GB free (SSD preferred)',
            'gpu': 'Optional; CPU-only works with yolov8n at lower FPS'
        },
        'recommended': {
            'cpu': '6–8 cores @ 3.0+ GHz',
            'ram': '16 GB',
            'disk': '20+ GB free SSD',
            'gpu': 'NVIDIA GPU (6–8 GB VRAM), e.g., RTX 2060/3060 (CUDA 11.8/12.1)'
        },
        'for_best_results': {
            'cpu': '8–12 cores modern CPU',
            'ram': '32 GB',
            'disk': 'NVMe SSD with ample space for models and recordings',
            'gpu': 'NVIDIA 8–12 GB VRAM (e.g., RTX 3060/3080/4060), latest drivers'
        },
        'other': [
            'Stable network for RTSP/IP cameras (wired Ethernet preferred)',
            'Up-to-date GPU drivers and CUDA/cuDNN (if using GPU)',
            'Run backend and frontend on same machine or low-latency network',
            'Close other heavy apps to avoid CPU/GPU contention'
        ]
    }
    return jsonify({'success': True, 'recommendations': rec})


if __name__ == '__main__':
    print("="*60)
    print("Object Detection & Counting System - Backend Server")
    print("="*60)
    print("\nStarting server on http://localhost:5000")
    print("React frontend will be served from http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down server...")
        if camera:
            camera.stop()
        print("[OK] Server stopped")
