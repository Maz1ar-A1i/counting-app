# Troubleshooting Guide

## Camera Not Starting

### Common Issues and Solutions

#### 1. Backend Server Not Running
**Symptom:** Frontend shows connection error or API calls fail

**Solution:**
```bash
# Start backend server
python backend/server.py
```

**Check:** Open browser console and look for connection errors. The backend should be running on `http://localhost:5000`

#### 2. Camera Permission Denied
**Symptom:** Backend logs show "Failed to start camera" or "Failed to open camera source"

**Solutions:**
- **Windows:** Check camera privacy settings
  - Settings > Privacy > Camera > Allow apps to access your camera
- **Linux:** Add user to video group
  ```bash
  sudo usermod -a -G video $USER
  # Log out and log back in
  ```
- **macOS:** Check System Preferences > Security & Privacy > Camera

#### 3. Camera Already in Use
**Symptom:** Camera starts but no frames are received

**Solutions:**
- Close other applications using the camera (Zoom, Teams, Skype, etc.)
- Restart the backend server
- Try a different camera index (1, 2, etc.)

#### 4. Wrong Camera Index
**Symptom:** "Failed to open camera source" error

**Solutions:**
- Try camera index 0, 1, or 2
- On Windows, check Device Manager for camera device
- Test with:
  ```python
  import cv2
  for i in range(3):
      cap = cv2.VideoCapture(i)
      if cap.isOpened():
          print(f"Camera {i} works")
          cap.release()
  ```

#### 5. Detector Model Not Found
**Symptom:** "Failed to load detector model" error

**Solutions:**
- Model will auto-download on first run (requires internet)
- Or manually download from: https://github.com/ultralytics/assets/releases
- Place `yolov8n.pt` in project root directory

#### 6. Dependencies Missing
**Symptom:** Import errors or module not found

**Solutions:**
```bash
# Install all dependencies
pip install -r requirements.txt

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 7. Port Already in Use
**Symptom:** "Address already in use" error

**Solutions:**
- Change port in `backend/server.py`:
  ```python
  socketio.run(app, host='0.0.0.0', port=5001, ...)
  ```
- Or kill process using port 5000:
  ```bash
  # Windows
  netstat -ano | findstr :5000
  taskkill /PID <PID> /F
  
  # Linux/Mac
  lsof -ti:5000 | xargs kill -9
  ```

#### 8. WebSocket Connection Failed
**Symptom:** Video not displaying, connection errors in browser console

**Solutions:**
- Check CORS settings in `backend/server.py`
- Verify SocketIO is installed: `pip install flask-socketio`
- Check firewall settings
- Try different browser (Chrome, Firefox, Edge)

#### 9. Frontend Not Building
**Symptom:** React app shows blank page or build errors

**Solutions:**
```bash
cd frontend
npm install
npm start
```

**Check:**
- Node.js version (requires 16+): `node --version`
- Clear cache: `npm cache clean --force`
- Delete `node_modules` and reinstall

#### 10. Low FPS / Laggy Performance
**Symptom:** Video is choppy or slow

**Solutions:**
- Enable GPU (install CUDA PyTorch)
- Reduce frame resolution in `backend/server.py`:
  ```python
  camera = ThreadedVideoCapture(source=source, width=320, height=240)
  ```
- Increase detection interval:
  ```python
  infer_every_n = 3  # Instead of 2
  ```
- Lower JPEG quality:
  ```python
  cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
  ```

## Debugging Steps

### 1. Check Backend Logs
Look for error messages in the terminal where backend is running:
```
[ERROR] Failed to start camera: ...
[ERROR] Detection error: ...
```

### 2. Test Backend API
```bash
# Test status
curl http://localhost:5000/api/status

# Test camera start
curl -X POST http://localhost:5000/api/camera/start \
  -H "Content-Type: application/json" \
  -d '{"source": 0}'
```

Or use the test script:
```bash
python test_backend.py
```

### 3. Check Browser Console
Open browser developer tools (F12) and check:
- Console tab for JavaScript errors
- Network tab for failed API calls
- WebSocket connection status

### 4. Verify Camera Access
Test camera with OpenCV directly:
```python
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("Camera works!")
    else:
        print("Camera opened but no frame")
    cap.release()
else:
    print("Camera failed to open")
```

### 5. Check System Resources
- CPU usage should be reasonable
- Memory should be available
- GPU memory (if using CUDA) should not be full

## Common Error Messages

### "Failed to open camera source: 0"
- Camera is not connected
- Camera is in use by another application
- Wrong camera index
- Camera driver issues

### "Failed to load detector model"
- Model file not found
- Internet connection required for auto-download
- Insufficient disk space
- PyTorch not installed correctly

### "ModuleNotFoundError: No module named 'X'"
- Missing Python dependency
- Virtual environment not activated
- Wrong Python version

### "Connection refused" or "ECONNREFUSED"
- Backend server not running
- Wrong port number
- Firewall blocking connection

### "WebSocket connection failed"
- SocketIO not installed
- CORS issues
- Network problems
- Browser compatibility

## Getting Help

1. Check backend console for detailed error messages
2. Check browser console for frontend errors
3. Verify all dependencies are installed
4. Test camera with simple OpenCV script
5. Check system permissions
6. Review this troubleshooting guide

## Still Having Issues?

Provide the following information:
- Operating system (Windows/Linux/macOS)
- Python version
- Node.js version
- Error messages from backend console
- Error messages from browser console
- Steps to reproduce the issue

