# React Frontend Setup Guide

This guide will help you set up and run the React-based frontend for the Object Detection & Counting System.

## Project Structure

```
Counting/
├── backend/
│   ├── server.py          # Flask backend API server
│   └── __init__.py
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── services/      # API services
│   │   └── App.js         # Main app component
│   └── package.json
├── modules/               # Python detection modules
└── requirements.txt       # Python dependencies
```

## Prerequisites

1. **Python 3.8+** with virtual environment
2. **Node.js 16+** and npm (for React frontend)
3. **PyTorch** (CPU or CUDA)
4. **Webcam** or IP camera

## Installation

### Step 1: Install Python Dependencies

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/Mac

# Install Python dependencies
pip install -r requirements.txt

# Install PyTorch (choose one)
# CPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# GPU (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Install Node.js Dependencies

```bash
# Navigate to frontend directory
cd frontend

# Install npm dependencies
npm install
```

## Running the Application

### Option 1: Run Backend and Frontend Separately (Development)

**Terminal 1 - Start Backend:**
```bash
# From project root
python backend/server.py
```

The backend will start on `http://localhost:5000`

**Terminal 2 - Start Frontend:**
```bash
# From frontend directory
cd frontend
npm start
```

The frontend will start on `http://localhost:3000` and automatically open in your browser.

### Option 2: Production Build

**Build React App:**
```bash
cd frontend
npm run build
```

**Run Backend (serves React build):**
```bash
# From project root
python backend/server.py
```

The application will be available at `http://localhost:5000`

## Features

### Frontend Features

1. **Live Video Stream**
   - Real-time video display with WebSocket streaming
   - High-quality JPEG compression for smooth streaming
   - Responsive canvas display

2. **Interactive Controls**
   - Start/Stop camera buttons
   - Camera source input (webcam index or IP URL)
   - Confidence and NMS IoU sliders
   - Line counting toggle

3. **Line Drawing**
   - Click and drag to draw counting line
   - Two-click mode (click start, click end)
   - Visual feedback during drawing
   - Reset line button

4. **Live Dashboard**
   - Real-time object counts by category
   - Total count display
   - FPS counter
   - Sorted by count (highest first)

5. **Responsive Design**
   - Modern dark theme
   - Mobile-friendly layout
   - Smooth animations

### Backend API Endpoints

- `GET /api/status` - Get system status
- `POST /api/camera/start` - Start camera stream
- `POST /api/camera/stop` - Stop camera stream
- `POST /api/detector/settings` - Update detection settings
- `POST /api/line/set` - Set counting line coordinates
- `POST /api/line/reset` - Reset counting line
- `POST /api/line/toggle` - Toggle line counting mode
- `GET /api/stats` - Get current statistics

### WebSocket Events

- `video_frame` - Real-time video frame updates
- `stats_update` - Statistics updates
- `connect` - Client connection
- `disconnect` - Client disconnection

## Usage

1. **Start the Backend**
   ```bash
   python backend/server.py
   ```

2. **Start the Frontend** (development mode)
   ```bash
   cd frontend
   npm start
   ```

3. **Open Browser**
   - Navigate to `http://localhost:3000` (development)
   - Or `http://localhost:5000` (production)

4. **Use the Application**
   - Enter camera source (0 for webcam, or IP camera URL)
   - Click "Start Camera"
   - Adjust confidence and IoU sliders as needed
   - Enable "Line Counting" to draw counting line
   - Click and drag on video to draw line
   - Objects crossing the line will be counted

## Configuration

### Backend Configuration

Edit `backend/server.py` to modify:
- Server port (default: 5000)
- Frame resolution (default: 640x360)
- Detection interval (default: every 2 frames)
- JPEG quality (default: 85)

### Frontend Configuration

Edit `frontend/src/services/api.js` to modify:
- API base URL (default: http://localhost:5000)
- WebSocket connection settings

## Troubleshooting

### Backend Issues

**Camera not starting:**
- Check camera permissions
- Try different camera index (0, 1, 2...)
- Verify camera is not used by another application

**Low FPS:**
- Enable GPU if available
- Reduce frame resolution in `backend/server.py`
- Increase detection interval (infer_every_n)

**WebSocket connection failed:**
- Check firewall settings
- Verify port 5000 is not in use
- Check CORS settings in `backend/server.py`

### Frontend Issues

**npm install fails:**
- Clear npm cache: `npm cache clean --force`
- Delete `node_modules` and `package-lock.json`, then reinstall
- Check Node.js version (requires 16+)

**React app won't start:**
- Check if port 3000 is available
- Verify all dependencies are installed
- Check browser console for errors

**Video not displaying:**
- Check browser console for WebSocket errors
- Verify backend is running
- Check network tab for API calls

### Performance Optimization

1. **Reduce Frame Resolution**
   - Edit `backend/server.py`: `width=640, height=360` → `width=320, height=240`

2. **Increase Detection Interval**
   - Edit `backend/server.py`: `infer_every_n = 2` → `infer_every_n = 3`

3. **Lower JPEG Quality**
   - Edit `backend/server.py`: `cv2.IMWRITE_JPEG_QUALITY, 85` → `70`

4. **Use Smaller Model**
   - Already using `yolov8n.pt` (nano) - fastest option

5. **Enable GPU**
   - Install CUDA-enabled PyTorch
   - GPU will be automatically detected and used

## Development

### Frontend Development

```bash
cd frontend
npm start
```

- Hot reload enabled
- Opens on http://localhost:3000
- Changes automatically reflect

### Backend Development

```bash
python backend/server.py
```

- Debug mode can be enabled in `server.py`
- Check console for logs
- API endpoints can be tested with Postman or curl

## Production Deployment

1. **Build React App**
   ```bash
   cd frontend
   npm run build
   ```

2. **Deploy Backend**
   - Use production WSGI server (gunicorn, uWSGI)
   - Configure reverse proxy (nginx)
   - Set up SSL certificates

3. **Environment Variables**
   - Set `REACT_APP_API_URL` for production API URL
   - Configure CORS for production domain

## API Examples

### Start Camera
```bash
curl -X POST http://localhost:5000/api/camera/start \
  -H "Content-Type: application/json" \
  -d '{"source": 0}'
```

### Update Settings
```bash
curl -X POST http://localhost:5000/api/detector/settings \
  -H "Content-Type: application/json" \
  -d '{"confidence": 0.6, "iou": 0.5}'
```

### Set Line
```bash
curl -X POST http://localhost:5000/api/line/set \
  -H "Content-Type: application/json" \
  -d '{"start": [100, 200], "end": [500, 200]}'
```

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review browser console and backend logs
3. Verify all dependencies are installed
4. Check camera and system permissions

## License

This project is open source. YOLO models are provided by Ultralytics under AGPL-3.0 license.

