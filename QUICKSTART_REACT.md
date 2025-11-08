# Quick Start - React Frontend

## Installation (5 minutes)

### 1. Install Python Dependencies
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. Install Node.js Dependencies
```bash
cd frontend
npm install
```

## Running the Application

### Development Mode (Recommended)

**Terminal 1 - Backend:**
```bash
python backend/server.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

Open browser to `http://localhost:3000`

### Production Mode

**Build React App:**
```bash
cd frontend
npm run build
```

**Run Backend (serves React build):**
```bash
python backend/server.py
```

Open browser to `http://localhost:5000`

## Usage

1. Enter camera source (0 for webcam, or IP camera URL)
2. Click **"Start Camera"**
3. Adjust confidence and IoU sliders
4. Enable **"Line Counting"** or press **L**
5. Click and drag on video to draw counting line
6. Objects crossing the line will be counted
7. Press **Q** to stop camera

## Keyboard Shortcuts

- **L** - Toggle line counting mode
- **Q** - Stop camera (when running)

## Features

✅ Real-time video streaming via WebSocket  
✅ Interactive line drawing  
✅ Live object counts dashboard  
✅ Adjustable detection settings  
✅ Modern React UI with dark theme  
✅ Responsive design  

## Troubleshooting

**Backend won't start:**
- Check Python dependencies: `pip install -r requirements.txt`
- Verify PyTorch is installed
- Check port 5000 is available

**Frontend won't start:**
- Check Node.js version (requires 16+): `node --version`
- Clear cache: `npm cache clean --force`
- Delete `node_modules` and reinstall: `npm install`

**No video:**
- Check backend is running
- Verify camera permissions
- Check browser console for errors

**Low FPS:**
- Enable GPU (install CUDA PyTorch)
- Reduce frame resolution in `backend/server.py`
- Increase detection interval in `backend/server.py`

For detailed information, see `README_REACT.md`

