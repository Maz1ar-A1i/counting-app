# React Frontend Setup Guide

## Prerequisites

- Node.js 16+ and npm
- Python backend running on `http://localhost:5000`

## Installation

### 1. Install Dependencies

```bash
cd frontend
npm install
```

This will install:
- React 18
- Framer Motion (animations)
- Lucide React (icons)
- TailwindCSS (styling)
- Socket.IO Client (WebSocket)
- Axios (HTTP requests)

### 2. Configure TailwindCSS

TailwindCSS is already configured in `tailwind.config.js` and `postcss.config.js`.

## Running the Application

### Development Mode

```bash
npm start
```

The app will open at `http://localhost:3000`

### Production Build

```bash
npm run build
```

The built files will be in the `build/` directory.

## Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── VideoFeed.jsx      # Video display with line drawing
│   │   ├── ControlPanel.jsx   # Camera controls and settings
│   │   └── StatsPanel.jsx     # Statistics and detection log
│   ├── services/
│   │   └── api.js             # API and WebSocket client
│   ├── App.jsx                # Main app component
│   ├── index.js               # React entry point
│   └── index.css              # Global styles + Tailwind
├── package.json
├── tailwind.config.js
└── postcss.config.js
```

## Features

### ✅ Modern UI
- TailwindCSS for styling
- Framer Motion for smooth animations
- Lucide React icons
- Dark mode theme
- Responsive design

### ✅ Real-time Updates
- WebSocket connection for live video frames
- REST API for control actions
- Automatic reconnection on disconnect

### ✅ Interactive Controls
- Connect/Stop camera buttons
- Confidence and IoU sliders
- Line counting toggle
- Reset line button

### ✅ Live Statistics
- Total object count
- FPS indicator
- Counts by category with icons
- Detection log with timestamps

### ✅ Keyboard Shortcuts
- `L` - Toggle line counting mode
- `Q` - Stop camera

## API Integration

The frontend connects to the backend via:

### REST API Endpoints
- `POST /api/camera/start` - Start camera
- `POST /api/camera/stop` - Stop camera
- `POST /api/detector/settings` - Update settings
- `POST /api/line/set` - Set counting line
- `POST /api/line/reset` - Reset line
- `POST /api/line/toggle` - Toggle line mode
- `GET /api/status` - Get system status
- `GET /api/stats` - Get statistics

### WebSocket Events
- `video_frame` - Real-time video frame updates
- `stats_update` - Statistics updates
- `connect` - Connection established
- `disconnect` - Connection lost

## Configuration

### Change API URL

Edit `src/services/api.js`:

```javascript
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
```

Or set environment variable:

```bash
REACT_APP_API_URL=http://your-backend-url:5000 npm start
```

## Troubleshooting

### Port Already in Use

Change port in `package.json`:

```json
"scripts": {
  "start": "PORT=3001 react-scripts start"
}
```

### WebSocket Connection Failed

1. Check backend is running
2. Verify CORS settings in backend
3. Check firewall settings
4. Try different browser

### Build Errors

```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

### TailwindCSS Not Working

Make sure `index.css` imports Tailwind:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

## Performance Tips

1. **Reduce Frame Rate**: Backend sends frames every 2 frames by default
2. **Lower Resolution**: Backend uses 640x360 by default
3. **Disable Animations**: Remove Framer Motion if needed
4. **Optimize Images**: Backend uses JPEG quality 85

## Browser Support

- Chrome/Edge (recommended)
- Firefox
- Safari
- Opera

## Development Tips

1. Use React DevTools for debugging
2. Check browser console for WebSocket logs
3. Monitor Network tab for API calls
4. Use Redux DevTools if adding state management

## Next Steps

1. Add authentication if needed
2. Implement data persistence
3. Add export functionality
4. Create mobile app version
5. Add more visualization options

