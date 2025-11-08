# Quick Start Guide

## Installation

```bash
cd frontend
npm install
```

## Run Development Server

```bash
npm start
```

Opens at `http://localhost:3000`

## Features

✅ **Modern UI** - TailwindCSS + Framer Motion  
✅ **Side-by-Side Layout** - Video left, Stats right  
✅ **Real-time Updates** - WebSocket streaming  
✅ **FPS Indicator** - Live performance metrics  
✅ **Detection Log** - Timestamped object detections  
✅ **Interactive Controls** - Connect/Stop, Settings, Line Drawing  
✅ **Keyboard Shortcuts** - L (line mode), Q (quit)  

## File Structure

```
src/
├── components/
│   ├── VideoFeed.jsx      # Video display + line drawing
│   ├── ControlPanel.jsx   # Camera controls
│   └── StatsPanel.jsx     # Statistics + log
├── services/
│   └── api.js             # API + WebSocket client
├── App.jsx                # Main component
├── index.js               # Entry point
└── index.css              # TailwindCSS styles
```

## Backend Connection

Make sure backend is running on `http://localhost:5000`

See `SETUP.md` for detailed documentation.

