import React, { useState, useEffect, useCallback, useRef } from 'react';
import './App.css';
import VideoDisplay from './components/VideoDisplay';
import ControlPanel from './components/ControlPanel';
import Dashboard from './components/Dashboard';
import { socket, api } from './services/api';

function App() {
  const [cameraRunning, setCameraRunning] = useState(false);
  const [videoFrame, setVideoFrame] = useState(null);
  const [stats, setStats] = useState({
    fps: 0,
    counts: {},
    total_count: 0,
    tracked_objects: []
  });
  const [settings, setSettings] = useState({
    confidence: 0.5,
    iou: 0.45,
    cameraSource: '0'
  });
  const [lineMode, setLineMode] = useState(false);
  const [status, setStatus] = useState('Ready - Press Start Camera');
  
  // Accumulate counts across frames
  const accumulatedCountsRef = useRef({});
  const lastStatsRef = useRef(null);

  const handleStartCamera = async () => {
    try {
      const source = parseInt(settings.cameraSource) || settings.cameraSource;
      const response = await api.post('/api/camera/start', { source });
      if (response.data.success) {
        setCameraRunning(true);
        setStatus('Camera running');
        // Reset accumulated counts when starting
        accumulatedCountsRef.current = {};
      } else {
        setStatus(`Error: ${response.data.message}`);
      }
    } catch (error) {
      setStatus(`Error: ${error.message}`);
    }
  };

  const handleStopCamera = async () => {
    try {
      const response = await api.post('/api/camera/stop');
      if (response.data.success) {
        setCameraRunning(false);
        setStatus('Camera stopped');
        setVideoFrame(null);
        // Reset accumulated counts when stopping
        accumulatedCountsRef.current = {};
        setStats({
          fps: 0,
          counts: {},
          total_count: 0,
          tracked_objects: []
        });
      }
    } catch (error) {
      setStatus(`Error: ${error.message}`);
    }
  };

  const handleUpdateSettings = async (newSettings) => {
    try {
      await api.post('/api/detector/settings', newSettings);
      setSettings({ ...settings, ...newSettings });
    } catch (error) {
      console.error('Failed to update settings:', error);
    }
  };

  const handleSetLine = async (start, end) => {
    try {
      await api.post('/api/line/set', { start, end });
      setStatus('Line set! Counting enabled.');
    } catch (error) {
      setStatus(`Error: ${error.message}`);
    }
  };

  const handleResetLine = async () => {
    try {
      await api.post('/api/line/reset');
      setStatus('Line reset');
      // Reset accumulated counts when resetting line
      accumulatedCountsRef.current = {};
      setStats(prev => ({
        ...prev,
        counts: {},
        total_count: 0
      }));
    } catch (error) {
      setStatus(`Error: ${error.message}`);
    }
  };

  const handleToggleLineMode = useCallback(async () => {
    try {
      const response = await api.post('/api/line/toggle');
      if (response.data.success) {
        setLineMode(response.data.enabled);
        setStatus(response.data.enabled ? 'Line counting enabled' : 'Line counting disabled');
      }
    } catch (error) {
      setStatus(`Error: ${error.message}`);
    }
  }, []);

  useEffect(() => {
    // Socket connection
    socket.on('connect', () => {
      console.log('Connected to server');
      setStatus('Connected to server');
    });

    socket.on('video_frame', (data) => {
      setVideoFrame(data.frame);
      if (data.stats) {
        // Accumulate counts - add new counts to existing ones
        const newCounts = data.stats.counts || {};
        Object.keys(newCounts).forEach(key => {
          if (!accumulatedCountsRef.current[key]) {
            accumulatedCountsRef.current[key] = 0;
          }
          // Only add if this is a new count (compare with last stats)
          if (!lastStatsRef.current || 
              !lastStatsRef.current.counts || 
              (lastStatsRef.current.counts[key] || 0) < newCounts[key]) {
            accumulatedCountsRef.current[key] = newCounts[key];
          }
        });
        
        // Calculate total from accumulated counts
        const total = Object.values(accumulatedCountsRef.current).reduce((a, b) => a + b, 0);
        
        setStats({
          fps: data.stats.fps || 0,
          counts: { ...accumulatedCountsRef.current },
          total_count: total,
          tracked_objects: data.stats.tracked_objects || []
        });
        
        lastStatsRef.current = data.stats;
      }
    });

    socket.on('stats_update', (data) => {
      // Update FPS and tracked objects, but keep accumulated counts
      setStats(prev => ({
        ...prev,
        fps: data.fps || prev.fps,
        tracked_objects: data.tracked_objects || prev.tracked_objects
      }));
    });

    socket.on('disconnect', () => {
      console.log('Disconnected from server');
      setStatus('Disconnected from server');
    });

    return () => {
      socket.off('connect');
      socket.off('video_frame');
      socket.off('stats_update');
      socket.off('disconnect');
    };
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e) => {
      if (e.key === 'l' || e.key === 'L') {
        e.preventDefault();
        handleToggleLineMode();
      } else if (e.key === 'q' || e.key === 'Q') {
        if (cameraRunning) {
          e.preventDefault();
          handleStopCamera();
        }
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => {
      window.removeEventListener('keydown', handleKeyPress);
    };
  }, [lineMode, cameraRunning, handleToggleLineMode]);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Real-Time Object Detection & Counting System</h1>
        <div className="status-bar">
          <span className={`status ${cameraRunning ? 'running' : 'stopped'}`}>
            {status}
          </span>
        </div>
      </header>
      
      <div className="main-container">
        <div className="content-wrapper">
          {/* Left: Video Display */}
          <div className="video-section">
            <ControlPanel
              cameraRunning={cameraRunning}
              settings={settings}
              lineMode={lineMode}
              onStartCamera={handleStartCamera}
              onStopCamera={handleStopCamera}
              onUpdateSettings={handleUpdateSettings}
              onToggleLineMode={handleToggleLineMode}
              onResetLine={handleResetLine}
              onCameraSourceChange={(source) => setSettings({ ...settings, cameraSource: source })}
            />
            
            <VideoDisplay
              videoFrame={videoFrame}
              onLineDraw={handleSetLine}
              lineMode={lineMode}
            />
          </div>
          
          {/* Right: Dashboard/Catalog */}
          <div className="dashboard-section">
            <Dashboard stats={stats} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
