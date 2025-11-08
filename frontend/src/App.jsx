import React, { useState, useEffect, useCallback, useRef } from 'react';
import Dashboard from './components/Dashboard';
import CameraView from './components/CameraView';
import { socket, api } from './services/api';

function App() {
  const [showCameraView, setShowCameraView] = useState(false);
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
  const [status, setStatus] = useState('Ready');
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  
  const accumulatedCountsRef = useRef({});
  const lastStatsRef = useRef(null);
  const detectionLogRef = useRef([]);
  const [detectionLog, setDetectionLog] = useState([]);

  const addToLog = (message) => {
    const timestamp = new Date().toLocaleTimeString();
    detectionLogRef.current.unshift({
      id: Date.now(),
      message,
      timestamp,
      type: 'detection'
    });
    if (detectionLogRef.current.length > 50) {
      detectionLogRef.current.pop();
    }
    setDetectionLog([...detectionLogRef.current]);
  };

  const handleOpenCamera = () => {
    setShowCameraView(true);
  };

  const handleCloseCameraView = async () => {
    if (cameraRunning) {
      await handleStopCamera();
    }
    setShowCameraView(false);
    setVideoFrame(null);
  };

  const handleStartCamera = async () => {
    try {
      const source = parseInt(settings.cameraSource) || settings.cameraSource;
      const response = await api.post('/api/camera/start', { source });
      if (response.data.success) {
        setCameraRunning(true);
        setStatus('Camera connected and streaming');
        setConnectionStatus('connected');
        accumulatedCountsRef.current = {};
        detectionLogRef.current = [];
        setDetectionLog([]);
        addToLog('Camera connected successfully');
      } else {
        setStatus(`Error: ${response.data.message}`);
        setConnectionStatus('error');
      }
    } catch (error) {
      setStatus(`Error: ${error.message}`);
      setConnectionStatus('error');
    }
  };

  const handleStopCamera = async () => {
    try {
      const response = await api.post('/api/camera/stop');
      if (response.data.success) {
        setCameraRunning(false);
        setStatus('Camera disconnected');
        setConnectionStatus('disconnected');
        setVideoFrame(null);
        accumulatedCountsRef.current = {};
        setStats({
          fps: 0,
          counts: {},
          total_count: 0,
          tracked_objects: []
        });
        addToLog('Camera disconnected');
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
      console.log('Setting line:', start, end);
      const response = await api.post('/api/line/set', { start, end });
      if (response.data.success) {
        setStatus('Counting line set!');
        addToLog('Counting line drawn on video');
      } else {
        setStatus(`Error: ${response.data.message}`);
      }
    } catch (error) {
      console.error('Line set error:', error);
      setStatus(`Error: ${error.message}`);
    }
  };

  const handleResetLine = async () => {
    try {
      await api.post('/api/line/reset');
      setStatus('Line reset');
      accumulatedCountsRef.current = {};
      setStats(prev => ({
        ...prev,
        counts: {},
        total_count: 0
      }));
      addToLog('Counting line reset');
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
        addToLog(response.data.enabled ? 'Line counting mode enabled' : 'Line counting mode disabled');
      }
    } catch (error) {
      setStatus(`Error: ${error.message}`);
    }
  }, []);

  useEffect(() => {
    socket.on('connect', () => {
      console.log('Connected to server');
      setStatus('Connected to server');
      setConnectionStatus('connected');
    });

    socket.on('video_frame', (data) => {
      setVideoFrame(data.frame);
      if (data.stats) {
        const newCounts = data.stats.counts || {};
        const prevCounts = lastStatsRef.current?.counts || {};
        
        Object.keys(newCounts).forEach(key => {
          if ((newCounts[key] || 0) > (prevCounts[key] || 0)) {
            addToLog(`${formatClassName(key)} detected (${newCounts[key]} total)`);
          }
        });
        
        Object.keys(newCounts).forEach(key => {
          if (!accumulatedCountsRef.current[key]) {
            accumulatedCountsRef.current[key] = 0;
          }
          if (newCounts[key] > accumulatedCountsRef.current[key]) {
            accumulatedCountsRef.current[key] = newCounts[key];
          }
        });
        
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
      setStats(prev => ({
        ...prev,
        fps: data.fps || prev.fps,
        tracked_objects: data.tracked_objects || prev.tracked_objects
      }));
    });

    socket.on('disconnect', () => {
      console.log('Disconnected from server');
      setStatus('Disconnected from server');
      setConnectionStatus('disconnected');
    });

    return () => {
      socket.off('connect');
      socket.off('video_frame');
      socket.off('stats_update');
      socket.off('disconnect');
    };
  }, []);

  useEffect(() => {
    const handleKeyPress = (e) => {
      if (showCameraView) {
        if (e.key === 'l' || e.key === 'L') {
          e.preventDefault();
          handleToggleLineMode();
        } else if (e.key === 'q' || e.key === 'Q') {
          if (cameraRunning) {
            e.preventDefault();
            handleCloseCameraView();
          }
        }
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => {
      window.removeEventListener('keydown', handleKeyPress);
    };
  }, [lineMode, cameraRunning, showCameraView, handleToggleLineMode]);

  const formatClassName = (name) => {
    return name.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
  };

  // COMPLETELY SEPARATE PAGES - Only show one at a time
  // When showCameraView is true, ONLY show CameraView (full screen, no dashboard)
  // When showCameraView is false, ONLY show Dashboard (no camera view)
  
  if (showCameraView) {
    return (
      <div className="fixed inset-0 z-50">
        <CameraView
          videoFrame={videoFrame}
          stats={stats}
          detectionLog={detectionLog}
          lineMode={lineMode}
          cameraRunning={cameraRunning}
          onLineDraw={handleSetLine}
          onClose={handleCloseCameraView}
          onToggleLineMode={handleToggleLineMode}
          onResetLine={handleResetLine}
          onStartCamera={handleStartCamera}
          onStopCamera={handleStopCamera}
        />
      </div>
    );
  }

  // DASHBOARD PAGE - Only shown when showCameraView is false
  // This is a completely separate page with NO camera view
  return (
    <div className="fixed inset-0 z-40">
      <Dashboard
        onOpenCamera={handleOpenCamera}
        settings={settings}
        onUpdateSettings={handleUpdateSettings}
        onCameraSourceChange={(source) => setSettings({ ...settings, cameraSource: source })}
        connectionStatus={connectionStatus}
        status={status}
      />
    </div>
  );
}

export default App;
