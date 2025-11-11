import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Video, Power, Settings } from 'lucide-react';
import VideoFeed from '../components/VideoFeed';
import StatsPanel from '../components/StatsPanel';
import { socket, api, cameraApi } from '../services/api';

function LiveCamera() {
  const navigate = useNavigate();
  const [cameraRunning, setCameraRunning] = useState(false);
  const [videoFrame, setVideoFrame] = useState(null);
  const [stats, setStats] = useState({
    fps: 0,
    counts: {},
    total_count: 0,
    tracked_objects: []
  });
  const [settings, setSettings] = useState({
    confidence: 0.25,  // Lower threshold for better detection in complex scenes
    iou: 0.45,
    cameraSource: '0'
  });
  const [status, setStatus] = useState('Ready');
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [detectionLog, setDetectionLog] = useState([]);
  
  const accumulatedCountsRef = useRef({});
  const lastStatsRef = useRef(null);
  const detectionLogRef = useRef([]);

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

  const handleStartCamera = async () => {
    try {
      setStatus('Connecting to camera...');
      setConnectionStatus('connecting');
      
      const source = parseInt(settings.cameraSource) || settings.cameraSource;
      
      const response = await cameraApi.post('/api/camera/start', { source });
      
      if (response.data.success) {
        setCameraRunning(true);
        setStatus('Camera connected and streaming');
        setConnectionStatus('connected');
        accumulatedCountsRef.current = {};
        detectionLogRef.current = [];
        setDetectionLog([]);
        addToLog('Camera connected successfully');
      } else {
        const errorMsg = response.data.message || 'Failed to start camera';
        setStatus(`Error: ${errorMsg}`);
        setConnectionStatus('error');
        addToLog(`Camera connection failed: ${errorMsg}`);
      }
    } catch (error) {
      let errorMsg = 'Unknown error';
      
      if (error.response) {
        errorMsg = error.response.data?.message || error.response.data?.error_details || error.response.statusText;
      } else if (error.request) {
        if (error.code === 'ECONNABORTED') {
          errorMsg = 'Connection timeout. RTSP camera may be slow to connect. Please try again.';
        } else {
          errorMsg = 'No response from server. Make sure the backend is running.';
        }
      } else {
        errorMsg = error.message || 'Failed to start camera';
      }
      
      setStatus(`Error: ${errorMsg}`);
      setConnectionStatus('error');
      addToLog(`Camera connection failed: ${errorMsg}`);
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

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (cameraRunning) {
        handleStopCamera();
      }
    };
  }, []);

  const formatClassName = (name) => {
    return name.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 overflow-y-auto">
      {/* Header */}
      <motion.header
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="bg-slate-800/50 backdrop-blur-sm border-b border-slate-700 px-6 py-4 shadow-lg sticky top-0 z-10"
      >
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => {
                if (cameraRunning) {
                  handleStopCamera();
                }
                navigate('/');
              }}
              className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-slate-300" />
            </motion.button>
            <div className="flex items-center gap-3">
              <Video className="w-6 h-6 text-blue-400" />
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                Live Camera Detection
              </h1>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={cameraRunning ? handleStopCamera : handleStartCamera}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                cameraRunning
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'bg-green-600 hover:bg-green-700 text-white'
              }`}
            >
              <Power className="w-4 h-4" />
              {cameraRunning ? 'Stop Camera' : 'Start Camera'}
            </motion.button>
          </div>
        </div>
      </motion.header>

      {/* Main Content - Scrollable */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Side - Camera View (2 columns) */}
          <div className="lg:col-span-2 space-y-6">
            {/* Camera Source Selection */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-slate-800 rounded-xl p-4 border border-slate-700"
            >
              <div className="flex items-center gap-2 mb-3">
                <Settings className="w-5 h-5 text-blue-400" />
                <h3 className="text-sm font-semibold text-slate-200">Camera Source</h3>
              </div>
              <input
                type="text"
                value={settings.cameraSource}
                onChange={(e) => setSettings({ ...settings, cameraSource: e.target.value })}
                placeholder="0 for webcam, or RTSP/IP camera URL"
                disabled={cameraRunning}
                className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
              />
            </motion.div>

            {/* Video Feed */}
            <VideoFeed
              videoFrame={videoFrame}
              onLineDraw={null}
              lineMode={false}
              cameraRunning={cameraRunning}
              mirror={false}
            />
          </div>

          {/* Right Side - Object Catalog (1 column) */}
          <div className="lg:col-span-1">
            <StatsPanel 
              stats={stats} 
              detectionLog={detectionLog}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default LiveCamera;

