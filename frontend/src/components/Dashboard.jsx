import React from 'react';
import { motion } from 'framer-motion';
import { Camera, Settings, BarChart3, Activity, Zap } from 'lucide-react';

function Dashboard({ onOpenCamera, settings, onUpdateSettings, onCameraSourceChange, connectionStatus, status }) {
  const [localConfidence, setLocalConfidence] = React.useState(settings.confidence);
  const [localIou, setLocalIou] = React.useState(settings.iou);

  const handleConfidenceChange = (value) => {
    const conf = parseFloat(value);
    setLocalConfidence(conf);
    onUpdateSettings({ confidence: conf });
  };

  const handleIouChange = (value) => {
    const iou = parseFloat(value);
    setLocalIou(iou);
    onUpdateSettings({ iou: iou });
  };

  return (
    <div className="h-screen w-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 overflow-hidden">
      {/* Status Bar */}
      <div className="bg-slate-800 border-b border-slate-700 px-6 py-2 flex items-center justify-end gap-4">
        <div className="flex items-center gap-2">
          <Activity className={`w-4 h-4 ${
            connectionStatus === 'connected' ? 'text-green-400' : 
            connectionStatus === 'error' ? 'text-red-400' : 'text-gray-400'
          }`} />
          <span className={`text-sm font-medium ${
            connectionStatus === 'connected' ? 'text-green-400' : 
            connectionStatus === 'error' ? 'text-red-400' : 'text-gray-400'
          }`}>
            {connectionStatus === 'connected' ? '🟢 Connected' : 
             connectionStatus === 'error' ? '🔴 Error' : '⚪ Disconnected'}
          </span>
        </div>
        <div className="text-sm text-slate-400">{status}</div>
      </div>

      <div className="flex items-center justify-center min-h-[calc(100vh-60px)] p-8">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="max-w-4xl w-full space-y-6"
        >
          {/* Header */}
          <div className="text-center mb-8">
            <motion.div
              initial={{ y: -20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              className="flex items-center justify-center gap-3 mb-4"
            >
              <Camera className="w-12 h-12 text-blue-400" />
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                Object Detection & Counting System
              </h1>
            </motion.div>
            <p className="text-slate-400 text-lg">
              Real-time object detection, classification, and counting
            </p>
          </div>

          {/* Main Card */}
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="bg-slate-800 rounded-2xl p-8 shadow-2xl border border-slate-700"
          >
            {/* Settings Section */}
            <div className="mb-8">
              <h2 className="text-xl font-semibold text-slate-200 mb-6 flex items-center gap-2">
                <Settings className="w-6 h-6 text-blue-400" />
                Configuration
              </h2>
              
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Camera Source
                  </label>
                  <input
                    type="text"
                    value={settings.cameraSource}
                    onChange={(e) => onCameraSourceChange(e.target.value)}
                    placeholder="0 for webcam, or IP camera URL"
                    className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div className="grid grid-cols-2 gap-6">
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-sm font-medium text-slate-300">
                        Confidence Threshold
                      </label>
                      <span className="text-sm text-blue-400 font-mono">
                        {localConfidence.toFixed(2)}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="0.95"
                      step="0.05"
                      value={localConfidence}
                      onChange={(e) => handleConfidenceChange(e.target.value)}
                      className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-sm font-medium text-slate-300">
                        NMS IoU Threshold
                      </label>
                      <span className="text-sm text-blue-400 font-mono">
                        {localIou.toFixed(2)}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="0.9"
                      step="0.05"
                      value={localIou}
                      onChange={(e) => handleIouChange(e.target.value)}
                      className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Open Camera Button */}
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onOpenCamera}
              className="w-full bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white font-bold py-4 px-8 rounded-xl text-lg shadow-lg flex items-center justify-center gap-3 transition-all"
            >
              <Camera className="w-6 h-6" />
              OPEN CAMERA
            </motion.button>

            <p className="text-center text-slate-400 text-sm mt-4">
              Click to open camera view with live detection and catalog
            </p>
          </motion.div>

          {/* Info Cards */}
          <div className="grid grid-cols-3 gap-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-slate-800 rounded-xl p-4 border border-slate-700 text-center"
            >
              <BarChart3 className="w-8 h-8 text-blue-400 mx-auto mb-2" />
              <h3 className="text-sm font-semibold text-slate-200 mb-1">Real-time Detection</h3>
              <p className="text-xs text-slate-400">Multiple object types</p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="bg-slate-800 rounded-xl p-4 border border-slate-700 text-center"
            >
              <Zap className="w-8 h-8 text-green-400 mx-auto mb-2" />
              <h3 className="text-sm font-semibold text-slate-200 mb-1">Line Counting</h3>
              <p className="text-xs text-slate-400">Cross-line detection</p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="bg-slate-800 rounded-xl p-4 border border-slate-700 text-center"
            >
              <BarChart3 className="w-8 h-8 text-orange-400 mx-auto mb-2" />
              <h3 className="text-sm font-semibold text-slate-200 mb-1">Live Catalog</h3>
              <p className="text-xs text-slate-400">Real-time statistics</p>
            </motion.div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

export default Dashboard;
