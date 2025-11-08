import React from 'react';
import { motion } from 'framer-motion';
import { X, Video, Power } from 'lucide-react';
import VideoFeed from './VideoFeed';
import StatsPanel from './StatsPanel';

function CameraView({
  videoFrame,
  stats,
  detectionLog,
  lineMode,
  cameraRunning,
  onLineDraw,
  onClose,
  onToggleLineMode,
  onResetLine,
  onStartCamera,
  onStopCamera
}) {
  return (
    <div className="h-screen w-screen bg-slate-900 text-white flex flex-col overflow-hidden">
      {/* Header with Close Button */}
      <motion.header 
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="bg-slate-800 border-b border-slate-700 px-6 py-3 shadow-lg flex-shrink-0 flex items-center justify-between"
      >
        <div className="flex items-center gap-3">
          <Video className="w-6 h-6 text-blue-400" />
          <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
            Camera View & Object Catalog
          </h1>
        </div>
        
        <div className="flex items-center gap-3">
          {/* Start/Stop Camera Buttons */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={cameraRunning ? onStopCamera : onStartCamera}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              cameraRunning
                ? 'bg-red-600 hover:bg-red-700 text-white'
                : 'bg-green-600 hover:bg-green-700 text-white'
            }`}
          >
            <Power className="w-4 h-4" />
            {cameraRunning ? 'Stop Camera' : 'Start Camera'}
          </motion.button>
          
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={onClose}
            className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition-colors"
          >
            <X className="w-4 h-4" />
            Close
          </motion.button>
        </div>
      </motion.header>

      {/* Main Content - TRUE SIDE BY SIDE - NO SCROLLING */}
      <div className="flex-1 flex overflow-hidden">
        {/* LEFT SIDE - CAMERA (50%) */}
        <div className="w-1/2 border-r border-slate-700 overflow-y-auto bg-slate-900">
          <div className="p-4">
            <VideoFeed
              videoFrame={videoFrame}
              onLineDraw={onLineDraw}
              lineMode={lineMode}
              cameraRunning={cameraRunning}
            />
            
            {/* Line Controls */}
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-4 bg-slate-800 rounded-xl p-4 border border-slate-700"
            >
              <h3 className="text-sm font-semibold text-slate-200 mb-3">Line Counting Controls</h3>
              <div className="flex gap-2">
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={onToggleLineMode}
                  className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    lineMode
                      ? 'bg-cyan-600 hover:bg-cyan-700 text-white'
                      : 'bg-slate-700 hover:bg-slate-600 text-slate-300'
                  }`}
                >
                  {lineMode ? 'Disable Line Mode (L)' : 'Enable Line Mode (L)'}
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={onResetLine}
                  disabled={!lineMode}
                  className="px-4 py-2 bg-orange-600 hover:bg-orange-700 disabled:bg-slate-600 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors"
                >
                  Reset Line
                </motion.button>
              </div>
              {lineMode && (
                <p className="text-xs text-slate-400 mt-2 text-center">
                  Click and drag on video to draw counting line
                </p>
              )}
            </motion.div>
          </div>
        </div>

        {/* RIGHT SIDE - CATALOG (50%) */}
        <div className="w-1/2 overflow-y-auto bg-slate-900">
          <div className="p-4">
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

export default CameraView;
