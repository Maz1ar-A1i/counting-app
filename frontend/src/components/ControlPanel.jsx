import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Power, Settings, LineChart, Camera as CameraIcon } from 'lucide-react';

function ControlPanel({
  cameraRunning,
  settings,
  lineMode,
  onStartCamera,
  onStopCamera,
  onUpdateSettings,
  onToggleLineMode,
  onResetLine,
  onCameraSourceChange
}) {
  const [localConfidence, setLocalConfidence] = useState(settings.confidence);
  const [localIou, setLocalIou] = useState(settings.iou);

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
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-slate-800 rounded-xl p-4 shadow-2xl border border-slate-700 space-y-4"
    >
      {/* Camera Controls */}
      <div>
        <h3 className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-2">
          <CameraIcon className="w-4 h-4 text-blue-400" />
          Camera Controls
        </h3>
        
        <div className="space-y-3">
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1">
              Camera Source
            </label>
            <input
              type="text"
              value={settings.cameraSource}
              onChange={(e) => onCameraSourceChange(e.target.value)}
              placeholder="0 for webcam"
              disabled={cameraRunning}
              className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white text-sm placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
            />
          </div>
          
          <div className="flex gap-2">
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={onStartCamera}
              disabled={cameraRunning}
              className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-green-600 hover:bg-green-700 disabled:bg-slate-600 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors"
            >
              <Power className="w-3 h-3" />
              Start Camera
            </motion.button>
            
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={onStopCamera}
              disabled={!cameraRunning}
              className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-red-600 hover:bg-red-700 disabled:bg-slate-600 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors"
            >
              <Power className="w-3 h-3" />
              Stop Camera
            </motion.button>
          </div>
        </div>
      </div>

      {/* Detection Settings */}
      <div>
        <h3 className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-2">
          <Settings className="w-4 h-4 text-blue-400" />
          Detection Settings
        </h3>
        
        <div className="space-y-3">
          <div>
            <div className="flex justify-between items-center mb-1">
              <label className="text-xs font-medium text-slate-300">
                Confidence
              </label>
              <span className="text-xs text-blue-400 font-mono">
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
            <div className="flex justify-between items-center mb-1">
              <label className="text-xs font-medium text-slate-300">
                NMS IoU
              </label>
              <span className="text-xs text-blue-400 font-mono">
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

      {/* Line Counting */}
      <div>
        <h3 className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-2">
          <LineChart className="w-4 h-4 text-blue-400" />
          Line Counting
        </h3>
        
        <div className="space-y-2">
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={onToggleLineMode}
            className={`w-full px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
              lineMode
                ? 'bg-cyan-600 hover:bg-cyan-700 text-white'
                : 'bg-slate-700 hover:bg-slate-600 text-slate-300'
            }`}
          >
            {lineMode ? 'Disable Line Counting (L)' : 'Enable Line Counting (L)'}
          </motion.button>
          
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={onResetLine}
            disabled={!lineMode}
            className="w-full px-3 py-2 bg-orange-600 hover:bg-orange-700 disabled:bg-slate-600 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors"
          >
            Reset Line
          </motion.button>
          
          <p className="text-xs text-slate-400 mt-1">
            {lineMode 
              ? 'Press L or click to draw counting line on video'
              : 'Enable line counting to draw a counting line'}
          </p>
        </div>
      </div>
    </motion.div>
  );
}

export default ControlPanel;
