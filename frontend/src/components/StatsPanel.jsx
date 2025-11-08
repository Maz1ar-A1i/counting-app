import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BarChart3, Activity, Clock, TrendingUp } from 'lucide-react';

function StatsPanel({ stats, detectionLog }) {
  const { fps, counts, total_count } = stats;
  const [displayFPS, setDisplayFPS] = useState(0);

  useEffect(() => {
    if (fps > 0) {
      const cappedFPS = Math.min(fps, 60);
      setDisplayFPS(prev => prev * 0.7 + cappedFPS * 0.3);
    } else {
      setDisplayFPS(0);
    }
  }, [fps]);

  const getIconForClass = (class_name) => {
    const icons = {
      'person': '👤', 'car': '🚗', 'truck': '🚚', 'bus': '🚌',
      'motorcycle': '🏍️', 'bicycle': '🚲', 'dog': '🐕', 'cat': '🐈',
      'bird': '🐦', 'laptop': '💻', 'cell phone': '📱', 'backpack': '🎒',
      'handbag': '👜', 'suitcase': '🧳', 'bottle': '🍼', 'cup': '☕',
      'chair': '🪑', 'couch': '🛋️', 'tv': '📺', 'keyboard': '⌨️', 'mouse': '🖱️'
    };
    return icons[class_name] || '📦';
  };

  const formatClassName = (name) => {
    return name.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
  };

  return (
    <div className="space-y-4">
      {/* Stats Cards */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-slate-800 rounded-xl p-4 shadow-2xl border border-slate-700"
      >
        <h2 className="text-lg font-semibold text-slate-200 mb-4 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-blue-400" />
          Object Catalog
        </h2>
        
        <div className="grid grid-cols-2 gap-3 mb-4">
          <motion.div
            whileHover={{ scale: 1.05 }}
            className="bg-gradient-to-br from-green-600 to-green-700 rounded-lg p-3 text-center"
          >
            <div className="text-xs text-green-100 mb-1">Total Counted</div>
            <div className="text-2xl font-bold text-white">{total_count || 0}</div>
          </motion.div>
          
          <motion.div
            whileHover={{ scale: 1.05 }}
            className="bg-gradient-to-br from-orange-600 to-orange-700 rounded-lg p-3 text-center"
          >
            <div className="text-xs text-orange-100 mb-1 flex items-center justify-center gap-1">
              <Activity className="w-3 h-3" />
              FPS
            </div>
            <div className="text-2xl font-bold text-white">{displayFPS.toFixed(1)}</div>
          </motion.div>
        </div>

        {/* Object Counts - Scrollable */}
        <div>
          <h3 className="text-xs font-semibold text-slate-300 mb-2 flex items-center gap-2">
            <TrendingUp className="w-3 h-3" />
            Counts by Category
          </h3>
          
          <div className="space-y-2 overflow-y-auto" style={{ maxHeight: '250px' }}>
            <AnimatePresence>
              {Object.keys(counts).length > 0 ? (
                Object.entries(counts)
                  .sort((a, b) => b[1] - a[1])
                  .map(([class_name, count], index) => (
                    <motion.div
                      key={class_name}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: 20 }}
                      transition={{ delay: index * 0.05 }}
                      className="flex items-center justify-between bg-slate-700 rounded-lg p-2 hover:bg-slate-600 transition-colors border-l-4 border-blue-500"
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-lg">{getIconForClass(class_name)}</span>
                        <span className="text-slate-200 text-sm font-medium">
                          {formatClassName(class_name)}
                        </span>
                      </div>
                      <span className="text-blue-400 font-bold">{count}</span>
                    </motion.div>
                  ))
              ) : (
                <div className="text-center py-6 text-slate-500">
                  <div className="text-3xl mb-2">📊</div>
                  <p className="text-xs">No objects detected yet</p>
                </div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </motion.div>

      {/* Detection Log - Scrollable */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-slate-800 rounded-xl p-4 shadow-2xl border border-slate-700"
      >
        <h2 className="text-lg font-semibold text-slate-200 mb-4 flex items-center gap-2">
          <Clock className="w-5 h-5 text-blue-400" />
          Detection Log
        </h2>
        
        <div className="space-y-2 overflow-y-auto" style={{ maxHeight: '250px' }}>
          <AnimatePresence>
            {detectionLog && detectionLog.length > 0 ? (
              detectionLog.map((log) => (
                <motion.div
                  key={log.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="bg-slate-700 rounded-lg p-2 text-xs border-l-4 border-cyan-500"
                >
                  <div className="flex items-start justify-between gap-2">
                    <span className="text-slate-300 flex-1 break-words">{log.message}</span>
                    <span className="text-slate-500 text-xs font-mono whitespace-nowrap flex-shrink-0">
                      {log.timestamp}
                    </span>
                  </div>
                </motion.div>
              ))
            ) : (
              <div className="text-center py-4 text-slate-500 text-xs">
                <Clock className="w-6 h-6 mx-auto mb-2 opacity-50" />
                <p>No detections logged yet</p>
              </div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>
    </div>
  );
}

export default StatsPanel;
