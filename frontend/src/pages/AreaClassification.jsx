import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Ruler, Power, Settings } from 'lucide-react';
import { api, cameraApi, socket } from '../services/api';

function AreaClassification() {
  const navigate = useNavigate();
  const [running, setRunning] = useState(false);
  const [frameB64, setFrameB64] = useState(null);
  const [stats, setStats] = useState({ fps: 0, counts: {}, total_regions: 0 });
  const [status, setStatus] = useState('Ready');
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [settings, setSettings] = useState({
    cameraSource: '0',
    min_area: 500
  });

  const canvasRef = useRef(null);
  const containerRef = useRef(null);

  useEffect(() => {
    socket.on('area_frame', (data) => {
      setFrameB64(data.frame);
      if (data.stats) {
        setStats({
          fps: data.stats.fps || 0,
          counts: data.stats.counts || {},
          total_regions: data.stats.total_regions || 0
        });
      }
    });
    socket.on('area_stats', (data) => {
      setStats((prev) => ({
        ...prev,
        fps: data.fps || prev.fps,
        counts: data.counts || prev.counts,
        total_regions: data.total_regions || prev.total_regions
      }));
    });
    return () => {
      socket.off('area_frame');
      socket.off('area_stats');
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frameB64) return;
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.onload = () => {
      const container = containerRef.current;
      const cw = (container?.clientWidth || 800) - 32;
      const ch = Math.max(400, container?.clientHeight || 400);
      const scale = Math.min(cw / img.width, ch / img.height, 1.0);
      const w = Math.floor(img.width * scale);
      const h = Math.floor(img.height * scale);
      canvas.width = w;
      canvas.height = h;
      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(img, 0, 0, w, h);
    };
    img.src = `data:image/jpeg;base64,${frameB64}`;
  }, [frameB64]);

  const start = async () => {
    try {
      setStatus('Starting area classification...');
      setConnectionStatus('connecting');
      const source = parseInt(settings.cameraSource) || settings.cameraSource;
      const resp = await cameraApi.post('/api/area/start', {
        source,
        min_area: settings.min_area
      });
      if (resp.data.success) {
        setRunning(true);
        setConnectionStatus('connected');
        setStatus('Area classification running');
      } else {
        setConnectionStatus('error');
        setStatus(resp.data.message || 'Failed to start');
      }
    } catch (e) {
      setConnectionStatus('error');
      setStatus(e.response?.data?.message || e.message);
    }
  };

  const stop = async () => {
    try {
      const resp = await api.post('/api/area/stop');
      if (resp.data.success) {
        setRunning(false);
        setFrameB64(null);
        setStats({ fps: 0, counts: {}, total_regions: 0 });
        setStatus('Stopped');
        setConnectionStatus('disconnected');
      }
    } catch (e) {
      setStatus(e.message);
    }
  };

  const updateSettings = async (partial) => {
    const next = { ...settings, ...partial };
    setSettings(next);
    try {
      await api.post('/api/area/settings', {
        min_area: next.min_area
      });
    } catch (e) {
      // ignore
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 overflow-y-auto">
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
                if (running) stop();
                navigate('/');
              }}
              className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-slate-300" />
            </motion.button>
            <div className="flex items-center gap-3">
              <Ruler className="w-6 h-6 text-amber-400" />
              <h1 className="text-xl font-bold bg-gradient-to-r from-amber-400 to-yellow-400 bg-clip-text text-transparent">
                Area-wise Classification
              </h1>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={running ? stop : start}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                running ? 'bg-red-600 hover:bg-red-700 text-white' : 'bg-green-600 hover:bg-green-700 text-white'
              }`}
            >
              <Power className="w-4 h-4" />
              {running ? 'Stop' : 'Start'}
            </motion.button>
          </div>
        </div>
      </motion.header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="bg-slate-800 rounded-xl p-4 border border-slate-700">
              <div className="flex items-center gap-2 mb-3">
                <Settings className="w-5 h-5 text-amber-400" />
                <h3 className="text-sm font-semibold text-slate-200">Source & Settings</h3>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <input
                  type="text"
                  value={settings.cameraSource}
                  onChange={(e) => setSettings({ ...settings, cameraSource: e.target.value })}
                  placeholder="0 for webcam, or RTSP/IP camera URL"
                  disabled={running}
                  className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-amber-500 disabled:opacity-50"
                />
                <input
                  type="number"
                  min={0}
                  value={settings.min_area}
                  onChange={(e) => updateSettings({ min_area: Math.max(0, parseInt(e.target.value) || 0) })}
                  className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-amber-500"
                />
                <div className="px-4 py-2 text-slate-300 bg-slate-700/50 rounded-lg border border-slate-600">
                  Status: <span className="font-medium">{status}</span>
                </div>
              </div>
            </motion.div>

            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="bg-slate-800 rounded-xl p-4 border border-slate-700">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-lg font-semibold text-slate-200 flex items-center gap-2">
                  <Ruler className="w-5 h-5 text-amber-400" />
                  Area View
                </h2>
                <div className="text-sm text-slate-400">{connectionStatus}</div>
              </div>
              <div ref={containerRef} className="relative bg-black rounded-lg overflow-hidden flex items-center justify-center" style={{ minHeight: '400px', height: '60vh' }}>
                {frameB64 ? (
                  <canvas ref={canvasRef} style={{ display: 'block', maxWidth: '100%', maxHeight: '100%' }} />
                ) : (
                  <div className="flex flex-col items-center justify-center h-full text-slate-500">
                    <Ruler className="w-16 h-16 mb-4 opacity-50" />
                    <p className="text-lg font-medium">Not Running</p>
                    <p className="text-sm mt-2">Click "Start" to begin area-wise classification</p>
                  </div>
                )}
              </div>
            </motion.div>
          </div>

          <div className="lg:col-span-1">
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="bg-slate-800 rounded-xl p-4 border border-slate-700">
              <h3 className="text-lg font-semibold text-slate-200 mb-3">Statistics</h3>
              <div className="space-y-2 text-slate-300">
                <div className="flex justify-between"><span>FPS</span><span className="font-mono">{stats.fps.toFixed(1)}</span></div>
                <div className="flex justify-between"><span>Total Regions</span><span className="font-mono">{stats.total_regions}</span></div>
                <div className="mt-3">
                  <div className="text-sm text-slate-400 mb-1">Counts by size</div>
                  <div className="space-y-1">
                    {Object.keys(stats.counts || {}).length === 0 && <div className="text-slate-500 text-sm">No regions</div>}
                    {Object.entries(stats.counts || {}).map(([k, v]) => (
                      <div key={k} className="flex justify-between text-sm"><span className="capitalize">{k}</span><span className="font-mono">{v}</span></div>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AreaClassification;


