import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Activity, HardDrive, Cpu, Gauge } from 'lucide-react';
import { systemApi } from '../services/api';

function StatRow({ label, value }) {
  return (
    <div className="flex items-center justify-between py-1 text-slate-300">
      <span>{label}</span>
      <span className="font-mono">{value}</span>
    </div>
  );
}

function Card({ title, icon: Icon, children }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-slate-800 rounded-xl p-4 border border-slate-700"
    >
      <div className="flex items-center gap-2 mb-3">
        <Icon className="w-5 h-5 text-blue-400" />
        <h3 className="text-sm font-semibold text-slate-200">{title}</h3>
      </div>
      {children}
    </motion.div>
  );
}

function SystemStatus() {
  const navigate = useNavigate();
  const [metrics, setMetrics] = useState(null);
  const [recs, setRecs] = useState(null);
  const [error, setError] = useState(null);

  const fetchAll = async () => {
    try {
      const [m, r] = await Promise.all([systemApi.getMetrics(), systemApi.getRecommendations()]);
      setMetrics(m.data);
      setRecs(r.data);
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  };

  useEffect(() => {
    fetchAll();
    const id = setInterval(fetchAll, 3000);
    return () => clearInterval(id);
  }, []);

  const cpu = metrics?.cpu || {};
  const ram = metrics?.ram || {};
  const disk = metrics?.disk || {};
  const gpu = metrics?.gpu || null;

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
              onClick={() => navigate('/')}
              className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-slate-300" />
            </motion.button>
            <div className="flex items-center gap-3">
              <Activity className="w-6 h-6 text-blue-400" />
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                System Status & Metrics
              </h1>
            </div>
          </div>
          <button onClick={fetchAll} className="px-3 py-2 rounded-lg bg-slate-700 text-slate-200 border border-slate-600 hover:bg-slate-600">
            Refresh
          </button>
        </div>
      </motion.header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {error && <div className="mb-4 text-red-400">Error: {error}</div>}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <Card title="CPU" icon={Cpu}>
            <StatRow label="Usage" value={`${cpu.percent ?? 0}%`} />
            <StatRow label="Physical Cores" value={cpu.cores_physical ?? '-'} />
            <StatRow label="Logical Cores" value={cpu.cores_logical ?? '-'} />
            <StatRow label="Freq (current)" value={cpu.freq_current_mhz ? `${cpu.freq_current_mhz.toFixed(0)} MHz` : '-'} />
            <StatRow label="Freq (max)" value={cpu.freq_max_mhz ? `${cpu.freq_max_mhz.toFixed(0)} MHz` : '-'} />
            <StatRow label="Processor" value={cpu.processor || '-'} />
          </Card>

          <Card title="Memory (RAM)" icon={Gauge}>
            <StatRow label="Used" value={`${ram.used_gb ?? 0} GB`} />
            <StatRow label="Available" value={`${ram.available_gb ?? 0} GB`} />
            <StatRow label="Total" value={`${ram.total_gb ?? 0} GB`} />
            <StatRow label="Usage" value={`${ram.percent ?? 0}%`} />
          </Card>

          <Card title="Disk" icon={HardDrive}>
            <StatRow label="Type" value={disk.kind || 'unknown'} />
            <StatRow label="Used" value={`${disk.used_gb ?? 0} GB`} />
            <StatRow label="Free" value={`${disk.free_gb ?? 0} GB`} />
            <StatRow label="Total" value={`${disk.total_gb ?? 0} GB`} />
            <StatRow label="Usage" value={`${disk.percent ?? 0}%`} />
          </Card>

          <Card title="GPU" icon={Activity}>
            {gpu ? (
              <>
                <StatRow label="Name" value={gpu.name || '-'} />
                <StatRow label="Total VRAM" value={`${gpu.total_mem_gb ?? 0} GB`} />
                <StatRow label="Allocated" value={`${gpu.allocated_gb ?? 0} GB`} />
                <StatRow label="Reserved" value={`${gpu.reserved_gb ?? 0} GB`} />
                <StatRow label="CUDA" value={gpu.cuda_version || '-'} />
              </>
            ) : (
              <div className="text-slate-400">No CUDA GPU detected or GPU info unavailable</div>
            )}
          </Card>

          <Card title="App Streams" icon={Activity}>
            <StatRow label="Detector Running" value={metrics?.streams?.detector_running ? 'Yes' : 'No'} />
            <StatRow label="Area Running" value={metrics?.streams?.area_running ? 'Yes' : 'No'} />
          </Card>

          <Card title="Recommendations" icon={Activity}>
            {recs?.recommendations ? (
              <div className="space-y-3 text-slate-300">
                <div>
                  <div className="text-sm text-slate-400">Minimum</div>
                  <div className="text-sm">{JSON.stringify(recs.recommendations.minimum)}</div>
                </div>
                <div>
                  <div className="text-sm text-slate-400">Recommended</div>
                  <div className="text-sm">{JSON.stringify(recs.recommendations.recommended)}</div>
                </div>
                <div>
                  <div className="text-sm text-slate-400">Best Results</div>
                  <div className="text-sm">{JSON.stringify(recs.recommendations.for_best_results)}</div>
                </div>
              </div>
            ) : (
              <div className="text-slate-400">Recommendations unavailable</div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}

export default SystemStatus;


