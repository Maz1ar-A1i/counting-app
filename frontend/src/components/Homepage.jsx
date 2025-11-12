import React from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Camera, Upload, LineChart, Ruler } from 'lucide-react';

function Homepage() {
  const navigate = useNavigate();

  const features = [
    {
      id: 'live-camera',
      title: 'Live Camera Detection',
      description: 'Real-time object detection and counting from your camera feed',
      icon: Camera,
      color: 'from-blue-600 to-cyan-600',
      hoverColor: 'from-blue-700 to-cyan-700',
      path: '/live-camera'
    },
    {
      id: 'upload-picture',
      title: 'Upload Picture for Detection',
      description: 'Upload an image and detect/count objects in the picture',
      icon: Upload,
      color: 'from-purple-600 to-pink-600',
      hoverColor: 'from-purple-700 to-pink-700',
      path: '/upload-picture'
    },
    {
      id: 'live-camera-line',
      title: 'Live Camera Detection with Line Feature',
      description: 'Real-time detection with line-crossing counting capabilities',
      icon: LineChart,
      color: 'from-green-600 to-emerald-600',
      hoverColor: 'from-green-700 to-emerald-700',
      path: '/live-camera-line'
    },
    {
      id: 'area-classification',
      title: 'Area-wise Classification',
      description: 'Detect regions by area size without ML models (separate mode)',
      icon: Ruler,
      color: 'from-amber-600 to-yellow-600',
      hoverColor: 'from-amber-700 to-yellow-700',
      path: '/area-classification'
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 overflow-y-auto">
      {/* Header with Logo */}
      <motion.header
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="bg-slate-800/50 backdrop-blur-sm border-b border-slate-700 px-6 py-4 shadow-lg sticky top-0 z-10"
      >
        <div className="max-w-7xl mx-auto flex items-center justify-center">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl flex items-center justify-center shadow-lg">
              <Camera className="w-7 h-7 text-white" />
            </div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
              Object Counting System
            </h1>
          </div>
        </div>
      </motion.header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 py-12">
        {/* Welcome Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Advanced Object Detection & Counting
          </h2>
          <p className="text-xl text-slate-400 max-w-2xl mx-auto">
            Choose a detection mode to get started with real-time or image-based object counting
          </p>
        </motion.div>

        {/* Feature Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <motion.div
                key={feature.id}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ scale: 1.02, y: -5 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => navigate(feature.path)}
                className="bg-slate-800 rounded-2xl p-8 border border-slate-700 cursor-pointer shadow-xl hover:shadow-2xl transition-all duration-300 group"
              >
                <div className={`w-16 h-16 bg-gradient-to-br ${feature.color} rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
                  <Icon className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-bold text-white mb-3">
                  {feature.title}
                </h3>
                <p className="text-slate-400 mb-6">
                  {feature.description}
                </p>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className={`w-full bg-gradient-to-r ${feature.color} hover:${feature.hoverColor} text-white font-semibold py-3 px-6 rounded-xl transition-all duration-300`}
                >
                  Get Started
                </motion.button>
              </motion.div>
            );
          })}
        </div>

        {/* Info Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-slate-800/50 backdrop-blur-sm rounded-2xl p-8 border border-slate-700"
        >
          <h3 className="text-2xl font-bold text-white mb-4 text-center">
            Features
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-4xl mb-2">🎯</div>
              <h4 className="text-lg font-semibold text-white mb-2">Real-time Detection</h4>
              <p className="text-slate-400 text-sm">Detect objects in real-time with high accuracy</p>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-2">📊</div>
              <h4 className="text-lg font-semibold text-white mb-2">Object Counting</h4>
              <p className="text-slate-400 text-sm">Count and categorize detected objects automatically</p>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-2">🚀</div>
              <h4 className="text-lg font-semibold text-white mb-2">Line Crossing</h4>
              <p className="text-slate-400 text-sm">Track objects crossing designated lines</p>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-2">📐</div>
              <h4 className="text-lg font-semibold text-white mb-2">Area-wise Classification</h4>
              <p className="text-slate-400 text-sm">Classify regions by size without touching ML models</p>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

export default Homepage;

