import React, { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Upload, ArrowLeft, Image as ImageIcon, Loader2, CheckCircle2, RefreshCw } from 'lucide-react';
import { api } from '../services/api';

function UploadPicture() {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [detectionResults, setDetectionResults] = useState(null);
  const [error, setError] = useState(null);
  const [reloading, setReloading] = useState(false);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Please select a valid image file');
        return;
      }
      setSelectedFile(file);
      setError(null);
      setDetectionResults(null);
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);

      const response = await api.post('/api/detect/image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000, // 60 seconds timeout
      });

      if (response.data.success) {
        setDetectionResults(response.data);
      } else {
        setError(response.data.message || 'Detection failed');
      }
    } catch (err) {
      console.error('Upload error:', err);
      setError(err.response?.data?.message || 'Failed to process image. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setDetectionResults(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleReloadModel = async () => {
    setReloading(true);
    setError(null);
    
    try {
      const response = await api.post('/api/detector/reload');
      if (response.data.success) {
        setError(null);
        alert(`Model reloaded successfully!\nDevice: ${response.data.device}\nModel: ${response.data.model_path}`);
      } else {
        setError(response.data.message || 'Failed to reload model');
      }
    } catch (err) {
      console.error('Reload error:', err);
      setError(err.response?.data?.message || 'Failed to reload model. Please try again.');
    } finally {
      setReloading(false);
    }
  };

  const formatClassName = (name) => {
    return name.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
  };

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
              onClick={() => navigate('/')}
              className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-slate-300" />
            </motion.button>
            <div className="flex items-center gap-3">
              <Upload className="w-6 h-6 text-purple-400" />
              <h1 className="text-xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                Upload Picture for Detection
              </h1>
            </div>
          </div>
          
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleReloadModel}
            disabled={reloading}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg transition-colors text-sm font-medium"
            title="Reload detection model if it wasn't fully loaded"
          >
            <RefreshCw className={`w-4 h-4 ${reloading ? 'animate-spin' : ''}`} />
            {reloading ? 'Reloading...' : 'Reload Model'}
          </motion.button>
        </div>
      </motion.header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Side - Upload Section */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            {/* Upload Card */}
            <div className="bg-slate-800 rounded-2xl p-6 border border-slate-700 shadow-xl">
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <ImageIcon className="w-5 h-5 text-purple-400" />
                Select Image
              </h2>
              
              {!preview ? (
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="border-2 border-dashed border-slate-600 rounded-xl p-12 text-center cursor-pointer hover:border-purple-500 transition-colors"
                >
                  <Upload className="w-16 h-16 text-slate-500 mx-auto mb-4" />
                  <p className="text-slate-300 mb-2">Click to upload an image</p>
                  <p className="text-sm text-slate-500">Supports JPG, PNG, GIF</p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="relative rounded-xl overflow-hidden border border-slate-700">
                    <img
                      src={preview}
                      alt="Preview"
                      className="w-full h-auto max-h-96 object-contain bg-slate-900"
                    />
                    {detectionResults?.annotated_image && (
                      <img
                        src={`data:image/jpeg;base64,${detectionResults.annotated_image}`}
                        alt="Detected"
                        className="w-full h-auto max-h-96 object-contain bg-slate-900 mt-4"
                      />
                    )}
                  </div>
                  <div className="flex gap-2">
                    <motion.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={handleReset}
                      className="flex-1 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors"
                    >
                      Change Image
                    </motion.button>
                    <motion.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={handleUpload}
                      disabled={uploading}
                      className="flex-1 px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    >
                      {uploading ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          Processing...
                        </>
                      ) : (
                        <>
                          <CheckCircle2 className="w-4 h-4" />
                          Detect Objects
                        </>
                      )}
                    </motion.button>
                  </div>
                </div>
              )}

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />

              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-4 p-3 bg-red-900/30 border border-red-700 rounded-lg text-red-300 text-sm"
                >
                  {error}
                </motion.div>
              )}
            </div>
          </motion.div>

          {/* Right Side - Results */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            {/* Results Card */}
            <div className="bg-slate-800 rounded-2xl p-6 border border-slate-700 shadow-xl">
              <h2 className="text-lg font-semibold text-white mb-4">Detection Results</h2>
              
              {detectionResults ? (
                <div className="space-y-4">
                  {/* Summary Stats */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gradient-to-br from-green-600 to-green-700 rounded-lg p-4 text-center">
                      <div className="text-xs text-green-100 mb-1">Total Objects</div>
                      <div className="text-3xl font-bold text-white">
                        {detectionResults.total_count || 0}
                      </div>
                    </div>
                    <div className="bg-gradient-to-br from-blue-600 to-blue-700 rounded-lg p-4 text-center">
                      <div className="text-xs text-blue-100 mb-1">Object Types</div>
                      <div className="text-3xl font-bold text-white">
                        {Object.keys(detectionResults.counts || {}).length}
                      </div>
                    </div>
                  </div>

                  {/* Object Catalog */}
                  <div>
                    <h3 className="text-sm font-semibold text-slate-300 mb-3">Detected Objects</h3>
                    <div className="space-y-2 max-h-96 overflow-y-auto">
                      {Object.entries(detectionResults.counts || {})
                        .sort((a, b) => b[1] - a[1])
                        .map(([class_name, count]) => (
                          <motion.div
                            key={class_name}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="flex items-center justify-between bg-slate-700 rounded-lg p-3 border-l-4 border-purple-500"
                          >
                            <div className="flex items-center gap-3">
                              <span className="text-2xl">{getIconForClass(class_name)}</span>
                              <span className="text-slate-200 font-medium">
                                {formatClassName(class_name)}
                              </span>
                            </div>
                            <span className="text-purple-400 font-bold text-lg">{count}</span>
                          </motion.div>
                        ))}
                      {Object.keys(detectionResults.counts || {}).length === 0 && (
                        <div className="text-center py-8 text-slate-500">
                          <div className="text-4xl mb-2">🔍</div>
                          <p>No objects detected</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 text-slate-500">
                  <div className="text-5xl mb-4">📊</div>
                  <p>Upload an image and click "Detect Objects" to see results</p>
                </div>
              )}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}

export default UploadPicture;

