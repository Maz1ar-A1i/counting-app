import React, { useRef, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Video, Crosshair } from 'lucide-react';

function VideoFeed({ videoFrame, onLineDraw, lineMode, cameraRunning }) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [drawing, setDrawing] = useState(false);
  const [lineStart, setLineStart] = useState(null);
  const [lineEnd, setLineEnd] = useState(null);
  const [tempLineEnd, setTempLineEnd] = useState(null);
  const [frameInfo, setFrameInfo] = useState({ width: 640, height: 360, scale: 1, offsetX: 0, offsetY: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (videoFrame) {
      const img = new Image();
      img.onload = () => {
        const container = containerRef.current;
        if (!container) return;
        
        const containerWidth = container.clientWidth - 32;
        const containerHeight = 400; // Fixed height for video area
        
        const scale = Math.min(
          containerWidth / img.width,
          containerHeight / img.height,
          1.0
        );
        
        const newWidth = Math.floor(img.width * scale);
        const newHeight = Math.floor(img.height * scale);
        
        canvas.width = newWidth;
        canvas.height = newHeight;
        
        const offsetX = (containerWidth - newWidth) / 2;
        const offsetY = (containerHeight - newHeight) / 2;
        
        setFrameInfo({
          width: img.width,
          height: img.height,
          scale: scale,
          offsetX: offsetX,
          offsetY: offsetY,
          displayWidth: newWidth,
          displayHeight: newHeight
        });
        
        ctx.drawImage(img, 0, 0, newWidth, newHeight);
        
        // Draw line if exists
        if (lineStart) {
          ctx.strokeStyle = '#00ffff';
          ctx.lineWidth = 3;
          ctx.setLineDash([]);
          ctx.beginPath();
          ctx.moveTo(lineStart.x, lineStart.y);
          const end = tempLineEnd || lineEnd || lineStart;
          ctx.lineTo(end.x, end.y);
          ctx.stroke();
          
          ctx.fillStyle = '#00ffff';
          ctx.beginPath();
          ctx.arc(lineStart.x, lineStart.y, 6, 0, 2 * Math.PI);
          ctx.fill();
          if (end !== lineStart) {
            ctx.beginPath();
            ctx.arc(end.x, end.y, 6, 0, 2 * Math.PI);
            ctx.fill();
          }
        }
      };
      img.src = `data:image/jpeg;base64,${videoFrame}`;
    }
  }, [videoFrame, lineStart, lineEnd, tempLineEnd]);

  const handleMouseDown = (e) => {
    if (!lineMode || !cameraRunning) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    if (x < 0 || y < 0 || x > canvas.width || y > canvas.height) return;
    
    if (!drawing) {
      setLineStart({ x, y });
      setDrawing(true);
      setLineEnd(null);
      setTempLineEnd(null);
    } else {
      setLineEnd({ x, y });
      finishLine(x, y);
    }
  };

  const handleMouseMove = (e) => {
    if (!lineMode || !drawing || !lineStart) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(canvas.width, e.clientX - rect.left));
    const y = Math.max(0, Math.min(canvas.height, e.clientY - rect.top));
    setTempLineEnd({ x, y });
  };

  const handleMouseUp = (e) => {
    if (!lineMode || !drawing || !lineStart) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(canvas.width, e.clientX - rect.left));
    const y = Math.max(0, Math.min(canvas.height, e.clientY - rect.top));
    
    if (tempLineEnd) {
      finishLine(tempLineEnd.x, tempLineEnd.y);
    }
  };

  const finishLine = (endX, endY) => {
    if (!lineStart || !onLineDraw) {
      setDrawing(false);
      setLineStart(null);
      setLineEnd(null);
      setTempLineEnd(null);
      return;
    }
    
    const { width: frameWidth, height: frameHeight, displayWidth, displayHeight } = frameInfo;
    
    if (displayWidth > 0 && displayHeight > 0) {
      const scaleX = frameWidth / displayWidth;
      const scaleY = frameHeight / displayHeight;
      
      const startX = Math.max(0, Math.min(frameWidth - 1, Math.floor(lineStart.x * scaleX)));
      const startY = Math.max(0, Math.min(frameHeight - 1, Math.floor(lineStart.y * scaleY)));
      const endX_scaled = Math.max(0, Math.min(frameWidth - 1, Math.floor(endX * scaleX)));
      const endY_scaled = Math.max(0, Math.min(frameHeight - 1, Math.floor(endY * scaleY)));
      
      onLineDraw([startX, startY], [endX_scaled, endY_scaled]);
    }
    
    setDrawing(false);
    setLineStart(null);
    setLineEnd(null);
    setTempLineEnd(null);
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="bg-slate-800 rounded-xl p-4 shadow-2xl border border-slate-700"
    >
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold text-slate-200 flex items-center gap-2">
          <Video className="w-5 h-5 text-blue-400" />
          Live Camera Feed
        </h2>
        {lineMode && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-center gap-2 text-cyan-400 text-sm"
          >
            <Crosshair className="w-4 h-4" />
            <span>Line Drawing Mode</span>
          </motion.div>
        )}
      </div>
      
      <div 
        ref={containerRef}
        className="relative bg-black rounded-lg overflow-hidden flex items-center justify-center"
        style={{ height: '400px' }}
      >
        {videoFrame ? (
          <canvas
            ref={canvasRef}
            className="cursor-crosshair"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            style={{ 
              cursor: lineMode ? 'crosshair' : 'default',
              display: 'block',
              maxWidth: '100%',
              maxHeight: '100%'
            }}
          />
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-slate-500">
            <Video className="w-16 h-16 mb-4 opacity-50" />
            <p className="text-lg font-medium">Camera Not Started</p>
            <p className="text-sm mt-2">Click "Connect Camera" to begin</p>
          </div>
        )}
        
        {lineMode && drawing && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-cyan-500/90 text-white px-4 py-2 rounded-full text-sm font-medium shadow-lg z-10"
          >
            Click again or release to finish line
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}

export default VideoFeed;
