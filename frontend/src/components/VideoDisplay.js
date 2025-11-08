import React, { useRef, useEffect, useState } from 'react';
import './VideoDisplay.css';

function VideoDisplay({ videoFrame, onLineDraw, lineMode }) {
  const canvasRef = useRef(null);
  const [drawing, setDrawing] = useState(false);
  const [lineStart, setLineStart] = useState(null);
  const [lineEnd, setLineEnd] = useState(null);
  const [tempLineEnd, setTempLineEnd] = useState(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw video frame
    if (videoFrame) {
      const img = new Image();
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        
        // Draw line if in drawing mode
        if (lineMode && lineStart) {
          ctx.strokeStyle = '#00ffff';
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.moveTo(lineStart.x, lineStart.y);
          const end = tempLineEnd || lineEnd || lineStart;
          ctx.lineTo(end.x, end.y);
          ctx.stroke();
          
          // Draw endpoints
          ctx.fillStyle = '#00ffff';
          ctx.beginPath();
          ctx.arc(lineStart.x, lineStart.y, 5, 0, 2 * Math.PI);
          ctx.fill();
          if (end !== lineStart) {
            ctx.beginPath();
            ctx.arc(end.x, end.y, 5, 0, 2 * Math.PI);
            ctx.fill();
          }
        }
      };
      img.src = `data:image/jpeg;base64,${videoFrame}`;
    }
  }, [videoFrame, lineStart, lineEnd, tempLineEnd, lineMode]);

  const handleMouseDown = (e) => {
    if (!lineMode) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    if (!drawing) {
      setLineStart({ x, y });
      setDrawing(true);
      setLineEnd(null);
      setTempLineEnd(null);
    } else {
      // Second click - finish line
      setLineEnd({ x, y });
      setDrawing(false);
      if (lineStart && onLineDraw) {
        onLineDraw([lineStart.x, lineStart.y], [x, y]);
      }
      setLineStart(null);
      setLineEnd(null);
    }
  };

  const handleMouseMove = (e) => {
    if (!lineMode || !drawing || !lineStart) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setTempLineEnd({ x, y });
  };

  const handleMouseUp = (e) => {
    if (!lineMode || !drawing) return;
    
    // Finish line on mouse release (drag mode)
    if (lineStart && tempLineEnd && onLineDraw) {
      setLineEnd(tempLineEnd);
      onLineDraw([lineStart.x, lineStart.y], [tempLineEnd.x, tempLineEnd.y]);
      setDrawing(false);
      setLineStart(null);
      setLineEnd(null);
      setTempLineEnd(null);
    }
  };

  return (
    <div className="video-display">
      <div className="video-container">
        {videoFrame ? (
          <canvas
            ref={canvasRef}
            className="video-canvas"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            style={{ cursor: lineMode ? 'crosshair' : 'default' }}
          />
        ) : (
          <div className="video-placeholder">
            <p>Camera not started</p>
            <p className="hint">Click "Start Camera" to begin</p>
          </div>
        )}
        {lineMode && (
          <div className="drawing-hint">
            {drawing ? 'Click again or release to finish line' : 'Click to start drawing line'}
          </div>
        )}
      </div>
    </div>
  );
}

export default VideoDisplay;

