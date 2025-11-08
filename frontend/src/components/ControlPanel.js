import React, { useState } from 'react';
import './ControlPanel.css';

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
    <div className="control-panel">
      <div className="control-section">
        <h3>Camera Controls</h3>
        <div className="control-group">
          <label>Camera Source:</label>
          <input
            type="text"
            value={settings.cameraSource}
            onChange={(e) => onCameraSourceChange(e.target.value)}
            placeholder="0 for webcam, or IP URL"
            disabled={cameraRunning}
          />
        </div>
        <div className="button-group">
          <button
            className="btn btn-success"
            onClick={onStartCamera}
            disabled={cameraRunning}
          >
            Start Camera
          </button>
          <button
            className="btn btn-danger"
            onClick={onStopCamera}
            disabled={!cameraRunning}
          >
            Stop Camera
          </button>
        </div>
      </div>

      <div className="control-section">
        <h3>Detection Settings</h3>
        <div className="control-group">
          <label>Confidence: {localConfidence.toFixed(2)}</label>
          <input
            type="range"
            min="0.1"
            max="0.95"
            step="0.05"
            value={localConfidence}
            onChange={(e) => handleConfidenceChange(e.target.value)}
          />
        </div>
        <div className="control-group">
          <label>NMS IoU: {localIou.toFixed(2)}</label>
          <input
            type="range"
            min="0.1"
            max="0.9"
            step="0.05"
            value={localIou}
            onChange={(e) => handleIouChange(e.target.value)}
          />
        </div>
      </div>

      <div className="control-section">
        <h3>Line Counting</h3>
        <div className="control-group">
          <button
            className={`btn ${lineMode ? 'btn-success' : 'btn-secondary'}`}
            onClick={onToggleLineMode}
          >
            {lineMode ? 'Disable Line Counting' : 'Enable Line Counting'}
          </button>
          <button
            className="btn btn-warning"
            onClick={onResetLine}
            disabled={!lineMode}
          >
            Reset Line
          </button>
        </div>
        <div className="hint">
          {lineMode ? 'Press L or click to draw counting line on video' : 'Enable line counting to draw a counting line'}
        </div>
      </div>
    </div>
  );
}

export default ControlPanel;

