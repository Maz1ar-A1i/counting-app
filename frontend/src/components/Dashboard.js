import React, { useEffect, useState } from 'react';
import './Dashboard.css';

function Dashboard({ stats }) {
  const { fps, counts, total_count } = stats;
  const [displayFPS, setDisplayFPS] = useState(0);

  // Smooth FPS display to avoid flickering
  useEffect(() => {
    if (fps > 0) {
      // Cap FPS display at reasonable value (e.g., 60)
      const cappedFPS = Math.min(fps, 60);
      setDisplayFPS(prev => {
        // Smooth transition
        return prev * 0.7 + cappedFPS * 0.3;
      });
    } else {
      setDisplayFPS(0);
    }
  }, [fps]);

  return (
    <div className="dashboard">
      <h2>Object Catalog</h2>
      
      <div className="stat-card total">
        <div className="stat-label">Total Counted</div>
        <div className="stat-value">{total_count || 0}</div>
      </div>

      <div className="stat-card fps">
        <div className="stat-label">FPS</div>
        <div className="stat-value">{displayFPS.toFixed(1)}</div>
      </div>

      <div className="counts-section">
        <h3>Counts by Category</h3>
        <div className="counts-list">
          {Object.keys(counts).length > 0 ? (
            Object.entries(counts)
              .sort((a, b) => b[1] - a[1])
              .map(([class_name, count]) => (
                <div key={class_name} className="count-item">
                  <div className="count-item-left">
                    <span className="count-icon">{getIconForClass(class_name)}</span>
                    <span className="count-class">{formatClassName(class_name)}</span>
                  </div>
                  <span className="count-value">{count}</span>
                </div>
              ))
          ) : (
            <div className="no-data">
              <div className="no-data-icon">📊</div>
              <p>No objects detected yet</p>
              <p className="no-data-hint">Start camera to begin detection</p>
            </div>
          )}
        </div>
      </div>

      <div className="instructions">
        <h3>Quick Guide</h3>
        <ul>
          <li>Click "Start Camera" to begin</li>
          <li>Press <kbd>L</kbd> to toggle line mode</li>
          <li>Draw line on video to count crossings</li>
          <li>Adjust sliders for accuracy</li>
        </ul>
      </div>
    </div>
  );
}

function getIconForClass(class_name) {
  const icons = {
    'person': '👤',
    'car': '🚗',
    'truck': '🚚',
    'bus': '🚌',
    'motorcycle': '🏍️',
    'bicycle': '🚲',
    'dog': '🐕',
    'cat': '🐈',
    'bird': '🐦',
    'laptop': '💻',
    'cell phone': '📱',
    'backpack': '🎒',
    'handbag': '👜',
    'suitcase': '🧳',
    'bottle': '🍼',
    'cup': '☕',
    'chair': '🪑',
    'couch': '🛋️',
    'tv': '📺',
    'keyboard': '⌨️',
    'mouse': '🖱️'
  };
  return icons[class_name] || '📦';
}

function formatClassName(class_name) {
  return class_name
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

export default Dashboard;
