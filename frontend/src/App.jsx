import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Homepage from './components/Homepage';
import LiveCamera from './pages/LiveCamera';
import UploadPicture from './pages/UploadPicture';
import LiveCameraLine from './pages/LiveCameraLine';
import AreaClassification from './pages/AreaClassification';
import SystemStatus from './pages/SystemStatus';
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Homepage />} />
        <Route path="/live-camera" element={<LiveCamera />} />
        <Route path="/upload-picture" element={<UploadPicture />} />
        <Route path="/live-camera-line" element={<LiveCameraLine />} />
        <Route path="/area-classification" element={<AreaClassification />} />
        <Route path="/system-status" element={<SystemStatus />} />
      </Routes>
    </Router>
  );
}

export default App;
