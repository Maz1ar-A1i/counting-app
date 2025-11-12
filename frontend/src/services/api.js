import axios from 'axios';
import io from 'socket.io-client';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

// Create axios instance for REST API
export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000, // 60 second timeout (RTSP connections can take longer)
});

// Create a separate instance for camera operations with longer timeout
export const cameraApi = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 90000, // 90 second timeout for camera start/stop operations
});

// System metrics helper
export const systemApi = {
  getMetrics: () => api.get('/api/system/metrics'),
  getRecommendations: () => api.get('/api/system/recommendations'),
};

// Create Socket.IO instance for WebSocket
export const socket = io(API_BASE_URL, {
  transports: ['websocket', 'polling'],
  reconnection: true,
  reconnectionDelay: 1000,
  reconnectionDelayMax: 5000,
  reconnectionAttempts: 5,
  timeout: 20000,
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`[API] ${config.method.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('[API] Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response) {
      console.error('[API] Response error:', error.response.status, error.response.data);
    } else if (error.request) {
      console.error('[API] No response received:', error.request);
    } else {
      console.error('[API] Error:', error.message);
    }
    return Promise.reject(error);
  }
);

// Socket event logging
socket.on('connect', () => {
  console.log('[WebSocket] Connected to server');
});

socket.on('disconnect', (reason) => {
  console.log('[WebSocket] Disconnected:', reason);
});

socket.on('connect_error', (error) => {
  console.error('[WebSocket] Connection error:', error);
});

export default { api, socket, cameraApi, systemApi };
