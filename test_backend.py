# test_backend.py
"""
Simple test script to verify backend API is working.
"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_status():
    """Test status endpoint."""
    print("Testing /api/status...")
    try:
        response = requests.get(f"{BASE_URL}/api/status")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_start_camera():
    """Test camera start endpoint."""
    print("\nTesting /api/camera/start...")
    try:
        response = requests.post(
            f"{BASE_URL}/api/camera/start",
            json={"source": 0},
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200 and response.json().get('success', False)
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_stop_camera():
    """Test camera stop endpoint."""
    print("\nTesting /api/camera/stop...")
    try:
        response = requests.post(f"{BASE_URL}/api/camera/stop")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Backend API Test")
    print("="*60)
    
    # Test status
    if not test_status():
        print("\n[ERROR] Status test failed. Is the server running?")
        print("Start server with: python backend/server.py")
        exit(1)
    
    # Test camera start
    if test_start_camera():
        print("\n[OK] Camera started successfully")
        time.sleep(2)
        test_stop_camera()
    else:
        print("\n[ERROR] Camera start failed. Check console for errors.")
    
    print("\n" + "="*60)
    print("Test completed")

