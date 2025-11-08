"""
Quick RTSP camera connection test script.
Tests if your RTSP stream is accessible and working.
"""

import cv2
import sys

# Your RTSP URL
RTSP_URL = "rtsp://admin:dl2025d1@192.168.1.111:554/cam/realmonitor?channel=1&subtype=0"

print("=" * 60)
print("RTSP Camera Connection Test")
print("=" * 60)
print(f"\nTesting RTSP URL:")
print(f"{RTSP_URL[:50]}...")  # Show first 50 chars for security
print("\n" + "=" * 60)

# Try to connect
print("\n[1/3] Attempting to connect to RTSP stream...")
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("❌ FAILED: Could not open RTSP stream")
    print("\nPossible reasons:")
    print("  - Incorrect URL or credentials")
    print("  - Camera is offline or unreachable")
    print("  - Network connectivity issues")
    print("  - Firewall blocking port 554")
    print("  - RTSP not enabled on camera")
    sys.exit(1)

print("✅ Connection established!")

# Try to read a frame
print("\n[2/3] Attempting to read frame...")
ret, frame = cap.read()

if not ret or frame is None:
    print("❌ FAILED: Could not read frame from stream")
    print("\nPossible reasons:")
    print("  - Stream format not supported")
    print("  - Camera encoding issue")
    print("  - Network bandwidth too low")
    cap.release()
    sys.exit(1)

print(f"✅ Frame received! Size: {frame.shape[1]}x{frame.shape[0]}")

# Test multiple frames
print("\n[3/3] Testing continuous frame reading (5 frames)...")
success_count = 0
for i in range(5):
    ret, frame = cap.read()
    if ret and frame is not None:
        success_count += 1
        print(f"  Frame {i+1}/5: ✅")
    else:
        print(f"  Frame {i+1}/5: ❌")
    cv2.waitKey(100)  # Small delay

cap.release()

print("\n" + "=" * 60)
if success_count == 5:
    print("🎉 SUCCESS! RTSP stream is working perfectly!")
    print("\nYou can now use this URL in the web interface:")
    print(f"  {RTSP_URL}")
    print("\nSteps:")
    print("  1. Start backend: .\\run_backend.bat")
    print("  2. Open: http://localhost:5000")
    print("  3. Paste RTSP URL in 'Camera Source' field")
    print("  4. Click 'Start Camera'")
else:
    print(f"⚠️  PARTIAL SUCCESS: {success_count}/5 frames received")
    print("   Stream may be unstable. Check network connection.")
print("=" * 60)

