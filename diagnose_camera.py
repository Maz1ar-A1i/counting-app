"""
Camera Connection Diagnostic Tool
Helps identify why camera connection is failing.
"""

import cv2
import sys
import time

def test_rtsp_connection(rtsp_url):
    """Test RTSP connection with detailed diagnostics."""
    print("=" * 60)
    print("RTSP Camera Diagnostic Tool")
    print("=" * 60)
    print(f"\nTesting RTSP URL:")
    print(f"  {rtsp_url[:60]}...")
    print("\n" + "=" * 60)
    
    # Step 1: Test basic connectivity
    print("\n[Step 1] Testing network connectivity...")
    import socket
    from urllib.parse import urlparse
    
    try:
        parsed = urlparse(rtsp_url)
        host = parsed.hostname
        port = parsed.port or 554
        
        print(f"  Host: {host}")
        print(f"  Port: {port}")
        
        # Test TCP connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"  ✅ TCP connection to {host}:{port} successful")
        else:
            print(f"  ❌ TCP connection to {host}:{port} failed (Error: {result})")
            print(f"     Possible reasons:")
            print(f"     - Camera is offline")
            print(f"     - Firewall blocking port {port}")
            print(f"     - Incorrect IP address")
            return False
    except Exception as e:
        print(f"  ❌ Network test failed: {e}")
        return False
    
    # Step 2: Test OpenCV connection
    print("\n[Step 2] Testing OpenCV VideoCapture...")
    try:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("  ❌ VideoCapture failed to open stream")
            print("     Possible reasons:")
            print("     - Incorrect RTSP URL format")
            print("     - Wrong username/password")
            print("     - Camera does not support this stream format")
            print("     - Camera requires authentication")
            return False
        
        print("  ✅ VideoCapture opened successfully")
        
        # Step 3: Test frame reading
        print("\n[Step 3] Testing frame reading...")
        print("  Attempting to read frame (this may take 10-30 seconds)...")
        
        start_time = time.time()
        timeout = 30  # 30 second timeout
        
        frame_read = False
        while time.time() - start_time < timeout:
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"  ✅ Frame received! Size: {frame.shape[1]}x{frame.shape[0]}")
                frame_read = True
                break
            time.sleep(0.5)
        
        if not frame_read:
            print("  ❌ Failed to read frame within 30 seconds")
            print("     Possible reasons:")
            print("     - Stream is too slow or buffering")
            print("     - Camera encoding issue")
            print("     - Network bandwidth too low")
            cap.release()
            return False
        
        # Step 4: Test continuous reading
        print("\n[Step 4] Testing continuous frame reading (5 frames)...")
        success_count = 0
        for i in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                success_count += 1
                print(f"  Frame {i+1}/5: ✅")
            else:
                print(f"  Frame {i+1}/5: ❌")
            time.sleep(0.5)
        
        cap.release()
        
        if success_count == 5:
            print("\n" + "=" * 60)
            print("🎉 SUCCESS! RTSP stream is working perfectly!")
            print("=" * 60)
            print("\nThe camera should work in your application.")
            print("If you're still getting 500 errors, check:")
            print("  1. Backend server console for detailed error messages")
            print("  2. Make sure backend server is running")
            print("  3. Check if YOLO model is loaded (first run downloads it)")
            return True
        else:
            print(f"\n⚠️  PARTIAL SUCCESS: {success_count}/5 frames received")
            print("   Stream may be unstable but should work.")
            return True
            
    except Exception as e:
        print(f"  ❌ Error during testing: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    # Your RTSP URL
    RTSP_URL = "rtsp://admin:dl2025d1@192.168.1.111:554/cam/realmonitor?channel=1&subtype=0"
    
    print("\nStarting diagnostic...")
    print("This will test your RTSP camera connection step by step.\n")
    
    success = test_rtsp_connection(RTSP_URL)
    
    if not success:
        print("\n" + "=" * 60)
        print("❌ DIAGNOSIS: RTSP connection failed")
        print("=" * 60)
        print("\nRecommended actions:")
        print("  1. Verify camera is online: ping 192.168.1.111")
        print("  2. Test RTSP URL in VLC Media Player")
        print("  3. Check camera settings for RTSP enablement")
        print("  4. Verify username/password are correct")
        print("  5. Check firewall settings for port 554")
        print("\n" + "=" * 60)
        sys.exit(1)
    else:
        sys.exit(0)

