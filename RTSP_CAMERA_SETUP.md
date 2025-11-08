# RTSP Camera Setup Guide

## 📹 How to Connect RTSP Camera

Your RTSP camera URL is already configured in the code. Here's how to use it:

### RTSP URL Format
```
rtsp://username:password@ip_address:port/path
```

### Your Camera URL
```
rtsp://admin:dl2025d1@192.168.1.111:554/cam/realmonitor?channel=1&subtype=0
```

---

## 🚀 Quick Start

### Method 1: Using the Web Interface

1. **Start the backend server:**
   ```powershell
   .\run_backend.bat
   ```

2. **Open the web interface:**
   - Go to: `http://localhost:5000`
   - Click "OPEN CAMERA" button

3. **Enter RTSP URL:**
   - In the "Camera Source" field, paste your RTSP URL:
     ```
     rtsp://admin:dl2025d1@192.168.1.111:554/cam/realmonitor?channel=1&subtype=0
     ```

4. **Click "Start Camera"**

---

### Method 2: Test RTSP Connection First

Test if your RTSP stream is accessible:

```python
import cv2

rtsp_url = "rtsp://admin:dl2025d1@192.168.1.111:554/cam/realmonitor?channel=1&subtype=0"

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if cap.isOpened():
    print("✅ RTSP stream connected!")
    ret, frame = cap.read()
    if ret:
        print(f"✅ Frame received! Size: {frame.shape}")
    else:
        print("❌ Failed to read frame")
    cap.release()
else:
    print("❌ Failed to connect to RTSP stream")
```

Save as `test_rtsp.py` and run:
```powershell
python test_rtsp.py
```

---

## ⚙️ RTSP Configuration

The code is now optimized for RTSP streams with:

- **FFMPEG backend** - Better RTSP support
- **Minimal buffer** - Reduced latency (buffer size = 1)
- **Automatic reconnection** - Handles connection drops
- **Error handling** - Graceful fallback on failures

---

## 🔧 Troubleshooting

### Problem: "Failed to start camera" or "Connection lost"

**Solutions:**

1. **Check network connectivity:**
   ```powershell
   ping 192.168.1.111
   ```

2. **Verify RTSP URL:**
   - Make sure username/password are correct
   - Check IP address and port (554 is default RTSP port)
   - Verify the path is correct for your camera model

3. **Test with VLC Media Player:**
   - Open VLC → Media → Open Network Stream
   - Paste your RTSP URL
   - If VLC can't connect, the URL might be wrong

4. **Check firewall:**
   - Make sure port 554 is not blocked
   - Allow RTSP traffic through firewall

5. **Camera settings:**
   - Ensure RTSP is enabled on the camera
   - Check if camera supports the requested stream format
   - Some cameras require specific channel/subtype parameters

### Problem: High latency or lag

**Solutions:**

1. **Reduce frame size** (already optimized in code: 640x360)
2. **Check network bandwidth** - RTSP streams require stable network
3. **Use wired connection** instead of WiFi if possible

### Problem: Connection drops frequently

**Solutions:**

1. **Check camera settings:**
   - Increase timeout settings on camera
   - Disable power saving mode
   - Check if camera has connection limits

2. **Network stability:**
   - Use stable network connection
   - Check for network interference
   - Consider using a dedicated network for cameras

---

## 📝 RTSP URL Examples

### Different Camera Brands:

**Hikvision:**
```
rtsp://username:password@ip:554/Streaming/Channels/101
```

**Dahua:**
```
rtsp://username:password@ip:554/cam/realmonitor?channel=1&subtype=0
```

**Generic RTSP:**
```
rtsp://username:password@ip:554/stream1
```

---

## 🔒 Security Notes

- **Never commit RTSP URLs with passwords to version control**
- Consider using environment variables for sensitive URLs
- Change default camera passwords
- Use VPN for remote camera access

---

## ✅ Features Enabled for RTSP

- ✅ Automatic RTSP detection
- ✅ FFMPEG backend for better compatibility
- ✅ Minimal latency (buffer size = 1)
- ✅ Automatic reconnection handling
- ✅ Error recovery
- ✅ Frame rate optimization

---

## 📊 Performance

RTSP streams typically achieve:
- **Latency:** 200-500ms (depending on network)
- **FPS:** 15-30 FPS (depending on camera and network)
- **Stability:** Good with stable network connection

---

## 🆘 Need Help?

If you're still having issues:

1. Test RTSP URL with VLC Media Player first
2. Check camera documentation for correct RTSP URL format
3. Verify network connectivity and firewall settings
4. Check backend console for detailed error messages

---

**Last Updated:** 2024

