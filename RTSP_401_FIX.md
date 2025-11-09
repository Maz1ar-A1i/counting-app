# Fixing RTSP 401 Authentication Error

## Problem
Getting "401 authentication" error even though username and password are correct.

## Solutions Implemented

### 1. ✅ URL Encoding (Automatic)
The code now automatically URL-encodes credentials in RTSP URLs to handle special characters.

### 2. ✅ Multiple Connection Methods
The code tries multiple methods:
- TCP transport (more reliable)
- UDP transport (some cameras prefer this)
- Multiple retries for authentication

### 3. ✅ Extended Timeout
- Frontend timeout increased to 90 seconds for camera operations
- Backend allows up to 30 seconds for RTSP connection

---

## Manual Fixes

### Option 1: URL Encode Credentials Manually

If your password contains special characters, try URL-encoding them:

**Python script:**
```python
from urllib.parse import quote

username = "admin"
password = "dl2025d1"  # Your password

encoded_username = quote(username, safe='')
encoded_password = quote(password, safe='')

rtsp_url = f"rtsp://{encoded_username}:{encoded_password}@192.168.1.111:554/cam/realmonitor?channel=1&subtype=0"
```

### Option 2: Try Alternative RTSP URL Formats

Different camera brands use different RTSP URL formats:

**Dahua (your camera):**
```
rtsp://admin:dl2025d1@192.168.1.111:554/cam/realmonitor?channel=1&subtype=0
```

**Alternative formats to try:**
```
# Format 1: Main stream
rtsp://admin:dl2025d1@192.168.1.111:554/cam/realmonitor?channel=1&subtype=0

# Format 2: Sub stream
rtsp://admin:dl2025d1@192.168.1.111:554/cam/realmonitor?channel=1&subtype=1

# Format 3: Without query parameters
rtsp://admin:dl2025d1@192.168.1.111:554/cam/realmonitor

# Format 4: With TCP transport
rtsp://admin:dl2025d1@192.168.1.111:554/cam/realmonitor?channel=1&subtype=0&tcp=1
```

### Option 3: Check Camera Settings

1. **Verify RTSP is enabled:**
   - Login to camera web interface
   - Go to Network → RTSP settings
   - Ensure RTSP is enabled

2. **Check authentication method:**
   - Some cameras support Digest authentication
   - Some support Basic authentication
   - Try both if available

3. **Verify user permissions:**
   - Make sure the user has RTSP streaming permissions
   - Some cameras have separate permissions for streaming

4. **Check IP whitelist:**
   - Some cameras have IP whitelist/blacklist
   - Make sure your computer's IP is allowed

### Option 4: Use VLC to Test

Test your RTSP URL in VLC Media Player first:

1. Open VLC
2. Media → Open Network Stream
3. Paste your RTSP URL
4. If VLC works, the URL is correct
5. If VLC shows authentication error, check credentials

---

## Common Issues

### Issue 1: Password with Special Characters
**Solution:** URL encode the password
```python
from urllib.parse import quote
password = quote("your#password@123", safe='')
```

### Issue 2: Camera Requires Digest Authentication
**Solution:** OpenCV may not support Digest auth. Try:
- Using a different RTSP client
- Checking if camera supports Basic auth
- Using RTSP proxy/server

### Issue 3: Camera Blocks Multiple Connections
**Solution:** 
- Close other applications using the camera
- Check camera's connection limit settings
- Wait a few seconds between connection attempts

### Issue 4: Network/Firewall Issues
**Solution:**
```powershell
# Test network connectivity
ping 192.168.1.111

# Test RTSP port
telnet 192.168.1.111 554
```

---

## Debugging Steps

1. **Run diagnostic tool:**
   ```powershell
   python diagnose_camera.py
   ```

2. **Check backend console:**
   - Look for detailed error messages
   - Check which connection method was tried
   - See authentication attempts

3. **Test with Python script:**
   ```python
   import cv2
   rtsp_url = "rtsp://admin:dl2025d1@192.168.1.111:554/cam/realmonitor?channel=1&subtype=0"
   cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
   print(f"Opened: {cap.isOpened()}")
   ret, frame = cap.read()
   print(f"Frame read: {ret}")
   ```

4. **Check camera logs:**
   - Login to camera web interface
   - Check system logs for authentication attempts
   - See if your IP is being blocked

---

## What the Code Does Now

1. **Automatic URL encoding** - Handles special characters in credentials
2. **Multiple retry attempts** - Tries connecting 5 times with delays
3. **TCP transport** - Uses TCP for more reliable connection
4. **Extended timeouts** - Allows up to 30 seconds for connection
5. **Better error messages** - Shows exactly what failed

---

## Still Having Issues?

If you're still getting 401 errors:

1. **Verify credentials:**
   - Test username/password in camera web interface
   - Make sure no typos
   - Check if password was changed

2. **Try different RTSP URL format:**
   - Check camera documentation
   - Try alternative formats listed above

3. **Check camera firmware:**
   - Update camera firmware if outdated
   - Some older firmware has RTSP bugs

4. **Contact camera manufacturer:**
   - They may have specific RTSP URL requirements
   - They may need to enable RTSP streaming

---

## Test Your Fix

After trying fixes, test with:
```powershell
python test_rtsp.py
```

Or use the diagnostic tool:
```powershell
python diagnose_camera.py
```

---

**Last Updated:** 2024

