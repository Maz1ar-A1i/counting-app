"""
RTSP Authentication Fix Tool
Helps fix 401 authentication errors with RTSP cameras.
"""

from urllib.parse import quote, urlparse, urlunparse

def fix_rtsp_url(rtsp_url):
    """
    Fix RTSP URL by properly encoding credentials.
    This helps resolve 401 authentication errors.
    """
    try:
        parsed = urlparse(rtsp_url)
        
        print("Original RTSP URL:")
        print(f"  {rtsp_url}")
        print(f"\nParsed components:")
        print(f"  Username: {parsed.username}")
        print(f"  Password: {'*' * len(parsed.password) if parsed.password else 'None'}")
        print(f"  Host: {parsed.hostname}")
        print(f"  Port: {parsed.port}")
        print(f"  Path: {parsed.path}")
        print(f"  Query: {parsed.query}")
        
        if parsed.username:
            # URL encode username and password
            encoded_username = quote(parsed.username, safe='')
            encoded_password = quote(parsed.password or '', safe='')
            
            # Reconstruct URL with encoded credentials
            netloc = f"{encoded_username}:{encoded_password}@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            
            fixed_url = urlunparse((
                parsed.scheme,
                netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment
            ))
            
            print(f"\n✅ Fixed RTSP URL (with URL-encoded credentials):")
            print(f"  {fixed_url}")
            
            return fixed_url
        else:
            print("\n⚠️  No username found in URL. Adding credentials manually:")
            print("   Format: rtsp://username:password@ip:port/path")
            return rtsp_url
            
    except Exception as e:
        print(f"❌ Error processing URL: {e}")
        return rtsp_url

if __name__ == '__main__':
    # Your RTSP URL
    RTSP_URL = "rtsp://admin:dl2025d1@192.168.1.111:554/cam/realmonitor?channel=1&subtype=0"
    
    print("=" * 60)
    print("RTSP Authentication Fix Tool")
    print("=" * 60)
    print()
    
    fixed_url = fix_rtsp_url(RTSP_URL)
    
    print("\n" + "=" * 60)
    print("Usage:")
    print("=" * 60)
    print("1. Try the fixed URL in your application")
    print("2. If still getting 401, check:")
    print("   - Username and password are correct")
    print("   - Camera supports RTSP authentication")
    print("   - Camera IP and port are correct")
    print("   - Camera is not blocking your IP")
    print("=" * 60)

