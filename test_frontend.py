"""
Quick test to verify server is serving frontend correctly
"""

import requests
import sys

def test_server(url_base):
    print(f"Testing server at {url_base}")
    print("="*60)
    
    # Test root endpoint (should serve HTML)
    try:
        response = requests.get(f"{url_base}/")
        print(f"GET / - Status: {response.status_code}")
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' in content_type:
                print("OK - Frontend HTML is being served correctly")
            else:
                print(f"Warning - Unexpected content type: {content_type}")
        else:
            print(f"FAILED - Failed to get frontend: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
    except Exception as e:
        print(f"ERROR - Error accessing frontend: {e}")
        return False
    
    # Test static files
    try:
        response = requests.get(f"{url_base}/static/css/style.css")
        print(f"GET /static/css/style.css - Status: {response.status_code}")
        if response.status_code == 200:
            print("OK - CSS file is being served correctly")
        else:
            print(f"FAILED - Failed to get CSS: {response.status_code}")
    except Exception as e:
        print(f"ERROR - Error accessing CSS: {e}")
    
    # Test API
    try:
        response = requests.get(f"{url_base}/health")
        print(f"GET /health - Status: {response.status_code}")
        if response.status_code == 200:
            print("OK - API is working correctly")
        else:
            print(f"FAILED - API health check failed: {response.status_code}")
    except Exception as e:
        print(f"ERROR - Error accessing API: {e}")
    
    print("="*60)
    return True

if __name__ == "__main__":
    # Test localhost
    test_server("http://localhost:5000")
    
    print("\nIf tests fail, make sure you're running:")
    print("  python server.py")
    print("\nNOT:")
    print("  python api/app.py")
