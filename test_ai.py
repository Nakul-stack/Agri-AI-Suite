"""
Simple AI Test Script
Tests if the AI model is working correctly
"""

import requests
import json

def test_ai():
    print("="*60)
    print("Testing Crop Recommendation AI")
    print("="*60)
    
    # Get IP addresses
    import subprocess
    result = subprocess.run(['ipconfig'], capture_output=True, text=True, shell=True)
    ip_lines = [line for line in result.stdout.split('\n') if 'IPv4' in line]
    
    print("\nYour IP Addresses:")
    for line in ip_lines:
        if 'IPv4' in line:
            ip = line.split(':')[-1].strip()
            print(f"  - http://{ip}:5000")
    
    print("\n" + "="*60)
    print("Testing AI Model...")
    print("="*60)
    
    # Test health endpoint
    try:
        response = requests.get('http://localhost:5000/health', timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("\n✓ Server is running!")
            print(f"  Model: {health_data.get('model_name', 'N/A')}")
            print(f"  Accuracy: {health_data.get('accuracy', 0):.2%}")
        else:
            print(f"\n✗ Server returned status: {response.status_code}")
            return
    except Exception as e:
        print(f"\n✗ Cannot connect to server: {e}")
        print("\nMake sure the server is running:")
        print("  python server.py")
        return
    
    # Test prediction
    print("\n" + "="*60)
    print("Testing Prediction...")
    print("="*60)
    
    test_data = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.87,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9
    }
    
    try:
        response = requests.post(
            'http://localhost:5000/api/crop/recommend',
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✓ AI Prediction Successful!")
            print(f"\nRecommended Crop: {result['recommended_crop']}")
            print(f"Confidence: {result['confidence_score']:.2%}")
            print("\nTop Recommendations:")
            for i, rec in enumerate(result['top_recommendations'], 1):
                print(f"  {i}. {rec['crop']}: {rec['confidence_percentage']}")
            print("\n" + "="*60)
            print("✓ AI is working correctly!")
            print("="*60)
            print("\nAccess the web interface at:")
            for line in ip_lines:
                if 'IPv4' in line:
                    ip = line.split(':')[-1].strip()
                    print(f"  http://{ip}:5000")
        else:
            print(f"\n✗ Prediction failed: {response.status_code}")
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"\n✗ Prediction error: {e}")

if __name__ == "__main__":
    test_ai()

