"""
Quick test script to verify the API endpoints
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    """Test health check endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

def test_model_info():
    """Test model info endpoint"""
    print("Testing /api/model/info endpoint...")
    response = requests.get(f"{BASE_URL}/api/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_recommend():
    """Test crop recommendation endpoint"""
    print("Testing /api/crop/recommend endpoint...")
    
    payload = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.87,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9,
        "top_n": 3
    }
    
    response = requests.post(
        f"{BASE_URL}/api/crop/recommend",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    
    print(f"Recommended Crop: {result['recommended_crop']}")
    print(f"Confidence: {result['confidence_score']:.2%}")
    print("\nTop Recommendations:")
    for rec in result['top_recommendations']:
        print(f"  - {rec['crop']}: {rec['confidence_percentage']}")
    print()

def test_batch_recommend():
    """Test batch recommendation endpoint"""
    print("Testing /api/crop/batch-recommend endpoint...")
    
    payload = {
        "samples": [
            {
                "N": 90,
                "P": 42,
                "K": 43,
                "temperature": 20.87,
                "humidity": 82.0,
                "ph": 6.5,
                "rainfall": 202.9
            },
            {
                "N": 120,
                "P": 60,
                "K": 80,
                "temperature": 28.0,
                "humidity": 75.0,
                "ph": 6.8,
                "rainfall": 150.0
            }
        ],
        "top_n": 3
    }
    
    response = requests.post(
        f"{BASE_URL}/api/crop/batch-recommend",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Total Samples: {result['total_samples']}")
    print(f"Successful Predictions: {result['successful_predictions']}")
    print()

if __name__ == "__main__":
    print("="*60)
    print("API Test Script")
    print("="*60)
    print("\nMake sure the API server is running:")
    print("  python api/app.py\n")
    
    try:
        test_health()
        test_model_info()
        test_recommend()
        test_batch_recommend()
        
        print("="*60)
        print("All tests completed!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API server.")
        print("Please start the server first:")
        print("  python api/app.py")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

