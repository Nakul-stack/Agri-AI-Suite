"""
Test script for Fertilizer Optimization API
"""

import requests
import json

def test_fertilizer_api():
    print("="*60)
    print("Testing Fertilizer Optimization API")
    print("="*60)
    
    base_url = "http://localhost:5000"
    
    # Test data
    test_data = {
        "N": 37,
        "P": 0,
        "K": 0,
        "temperature": 26,
        "humidity": 52,
        "moisture": 38,
        "soil_type": "Sandy",
        "crop_type": "Maize"
    }
    
    try:
        print("\n1. Testing fertilizer recommendation...")
        response = requests.post(
            f"{base_url}/api/fertilizer/recommend",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("OK - API call successful!")
            print(f"\nRecommended Fertilizer: {result['recommended_fertilizer']}")
            print(f"Confidence: {result['confidence_score']:.2%}")
            
            print("\nFertilizer Information:")
            fert_info = result['recommendations']['fertilizer_info']
            print(f"  Composition: {fert_info.get('composition', 'N/A')}")
            print(f"  Primary Nutrient: {fert_info.get('primary_nutrient', 'N/A')}")
            print(f"  Suitable For: {fert_info.get('suitable_for', 'N/A')}")
            print(f"  Dosage Range: {fert_info.get('dosage_range', 'N/A')}")
            
            print("\nApplication Guidelines:")
            for guideline in result['recommendations']['application_guidelines'][:3]:
                print(f"  - {guideline}")
            
            print("\nSoil Health Notes:")
            for note in result['recommendations']['soil_health_notes'][:2]:
                print(f"  - {note}")
        else:
            print(f"FAILED - API call failed: {response.status_code}")
            print(f"  Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("FAILED - Cannot connect to server.")
        print("Make sure the server is running: python server.py")
    except Exception as e:
        print(f"ERROR - {e}")

if __name__ == "__main__":
    test_fertilizer_api()

