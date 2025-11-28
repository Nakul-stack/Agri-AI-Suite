"""
Quick Start Guide for Crop Recommendation System
Run this script to train the model and test it
"""

import os
import sys

def main():
    print("="*70)
    print("Crop Recommendation System - Quick Start")
    print("="*70)
    
    # Step 1: Check if dependencies are installed
    print("\n[Step 1] Checking dependencies...")
    try:
        import pandas
        import numpy
        import sklearn
        import flask
        print("✓ All dependencies are installed")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        return
    
    # Step 2: Check if dataset exists
    print("\n[Step 2] Checking dataset...")
    dataset_path = "DataSet/Crop_recommendation.csv"
    if os.path.exists(dataset_path):
        print(f"✓ Dataset found: {dataset_path}")
    else:
        print(f"✗ Dataset not found: {dataset_path}")
        return
    
    # Step 3: Train model
    print("\n[Step 3] Training model...")
    print("This may take a few minutes...")
    try:
        from src.train_crop_model import CropRecommendationModel
        trainer = CropRecommendationModel()
        trainer.train_models()
        trainer.save_model()
        print("\n✓ Model training completed successfully!")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Test prediction
    print("\n[Step 4] Testing prediction...")
    try:
        from src.predict_crop import CropRecommendationPredictor
        predictor = CropRecommendationPredictor()
        
        result = predictor.predict(
            N=90, P=42, K=43,
            temperature=20.87, humidity=82.0,
            ph=6.5, rainfall=202.9
        )
        
        print(f"✓ Prediction test successful!")
        print(f"  Recommended Crop: {result['recommended_crop']}")
        print(f"  Confidence: {result['confidence_score']:.2%}")
    except Exception as e:
        print(f"\n✗ Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Instructions
    print("\n" + "="*70)
    print("Setup Complete!")
    print("="*70)
    print("\nNext steps:")
    print("\n1. Test predictions:")
    print("   python example_usage.py")
    print("\n2. Start API server:")
    print("   python api/app.py")
    print("\n3. Test API endpoints:")
    print("   python test_api.py")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()

