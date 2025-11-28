"""
Example usage script for Crop Recommendation System
"""

from src.predict_crop import CropRecommendationPredictor

def main():
    print("="*60)
    print("Crop Recommendation System - Example Usage")
    print("="*60)
    
    try:
        # Initialize predictor
        print("\n1. Loading model...")
        predictor = CropRecommendationPredictor()
        
        # Example 1: Rice conditions
        print("\n2. Example Prediction 1 - Rice conditions:")
        print("   Input: N=90, P=42, K=43, temp=20.87°C, humidity=82%, ph=6.5, rainfall=202.9mm")
        result1 = predictor.predict(
            N=90, P=42, K=43,
            temperature=20.87, humidity=82.0,
            ph=6.5, rainfall=202.9
        )
        print(f"   Recommended: {result1['recommended_crop']}")
        print(f"   Confidence: {result1['confidence_score']:.2%}")
        
        # Example 2: Mango conditions
        print("\n3. Example Prediction 2 - Mango conditions:")
        print("   Input: N=120, P=60, K=80, temp=28°C, humidity=75%, ph=6.8, rainfall=150mm")
        result2 = predictor.predict(
            N=120, P=60, K=80,
            temperature=28.0, humidity=75.0,
            ph=6.8, rainfall=150.0
        )
        print(f"   Recommended: {result2['recommended_crop']}")
        print(f"   Confidence: {result2['confidence_score']:.2%}")
        
        # Example 3: Coffee conditions
        print("\n4. Example Prediction 3 - Coffee conditions:")
        print("   Input: N=85, P=35, K=40, temp=22°C, humidity=85%, ph=6.0, rainfall=220mm")
        result3 = predictor.predict(
            N=85, P=35, K=40,
            temperature=22.0, humidity=85.0,
            ph=6.0, rainfall=220.0
        )
        print(f"   Recommended: {result3['recommended_crop']}")
        print(f"   Confidence: {result3['confidence_score']:.2%}")
        
        # Display top recommendations
        print("\n5. Top 3 Recommendations for Example 1:")
        for i, rec in enumerate(result1['top_recommendations'], 1):
            print(f"   {i}. {rec['crop']}: {rec['confidence_percentage']}")
        
        # Display feature importance
        print("\n6. Feature Importance:")
        sorted_features = sorted(
            result1['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for feature, importance in sorted_features:
            print(f"   {feature}: {importance:.4f}")
        
        print("\n" + "="*60)
        print("Examples completed successfully!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease train the model first:")
        print("  python src/train_crop_model.py")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

