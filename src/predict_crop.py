import pandas as pd
import numpy as np
import joblib
import json
import os
from typing import Dict, List, Tuple

class CropRecommendationPredictor:
    """
    Crop Recommendation Predictor with confidence scores and explainability
    """
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.metadata = None
        self.label_encoder = None
        self.feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model, scaler, and metadata"""
        model_path = os.path.join(self.model_dir, 'crop_recommendation_model.pkl')
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        metadata_path = os.path.join(self.model_dir, 'model_metadata.json')
        encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Load label encoder if it exists (for enhanced model)
        if os.path.exists(encoder_path):
            self.label_encoder = joblib.load(encoder_path)
        else:
            self.label_encoder = None
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Model loaded: {self.metadata['model_name']}")
        print(f"Model Accuracy: {self.metadata['accuracy']:.4f}")
        if 'num_classes' in self.metadata:
            print(f"Number of crop types: {self.metadata['num_classes']}")
    
    def predict(self, N: float, P: float, K: float, 
                temperature: float, humidity: float, ph: float, rainfall: float,
                top_n: int = 3) -> Dict:
        """
        Predict crop recommendation with confidence scores
        
        Args:
            N, P, K: Soil nutrient levels (0-140)
            temperature: Temperature in Celsius
            humidity: Humidity percentage
            ph: Soil pH level
            rainfall: Rainfall in mm
            top_n: Number of top recommendations to return
        
        Returns:
            Dictionary with predictions, confidence scores, and explanations
        """
        # Prepare input data as DataFrame with feature names
        input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], 
                               columns=self.feature_names)
        
        # Scale if needed
        if self.metadata['needs_scaling']:
            input_data_scaled = self.scaler.transform(input_df)
            input_data = pd.DataFrame(input_data_scaled, columns=self.feature_names)
        else:
            input_data = input_df
        
        # Get predictions
        prediction_encoded = self.model.predict(input_data)[0]
        
        # Decode prediction if label encoder exists
        if self.label_encoder is not None:
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
        else:
            prediction = prediction_encoded
        
        # Get probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(input_data)[0]
            if self.label_encoder is not None:
                classes = self.label_encoder.classes_
            else:
                classes = self.model.classes_
            
            # Get top N predictions
            top_indices = np.argsort(probabilities)[::-1][:top_n]
            top_recommendations = []
            
            for idx in top_indices:
                top_recommendations.append({
                    'crop': classes[idx],
                    'confidence': float(probabilities[idx]),
                    'confidence_percentage': f"{probabilities[idx] * 100:.2f}%"
                })
        else:
            top_recommendations = [{
                'crop': prediction,
                'confidence': 1.0,
                'confidence_percentage': '100.00%'
            }]
        
        # Get feature importance
        feature_importance = self.get_feature_importance()
        
        # Generate explanation (use first row of DataFrame as dict)
        explanation = self.generate_explanation(input_data.iloc[0].to_dict(), prediction, feature_importance)
        
        result = {
            'recommended_crop': prediction,
            'top_recommendations': top_recommendations,
            'confidence_score': float(top_recommendations[0]['confidence']),
            'feature_importance': feature_importance,
            'explanation': explanation,
            'input_parameters': {
                'N': N,
                'P': P,
                'K': K,
                'temperature': temperature,
                'humidity': humidity,
                'ph': ph,
                'rainfall': rainfall
            }
        }
        
        return result
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from the model"""
        importance_path = os.path.join(self.model_dir, 'feature_importance.json')
        
        if os.path.exists(importance_path):
            with open(importance_path, 'r') as f:
                return json.load(f)
        
        # Fallback: If feature importance not available, return equal weights
        return {feature: 1.0 / len(self.feature_names) for feature in self.feature_names}
    
    def generate_explanation(self, input_features, prediction: str, 
                            feature_importance: Dict) -> Dict:
        """Generate explanation for the prediction"""
        # Convert input to dictionary (handle both array and dict)
        if isinstance(input_features, (pd.Series, dict)):
            input_dict = dict(input_features) if isinstance(input_features, pd.Series) else input_features
        else:
            input_dict = dict(zip(self.feature_names, input_features))
        
        # Get feature importance sorted
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Generate insights
        insights = []
        
        # Check soil nutrients
        avg_nutrient = (input_dict['N'] + input_dict['P'] + input_dict['K']) / 3
        if avg_nutrient < 40:
            insights.append("Soil nutrient levels are low. Consider adding fertilizers.")
        elif avg_nutrient > 100:
            insights.append("Soil nutrient levels are high. Good for nutrient-demanding crops.")
        
        # Check pH
        if input_dict['ph'] < 6.0:
            insights.append("Soil pH is acidic. Some crops prefer neutral to slightly alkaline soil.")
        elif input_dict['ph'] > 7.5:
            insights.append("Soil pH is alkaline. Consider crops that thrive in alkaline conditions.")
        
        # Check temperature
        if input_dict['temperature'] < 15:
            insights.append("Temperature is low. Suitable for cold-weather crops.")
        elif input_dict['temperature'] > 30:
            insights.append("Temperature is high. Suitable for heat-tolerant crops.")
        
        # Check rainfall
        if input_dict['rainfall'] < 50:
            insights.append("Low rainfall area. Consider drought-resistant crops or irrigation.")
        elif input_dict['rainfall'] > 200:
            insights.append("High rainfall area. Suitable for water-loving crops.")
        
        explanation = {
            'prediction': prediction,
            'top_influencing_factors': [
                {'feature': feat, 'importance': f"{imp:.4f}"} 
                for feat, imp in sorted_features[:3]
            ],
            'insights': insights,
            'model_confidence': f"{self.metadata['accuracy'] * 100:.2f}%"
        }
        
        return explanation
    
    def get_crop_info(self, crop_name: str) -> Dict:
        """Get general information about a crop"""
        # This can be extended with a database of crop information
        crop_info = {
            'name': crop_name,
            'growing_season': 'Varies by region',
            'water_requirements': 'Moderate',
            'soil_requirements': 'Well-drained',
            'temperature_range': 'Varies by crop'
        }
        
        return crop_info

if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("Crop Recommendation Predictor")
    print("="*60)
    
    try:
        predictor = CropRecommendationPredictor()
        
        # Example prediction
        result = predictor.predict(
            N=90,
            P=42,
            K=43,
            temperature=20.87,
            humidity=82.0,
            ph=6.5,
            rainfall=202.9
        )
        
        print("\nPrediction Result:")
        print(f"Recommended Crop: {result['recommended_crop']}")
        print(f"Confidence Score: {result['confidence_score']:.4f}")
        
        print("\nTop Recommendations:")
        for rec in result['top_recommendations']:
            print(f"  - {rec['crop']}: {rec['confidence_percentage']}")
        
        print("\nFeature Importance:")
        for feature, importance in result['feature_importance'].items():
            print(f"  - {feature}: {importance:.4f}")
        
        print("\nExplanation:")
        print(f"  Prediction: {result['explanation']['prediction']}")
        print(f"  Model Confidence: {result['explanation']['model_confidence']}")
        print("\n  Top Influencing Factors:")
        for factor in result['explanation']['top_influencing_factors']:
            print(f"    - {factor['feature']}: {factor['importance']}")
        
        print("\n  Insights:")
        for insight in result['explanation']['insights']:
            print(f"    - {insight}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using train_crop_model.py")

