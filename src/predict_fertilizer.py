import pandas as pd
import numpy as np
import joblib
import json
import os
from typing import Dict, List

class FertilizerOptimizationPredictor:
    """
    Fertilizer Optimization Predictor
    Predicts optimal fertilizer type based on soil, climate, and crop
    """
    
    def __init__(self, model_dir='models/fertilizer'):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.soil_encoder = None
        self.crop_encoder = None
        self.metadata = None
        
        self.load_models()
    
    def load_models(self):
        """Load fertilizer optimization models"""
        metadata_path = os.path.join(self.model_dir, 'model_metadata.json')
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Fertilizer models not found at {model_dir}. Please train the models first.")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load model
        model_path = os.path.join(self.model_dir, 'fertilizer_model.pkl')
        self.model = joblib.load(model_path)
        
        # Load scaler
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        self.scaler = joblib.load(scaler_path)
        
        # Load encoders
        label_encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')
        self.label_encoder = joblib.load(label_encoder_path)
        
        soil_encoder_path = os.path.join(self.model_dir, 'soil_encoder.pkl')
        self.soil_encoder = joblib.load(soil_encoder_path)
        
        crop_encoder_path = os.path.join(self.model_dir, 'crop_encoder.pkl')
        self.crop_encoder = joblib.load(crop_encoder_path)
        
        print(f"Fertilizer model loaded: {self.metadata['model_name']}")
        print(f"Model Accuracy: {self.metadata['accuracy']:.4f}")
        print(f"Available fertilizers: {len(self.metadata['fertilizer_classes'])}")
    
    def predict(self, N: float, P: float, K: float, 
                temperature: float, humidity: float, moisture: float = None,
                soil_type: str = None, crop_type: str = None,
                ph: float = None) -> Dict:
        """
        Predict fertilizer type
        
        Args:
            N, P, K: Current soil nutrient levels
            temperature: Temperature in Celsius
            humidity: Humidity percentage
            moisture: Soil moisture percentage (optional)
            soil_type: Soil type - Sandy, Loamy, Clayey, Black, Red (optional)
            crop_type: Crop type (optional)
            ph: Soil pH (optional, used to infer soil type if not provided)
        
        Returns:
            Dictionary with fertilizer recommendation
        """
        # Default values
        if moisture is None:
            moisture = 40  # Default moisture
        
        # Infer soil type from pH if not provided
        if soil_type is None:
            if ph is not None:
                if ph < 6.0:
                    soil_type = 'Clayey'
                elif ph < 7.0:
                    soil_type = 'Loamy'
                else:
                    soil_type = 'Sandy'
            else:
                soil_type = 'Loamy'  # Default
        
        # Encode categorical features
        if soil_type in self.soil_encoder.classes_:
            soil_encoded = self.soil_encoder.transform([soil_type])[0]
        else:
            soil_encoded = 0  # Default to first soil type
        
        if crop_type and crop_type in self.crop_encoder.classes_:
            crop_encoded = self.crop_encoder.transform([crop_type])[0]
        else:
            crop_encoded = 0  # Default to first crop
        
        # Prepare input features - include all features from metadata
        feature_names = self.metadata['feature_names']
        
        # Build input row dynamically based on available features
        input_row = []
        for feature in feature_names:
            if feature == 'temperature':
                input_row.append(temperature)
            elif feature == 'humidity':
                input_row.append(humidity)
            elif feature == 'moisture':
                input_row.append(moisture)
            elif feature == 'N':
                input_row.append(N)
            elif feature == 'P':
                input_row.append(P)
            elif feature == 'K':
                input_row.append(K)
            elif feature == 'soil_type_encoded':
                input_row.append(soil_encoded)
            elif feature == 'crop_type_encoded':
                input_row.append(crop_encoded)
            elif feature == 'ph':
                input_row.append(ph if ph is not None else 6.5)
            elif feature == 'rainfall':
                input_row.append(150)  # Default rainfall
            else:
                input_row.append(0)  # Default for unknown features
        
        input_data = pd.DataFrame([input_row], columns=feature_names)
        
        # Predict fertilizer
        if self.metadata['needs_scaling']:
            input_scaled = self.scaler.transform(input_data)
            prediction_encoded = self.model.predict(input_scaled)[0]
            probabilities = self.model.predict_proba(input_scaled)[0] if hasattr(self.model, 'predict_proba') else None
        else:
            prediction_encoded = self.model.predict(input_data)[0]
            probabilities = self.model.predict_proba(input_data)[0] if hasattr(self.model, 'predict_proba') else None
        
        # Decode prediction
        fertilizer_name = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get top recommendations
        top_recommendations = []
        if probabilities is not None:
            top_indices = np.argsort(probabilities)[::-1][:3]
            for idx in top_indices:
                fert_name = self.label_encoder.classes_[idx]
                top_recommendations.append({
                    'fertilizer': fert_name,
                    'confidence': float(probabilities[idx]),
                    'confidence_percentage': f"{probabilities[idx] * 100:.2f}%"
                })
        else:
            top_recommendations.append({
                'fertilizer': fertilizer_name,
                'confidence': 1.0,
                'confidence_percentage': '100.00%'
            })
        
        # Generate detailed recommendations
        recommendations = self.generate_recommendations(
            fertilizer_name, N, P, K, temperature, humidity, soil_type, crop_type
        )
        
        result = {
            'recommended_fertilizer': fertilizer_name,
            'confidence_score': float(top_recommendations[0]['confidence']),
            'top_recommendations': top_recommendations,
            'current_soil': {
                'N': N,
                'P': P,
                'K': K,
                'temperature': temperature,
                'humidity': humidity,
                'moisture': moisture,
                'soil_type': soil_type,
                'crop_type': crop_type if crop_type else 'Not specified',
                'ph': ph if ph else 'Not specified'
            },
            'recommendations': recommendations
        }
        
        return result
    
    def generate_recommendations(self, fertilizer_name: str, N: float, P: float, K: float,
                                temperature: float, humidity: float, soil_type: str, crop_type: str) -> Dict:
        """Generate detailed fertilizer recommendations"""
        recommendations = {
            'fertilizer_info': {},
            'application_guidelines': [],
            'dosage_estimate': {},
            'organic_alternatives': [],
            'soil_health_notes': []
        }
        
        # Fertilizer information based on name
        fertilizer_info = {
            'Urea': {
                'composition': '46% Nitrogen (N)',
                'primary_nutrient': 'Nitrogen',
                'suitable_for': 'Nitrogen-deficient soils',
                'dosage_range': '100-200 kg/ha'
            },
            'DAP': {
                'composition': '18% Nitrogen, 46% Phosphorus (P)',
                'primary_nutrient': 'Nitrogen and Phosphorus',
                'suitable_for': 'N and P deficient soils',
                'dosage_range': '150-250 kg/ha'
            },
            '14-35-14': {
                'composition': '14% N, 35% P, 14% K',
                'primary_nutrient': 'Balanced (High P)',
                'suitable_for': 'Phosphorus-deficient soils',
                'dosage_range': '200-300 kg/ha'
            },
            '10-26-26': {
                'composition': '10% N, 26% P, 26% K',
                'primary_nutrient': 'Phosphorus and Potassium (High P, High K)',
                'suitable_for': 'P and K deficient soils',
                'dosage_range': '200-300 kg/ha'
            },
            'MOP': {
                'composition': '60% Potassium (K)',
                'primary_nutrient': 'Potassium',
                'suitable_for': 'Potassium-deficient soils',
                'dosage_range': '100-150 kg/ha'
            },
            'Potash': {
                'composition': '60% Potassium (K)',
                'primary_nutrient': 'Potassium',
                'suitable_for': 'Potassium-deficient soils',
                'dosage_range': '100-150 kg/ha'
            },
            'NPK': {
                'composition': 'Varies (e.g., 15-15-15, 20-20-20)',
                'primary_nutrient': 'Balanced NPK',
                'suitable_for': 'General balanced nutrient requirements',
                'dosage_range': '150-250 kg/ha'
            },
            '28-28': {
                'composition': '28% N, 28% P',
                'primary_nutrient': 'Nitrogen and Phosphorus',
                'suitable_for': 'N and P deficient soils',
                'dosage_range': '150-250 kg/ha'
            },
            '17-17-17': {
                'composition': '17% N, 17% P, 17% K',
                'primary_nutrient': 'Balanced NPK',
                'suitable_for': 'Balanced nutrient requirements',
                'dosage_range': '200-300 kg/ha'
            },
            '20-20': {
                'composition': '20% N, 20% P',
                'primary_nutrient': 'Nitrogen and Phosphorus',
                'suitable_for': 'N and P deficient soils',
                'dosage_range': '150-250 kg/ha'
            }
        }
        
        recommendations['fertilizer_info'] = fertilizer_info.get(
            fertilizer_name,
            {'composition': 'Unknown', 'primary_nutrient': 'Unknown'}
        )
        
        # Application guidelines
        recommendations['application_guidelines'] = [
            f"Apply {fertilizer_name} before or during planting",
            "Mix thoroughly with soil for best results",
            "Apply in split doses: 50% before planting, 50% during growth",
            "Water after application to ensure proper nutrient absorption",
            "Avoid application during extreme weather conditions"
        ]
        
        # Dosage estimate based on deficiencies
        n_deficit = max(0, 100 - N)  # Assuming 100 is ideal
        p_deficit = max(0, 50 - P)   # Assuming 50 is ideal
        k_deficit = max(0, 150 - K)  # Assuming 150 is ideal
        
        recommendations['dosage_estimate'] = {
            'estimated_kg_per_ha': f"{150-250} kg/ha (adjust based on soil test)",
            'calculation_basis': 'Based on nutrient deficiencies and fertilizer composition'
        }
        
        # Organic alternatives
        if 'N' in fertilizer_name or fertilizer_name == 'Urea':
            recommendations['organic_alternatives'].append({
                'nutrient': 'Nitrogen',
                'sources': ['Compost (1-2% N)', 'Farmyard manure (0.5-1% N)', 'Legume cover crops', 'Green manure']
            })
        if 'P' in fertilizer_name or fertilizer_name in ['DAP', '14-35-14', '28-28', '20-20', 'NPK 10-26-26']:
            recommendations['organic_alternatives'].append({
                'nutrient': 'Phosphorus',
                'sources': ['Bone meal', 'Rock phosphate', 'Compost', 'Fish meal']
            })
        if 'K' in fertilizer_name or fertilizer_name in ['17-17-17', 'NPK 10-26-26']:
            recommendations['organic_alternatives'].append({
                'nutrient': 'Potassium',
                'sources': ['Wood ash', 'Compost', 'Kelp meal', 'Potassium sulfate (organic)']
            })
        
        # Soil health notes
        if N < 40:
            recommendations['soil_health_notes'].append("Nitrogen levels are low. Critical for plant growth and yield.")
        if P < 20:
            recommendations['soil_health_notes'].append("Phosphorus levels are low. Important for root development and flowering.")
        if K < 50:
            recommendations['soil_health_notes'].append("Potassium levels are low. Essential for disease resistance and fruit quality.")
        
        if soil_type == 'Sandy':
            recommendations['soil_health_notes'].append("Sandy soil: Higher fertilizer application frequency needed due to leaching.")
        elif soil_type == 'Clayey':
            recommendations['soil_health_notes'].append("Clayey soil: Better nutrient retention, but may need pH adjustment.")
        
        return recommendations

if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("Fertilizer Optimization Predictor")
    print("="*60)
    
    try:
        predictor = FertilizerOptimizationPredictor()
        
        # Example prediction
        result = predictor.predict(
            N=37, P=0, K=0,
            temperature=26, humidity=52, moisture=38,
            soil_type='Sandy', crop_type='Maize'
        )
        
        print("\nRecommended Fertilizer:", result['recommended_fertilizer'])
        print(f"Confidence: {result['confidence_score']:.2%}")
        
        print("\nFertilizer Info:")
        for key, value in result['recommendations']['fertilizer_info'].items():
            print(f"  {key}: {value}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the fertilizer models first using train_fertilizer_model.py")
