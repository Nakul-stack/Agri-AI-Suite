from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
import sys
import os

# Add src directory to path
project_root = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from predict_crop import CropRecommendationPredictor
from predict_fertilizer import FertilizerOptimizationPredictor

# Serve frontend (supports Vite build if available)
frontend_dir = os.path.join(project_root, 'frontend')
dist_dir = os.path.join(frontend_dir, 'dist')
use_vite_dist = os.path.exists(os.path.join(dist_dir, 'index.html'))

if use_vite_dist:
    # Serve built Vite app directly from dist
    app = Flask(__name__, static_folder=dist_dir, static_url_path='')
    index_dir = dist_dir
else:
    # Fallback to legacy static folder
    static_folder = os.path.join(frontend_dir, 'static')
    app = Flask(__name__, static_folder=static_folder, static_url_path='/static')
    index_dir = frontend_dir
CORS(app)  # Enable CORS for all routes

# Initialize predictors
try:
    model_dir = os.path.join(project_root, 'models')
    crop_predictor = CropRecommendationPredictor(model_dir=model_dir)
    print("="*60)
    print("Crop Recommendation Model loaded successfully!")
    print(f"Model: {crop_predictor.metadata['model_name']}")
    print(f"Accuracy: {crop_predictor.metadata['accuracy']:.4f}")
    print("="*60)
except Exception as e:
    print(f"ERROR: Failed to load crop model: {e}")
    crop_predictor = None

try:
    fertilizer_model_dir = os.path.join(project_root, 'models', 'fertilizer')
    fertilizer_predictor = FertilizerOptimizationPredictor(model_dir=fertilizer_model_dir)
    print("Fertilizer Optimization Model loaded successfully!")
    print("="*60)
except Exception as e:
    print(f"Warning: Failed to load fertilizer model: {e}")
    fertilizer_predictor = None

@app.route('/')
def index():
    return send_from_directory(index_dir, 'index.html')

# Direct prediction endpoint - simplified
@app.route('/api/crop/recommend', methods=['POST'])
def recommend_crop():
    """Crop recommendation endpoint - direct to model"""
    if crop_predictor is None:
        return jsonify({'error': 'Crop model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Get optional top_n parameter
        top_n = data.get('top_n', 3)
        
        # Make prediction directly
        result = crop_predictor.predict(
            N=float(data['N']),
            P=float(data['P']),
            K=float(data['K']),
            temperature=float(data['temperature']),
            humidity=float(data['humidity']),
            ph=float(data['ph']),
            rainfall=float(data['rainfall']),
            top_n=top_n
        )
        
        return jsonify(result), 200
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input data: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if crop_predictor is None:
        return jsonify({'error': 'Crop model not loaded'}), 500
    
    info = {
        'crop_model': {
            'model_name': crop_predictor.metadata['model_name'],
            'accuracy': crop_predictor.metadata['accuracy'],
            'training_date': crop_predictor.metadata['training_date'],
            'feature_names': crop_predictor.feature_names,
            'all_model_accuracies': crop_predictor.metadata.get('all_model_accuracies', {})
        }
    }
    
    if fertilizer_predictor:
        info['fertilizer_model'] = {
            'status': 'loaded',
            'model_name': fertilizer_predictor.metadata['model_name'],
            'accuracy': fertilizer_predictor.metadata['accuracy'],
            'available_fertilizers': fertilizer_predictor.metadata['fertilizer_classes'],
            'available_crops': fertilizer_predictor.metadata.get('crop_types', []),
            'available_soil_types': fertilizer_predictor.metadata.get('soil_types', [])
        }
    
    return jsonify(info), 200

@app.route('/api/fertilizer/recommend', methods=['POST'])
def recommend_fertilizer():
    """Fertilizer optimization endpoint"""
    if fertilizer_predictor is None:
        return jsonify({'error': 'Fertilizer model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        # Required fields
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Optional fields
        result = fertilizer_predictor.predict(
            N=float(data['N']),
            P=float(data['P']),
            K=float(data['K']),
            temperature=float(data['temperature']),
            humidity=float(data['humidity']),
            moisture=data.get('moisture'),
            soil_type=data.get('soil_type'),
            crop_type=data.get('crop_type'),
            ph=data.get('ph')
        )
        
        return jsonify(result), 200
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input data: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'crop_model_loaded': crop_predictor is not None,
        'fertilizer_model_loaded': fertilizer_predictor is not None,
        'crop_model_name': crop_predictor.metadata['model_name'] if crop_predictor else None,
        'crop_model_accuracy': crop_predictor.metadata['accuracy'] if crop_predictor else None
    }), 200

if __name__ == '__main__':
    print("="*60)
    print("Predictive Pestguard AI System")
    print("="*60)
    print("Modules:")
    print("  1. Crop Recommendation & Selection")
    print("  2. Fertilizer Optimization / Soil & Nutrient Management")
    print("="*60)
    print("\nStarting server...")
    print("Local access: http://localhost:5000")
    print("Network access: http://10.12.98.248:5000")
    print("Network access: http://172.16.0.2:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

