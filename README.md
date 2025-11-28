# Crop Recommendation System

## Overview

This is the first module of the Predictive Pestguard Enhanced Crop Recommendation System - **Crop Recommendation & Selection**.

The system uses machine learning to recommend the best crop based on soil nutrients (N, P, K), pH levels, and climate conditions (temperature, humidity, rainfall).

## Features

- **Multiple ML Algorithms**: Random Forest, Gradient Boosting, SVM, KNN
- **Model Selection**: Automatically selects the best performing model
- **Confidence Scores**: Provides confidence scores for predictions
- **Top N Recommendations**: Returns top N crop recommendations
- **Explainability**: Feature importance and prediction explanations
- **REST API**: Ready-to-use API endpoints for integration
- **Batch Processing**: Support for batch predictions
- **Modern Web Interface**: Responsive frontend with beautiful UI
- **Real-time Predictions**: Instant crop recommendations via web interface

## Project Structure

```
.
├── DataSet/
│   ├── Crop_recommendation.csv    # Main dataset
│   ├── dataset.csv                 # Additional dataset
│   └── dataset1.csv                # Additional dataset
├── src/
│   ├── train_crop_model.py        # Model training script
│   └── predict_crop.py            # Prediction script with explainability
├── api/
│   └── app.py                     # Flask REST API server (API only)
├── frontend/
│   ├── index.html                 # Frontend HTML
│   └── static/
│       ├── css/
│       │   └── style.css          # Frontend styles
│       └── js/
│           └── app.js              # Frontend JavaScript
├── models/                        # Trained models (created after training)
│   ├── crop_recommendation_model.pkl
│   ├── scaler.pkl
│   ├── model_metadata.json
│   └── feature_importance.json
├── server.py                      # Full-stack server (Frontend + API)
├── requirements.txt               # Python dependencies
└── README.md                      # This file

```

## Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

First, train the crop recommendation model:

```bash
python src/train_crop_model.py
```

This will:
- Train multiple ML algorithms (Random Forest, Gradient Boosting, SVM, KNN)
- Compare their performance
- Select the best model
- Save the model, scaler, and metadata to the `models/` directory

### 2. Make Predictions (Python Script)

Use the prediction script:

```python
from src.predict_crop import CropRecommendationPredictor

# Initialize predictor
predictor = CropRecommendationPredictor()

# Make prediction
result = predictor.predict(
    N=90,
    P=42,
    K=43,
    temperature=20.87,
    humidity=82.0,
    ph=6.5,
    rainfall=202.9,
    top_n=3
)

print(f"Recommended Crop: {result['recommended_crop']}")
print(f"Confidence: {result['confidence_score']:.4f}")
```

### 3. Use Web Interface (Frontend)

Start the full-stack server (includes frontend + API):

```bash
python server.py
```

Open your browser and navigate to:
- **Local**: `http://localhost:5000`
- **Network**: `http://10.12.98.248:5000` (or your server IP)

The web interface provides:
- Interactive form for entering soil and climate data
- Real-time crop recommendations with confidence scores
- Visual feature importance charts
- Top N recommendations display
- Insights and explanations

### 4. Use REST API Only

If you only want the API (without frontend):

```bash
python api/app.py
```

The API will be available at `http://localhost:5000`

#### API Endpoints

**Single Crop Recommendation:**
```bash
POST /api/crop/recommend
Content-Type: application/json

{
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.87,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.9,
    "top_n": 3
}
```

**Batch Crop Recommendations:**
```bash
POST /api/crop/batch-recommend
Content-Type: application/json

{
    "samples": [
        {
            "N": 90,
            "P": 42,
            "K": 43,
            "temperature": 20.87,
            "humidity": 82.0,
            "ph": 6.5,
            "rainfall": 202.9
        }
    ],
    "top_n": 3
}
```

**Model Information:**
```bash
GET /api/model/info
```

**Health Check:**
```bash
GET /health
```

## Dataset

The dataset (`Crop_recommendation.csv`) contains:
- **Features**: N, P, K (soil nutrients), temperature, humidity, ph, rainfall
- **Target**: Crop label (22 different crops)
- **Samples**: 2200 samples

**Crops included:**
apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute, kidneybeans, lentil, maize, mango, mothbeans, mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, rice, watermelon

## Model Explainability

The system provides:
- **Confidence Scores**: Probability scores for each recommendation
- **Feature Importance**: Which features most influence the prediction
- **Top Influencing Factors**: The top 3 most important features
- **Insights**: Automated insights based on input parameters

## Next Steps

This is Module 1 of 3:
1. ✅ **Crop Recommendation & Selection** (Current)
2. ⏳ Pest Forecasting / Pest & Disease Detection
3. ⏳ Fertilizer Optimization / Soil & Nutrient Management

## License

[Add your license here]

## Author

[Add your name/team here]

