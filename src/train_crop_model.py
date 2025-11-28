import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import json
from datetime import datetime

class CropRecommendationModel:
    """
    Crop Recommendation Model Trainer
    Supports multiple ML algorithms with model comparison and selection
    """
    
    def __init__(self, data_path='DataSet/Crop_recommendation.csv'):
        self.data_path = data_path
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
    def load_data(self):
        """Load and prepare the dataset"""
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)
        
        # Separate features and target
        X = df[self.feature_names]
        y = df['label']
        
        print(f"Dataset loaded: {df.shape[0]} samples, {len(y.unique())} crop types")
        print(f"Crops: {sorted(y.unique())}")
        
        return X, y
    
    def train_models(self, test_size=0.2, random_state=42):
        """Train multiple ML models and compare their performance"""
        X, y = self.load_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize models
        models_config = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, 
                random_state=random_state,
                max_depth=10,
                min_samples_split=5
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=random_state,
                max_depth=5,
                learning_rate=0.1
            ),
            'SVM': SVC(
                kernel='rbf',
                probability=True,
                random_state=random_state
            ),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        print("\n" + "="*60)
        print("Training Models...")
        print("="*60)
        
        results = {}
        
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            # Train model
            if name == 'SVM' or name == 'KNN':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            
            # Store model and results
            self.models[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'needs_scaling': name in ['SVM', 'KNN']
            }
            
            results[name] = accuracy
        
        # Select best model
        self.best_model_name = max(results, key=results.get)
        self.best_model = self.models[self.best_model_name]['model']
        
        print("\n" + "="*60)
        print(f"Best Model: {self.best_model_name} (Accuracy: {results[self.best_model_name]:.4f})")
        print("="*60)
        
        # Detailed evaluation of best model
        best_predictions = self.models[self.best_model_name]['predictions']
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, best_predictions))
        
        return self.best_model, results
    
    def save_model(self, model_dir='models'):
        """Save the best model, scaler, and metadata"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save best model
        model_path = os.path.join(model_dir, 'crop_recommendation_model.pkl')
        joblib.dump(self.best_model, model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")
        
        # Save metadata
        metadata = {
            'model_name': self.best_model_name,
            'accuracy': float(self.models[self.best_model_name]['accuracy']),
            'feature_names': self.feature_names,
            'needs_scaling': self.models[self.best_model_name]['needs_scaling'],
            'training_date': datetime.now().isoformat(),
            'all_model_accuracies': {k: float(v['accuracy']) for k, v in self.models.items()}
        }
        
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")
        
        # Save feature importance if available
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = dict(zip(
                self.feature_names, 
                self.best_model.feature_importances_
            ))
            importance_path = os.path.join(model_dir, 'feature_importance.json')
            with open(importance_path, 'w') as f:
                json.dump(feature_importance, f, indent=2)
            print(f"Feature importance saved to: {importance_path}")
        
        return model_path, scaler_path, metadata_path

if __name__ == "__main__":
    print("="*60)
    print("Crop Recommendation Model Training")
    print("="*60)
    
    # Initialize trainer
    trainer = CropRecommendationModel()
    
    # Train models
    best_model, results = trainer.train_models()
    
    # Save models
    trainer.save_model()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

