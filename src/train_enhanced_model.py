import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import json
from datetime import datetime

class EnhancedCropRecommendationModel:
    """
    Enhanced Crop Recommendation Model Trainer
    Supports multiple datasets with intelligent feature mapping
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        
    def load_and_prepare_datasets(self):
        """Load and merge multiple datasets"""
        print("="*60)
        print("Loading Datasets...")
        print("="*60)
        
        datasets = []
        
        # Dataset 1: Original crop recommendation dataset
        try:
            df1 = pd.read_csv('DataSet/Crop_recommendation.csv')
            print(f"\nDataset 1 (Crop_recommendation.csv):")
            print(f"  Rows: {len(df1)}")
            print(f"  Columns: {list(df1.columns)}")
            datasets.append(df1)
        except Exception as e:
            print(f"Warning: Could not load Crop_recommendation.csv: {e}")
        
        # Dataset 2: New dataset with soil properties and weather
        try:
            df2 = pd.read_csv('DataSet/Crop Recommendation using Soil Properties and Weather Prediction.csv')
            print(f"\nDataset 2 (Crop Recommendation using Soil Properties and Weather Prediction.csv):")
            print(f"  Rows: {len(df2)}")
            print(f"  Columns: {list(df2.columns)[:10]}...")
            
            # Prepare this dataset - map to common features
            df2_processed = self.prepare_dataset2(df2)
            if df2_processed is not None:
                datasets.append(df2_processed)
        except Exception as e:
            print(f"Warning: Could not load new dataset: {e}")
        
        if not datasets:
            raise ValueError("No datasets could be loaded!")
        
        # Merge datasets
        print("\n" + "="*60)
        print("Merging Datasets...")
        print("="*60)
        
        # Use common features across datasets
        common_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        merged_data = []
        for df in datasets:
            # Try to extract common features
            df_features = {}
            for feat in common_features:
                if feat in df.columns:
                    df_features[feat] = df[feat]
                elif feat == 'temperature' and 'T2M_MAX' in df.columns:
                    # Use average temperature from new dataset
                    temp_cols = [col for col in df.columns if 'T2M_MAX' in col]
                    if temp_cols:
                        df_features[feat] = df[temp_cols].mean(axis=1)
                elif feat == 'humidity' and 'QV2M' in df.columns:
                    # Use average humidity
                    hum_cols = [col for col in df.columns if 'QV2M' in col]
                    if hum_cols:
                        df_features[feat] = df[hum_cols].mean(axis=1)
                elif feat == 'rainfall' and 'PRECTOTCORR' in df.columns:
                    # Use average rainfall
                    rain_cols = [col for col in df.columns if 'PRECTOTCORR' in col]
                    if rain_cols:
                        df_features[feat] = df[rain_cols].sum(axis=1)  # Sum for annual rainfall
                elif feat == 'ph' and 'Ph' in df.columns:
                    df_features[feat] = df['Ph']
            
            # Check if we have all required features
            if all(feat in df_features for feat in common_features) and 'label' in df.columns:
                df_merged = pd.DataFrame(df_features)
                df_merged['label'] = df['label']
                merged_data.append(df_merged)
                print(f"  Added {len(df_merged)} rows from dataset")
        
        if not merged_data:
            # Fallback to original dataset only
            print("Warning: Could not merge datasets, using original dataset only")
            df_final = datasets[0]
        else:
            # Combine all merged datasets
            df_final = pd.concat(merged_data, ignore_index=True)
            print(f"\nTotal merged rows: {len(df_final)}")
        
        # Clean data
        df_final = df_final.dropna()
        df_final = df_final[df_final['N'] >= 0]  # Remove negative values
        
        print(f"\nFinal dataset shape: {df_final.shape}")
        print(f"Unique crops: {df_final['label'].nunique()}")
        print(f"Crops: {sorted(df_final['label'].unique())}")
        
        return df_final
    
    def prepare_dataset2(self, df):
        """Prepare the second dataset for merging"""
        # This dataset has different structure, we'll extract what we can
        common_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        processed = {}
        
        # Map existing features
        if 'Ph' in df.columns:
            processed['ph'] = df['Ph']
        if 'P' in df.columns:
            processed['P'] = df['P']
        if 'K' in df.columns:
            processed['K'] = df['K']
        if 'N' in df.columns:
            processed['N'] = df['N']
        
        # Calculate temperature (average of max temps)
        temp_cols = [col for col in df.columns if 'T2M_MAX' in col]
        if temp_cols:
            processed['temperature'] = df[temp_cols].mean(axis=1)
        
        # Calculate humidity (average of humidity columns)
        hum_cols = [col for col in df.columns if 'QV2M' in col]
        if hum_cols:
            processed['humidity'] = df[hum_cols].mean(axis=1) * 100  # Convert to percentage
        
        # Calculate rainfall (sum of seasonal rainfall)
        rain_cols = [col for col in df.columns if 'PRECTOTCORR' in col]
        if rain_cols:
            processed['rainfall'] = df[rain_cols].sum(axis=1) * 30  # Convert to mm (approximate)
        
        # Check if we have all features
        if all(feat in processed for feat in common_features) and 'label' in df.columns:
            result_df = pd.DataFrame(processed)
            result_df['label'] = df['label']
            return result_df
        
        return None
    
    def train_models(self, test_size=0.2, random_state=42):
        """Train multiple ML models"""
        df = self.load_and_prepare_datasets()
        
        # Separate features and target
        self.feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        X = df[self.feature_names]
        y = df['label']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize models
        models_config = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, 
                random_state=random_state,
                max_depth=15,
                min_samples_split=5
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=random_state,
                max_depth=7,
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
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            
            # Store model and results
            self.models[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'needs_scaling': name in ['SVM', 'KNN']
            }
            
            results[name] = accuracy
        
        # Select best model
        self.best_model_name = max(results, key=results.get)
        self.best_model = self.models[self.best_model_name]['model']
        
        print("\n" + "="*60)
        print(f"Best Model: {self.best_model_name} (Accuracy: {results[self.best_model_name]:.4f})")
        print("="*60)
        
        # Detailed evaluation
        best_predictions = self.models[self.best_model_name]['predictions']
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, best_predictions, 
                                  target_names=self.label_encoder.classes_))
        
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
        
        # Save label encoder
        encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        joblib.dump(self.label_encoder, encoder_path)
        print(f"Label encoder saved to: {encoder_path}")
        
        # Save metadata
        metadata = {
            'model_name': self.best_model_name,
            'accuracy': float(self.models[self.best_model_name]['accuracy']),
            'feature_names': self.feature_names,
            'needs_scaling': self.models[self.best_model_name]['needs_scaling'],
            'training_date': datetime.now().isoformat(),
            'all_model_accuracies': {k: float(v['accuracy']) for k, v in self.models.items()},
            'crop_classes': list(self.label_encoder.classes_),
            'num_classes': len(self.label_encoder.classes_)
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
    print("Enhanced Crop Recommendation Model Training")
    print("Training with Multiple Datasets")
    print("="*60)
    
    # Initialize trainer
    trainer = EnhancedCropRecommendationModel()
    
    # Train models
    best_model, results = trainer.train_models()
    
    # Save models
    trainer.save_model()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

