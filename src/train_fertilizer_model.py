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

class FertilizerOptimizationModel:
    """
    Enhanced Fertilizer Optimization Model Trainer
    Predicts optimal fertilizer type based on soil, climate, and crop
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.soil_encoder = LabelEncoder()
        self.crop_encoder = LabelEncoder()
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        
    def load_and_prepare_data(self):
        """Load and prepare fertilizer optimization datasets"""
        print("="*60)
        print("Loading Fertilizer Optimization Datasets...")
        print("="*60)
        
        datasets = []
        
        # Dataset 1: fertilizer_giant_training_dataset.csv (NEW - VERY LARGE DATASET)
        try:
            df_giant = pd.read_csv('dataset for fertilizer/fertilizer_giant_training_dataset.csv')
            print(f"\nDataset 1 (fertilizer_giant_training_dataset.csv):")
            print(f"  Rows: {len(df_giant)}")
            print(f"  Columns: {list(df_giant.columns)}")
            if 'Recommended Fertilizer' in df_giant.columns:
                print(f"  Fertilizers: {sorted(df_giant['Recommended Fertilizer'].astype(str).unique())}")
            
            df_giant_processed = self.prepare_giant_dataset(df_giant)
            if df_giant_processed is not None:
                datasets.append(df_giant_processed)
                print(f"  Processed: {len(df_giant_processed)} rows")
        except Exception as e:
            print(f"Warning: Could not load fertilizer_giant_training_dataset.csv: {e}")
            import traceback
            traceback.print_exc()

        # Dataset 2: fertilizer_test_data.csv (PRIMARY - LARGE DATASET)
        try:
            df_test = pd.read_csv('dataset for fertilizer/fertilizer_test_data.csv')
            print(f"\nDataset 2 (fertilizer_test_data.csv):")
            print(f"  Rows: {len(df_test)}")
            print(f"  Columns: {list(df_test.columns)}")
            print(f"  Fertilizers: {sorted(df_test['Expected Fertilizer'].unique())}")
            print(f"  Fertilizer distribution: {df_test['Expected Fertilizer'].value_counts().to_dict()}")
            
            df_test_processed = self.prepare_test_dataset(df_test)
            if df_test_processed is not None:
                datasets.append(df_test_processed)
                print(f"  Processed: {len(df_test_processed)} rows")
        except Exception as e:
            print(f"Warning: Could not load fertilizer_test_data.csv: {e}")
            import traceback
            traceback.print_exc()
        
        # Dataset 3: User's Fertilizer Prediction.csv (ADDITIONAL DATA)
        try:
            df1 = pd.read_csv('dataset for fertilizer/Fertilizer Prediction.csv')
            print(f"\nDataset 3 (Fertilizer Prediction.csv):")
            print(f"  Rows: {len(df1)}")
            print(f"  Columns: {list(df1.columns)}")
            print(f"  Fertilizers: {df1['Fertilizer Name'].unique()}")
            
            # Clean column names
            df1.columns = df1.columns.str.strip()
            
            # Prepare this dataset
            df1_processed = self.prepare_dataset1(df1)
            if df1_processed is not None:
                # Only add fertilizers that match the main dataset
                main_fertilizers = set()
                if datasets:
                    main_fertilizers = set(datasets[0]['fertilizer_name'].unique())
                    for d in datasets[1:]:
                        main_fertilizers |= set(d['fertilizer_name'].unique())
                df1_processed = self.standardize_fertilizer_names(df1_processed)
                if main_fertilizers:
                    df1_processed = df1_processed[df1_processed['fertilizer_name'].isin(main_fertilizers)]
                datasets.append(df1_processed)
                print(f"  Processed: {len(df1_processed)} rows (filtered to match main dataset)")
        except Exception as e:
            print(f"Warning: Could not load Fertilizer Prediction.csv: {e}")
            import traceback
            traceback.print_exc()
        
        if not datasets:
            raise ValueError("No datasets could be loaded!")
        
        # Combine datasets
        print("\n" + "="*60)
        print("Merging Datasets...")
        print("="*60)
        
        df_combined = pd.concat(datasets, ignore_index=True)
        
        # Standardize fertilizer names
        df_combined = self.standardize_fertilizer_names(df_combined)
        
        df_combined = df_combined.drop_duplicates()
        df_combined = df_combined.dropna()
        
        # Filter to keep only fertilizers with sufficient samples (at least 20)
        fertilizer_counts = df_combined['fertilizer_name'].value_counts()
        valid_fertilizers = fertilizer_counts[fertilizer_counts >= 20].index.tolist()
        df_combined = df_combined[df_combined['fertilizer_name'].isin(valid_fertilizers)]
        
        print(f"Combined dataset shape: {df_combined.shape}")
        print(f"Unique fertilizers: {df_combined['fertilizer_name'].nunique()}")
        print(f"Fertilizers: {sorted(df_combined['fertilizer_name'].unique())}")
        print(f"Fertilizer distribution:")
        print(df_combined['fertilizer_name'].value_counts())
        
        return df_combined
    
    def prepare_giant_dataset(self, df):
        """Prepare fertilizer_giant_training_dataset.csv dataset"""
        processed = pd.DataFrame()
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Map columns
        temp_col = [col for col in df.columns if 'Temperature' in col or 'temp' in col.lower()][0]
        humidity_col = [col for col in df.columns if 'Humidity' in col][0]
        moisture_col = [col for col in df.columns if 'Moisture' in col][0]
        
        processed['temperature'] = pd.to_numeric(df[temp_col], errors='coerce')
        processed['humidity'] = pd.to_numeric(df[humidity_col], errors='coerce')
        processed['moisture'] = pd.to_numeric(df[moisture_col], errors='coerce')
        
        processed['soil_type'] = df['Soil Type'].astype(str)
        processed['crop_type'] = df['Crop Type'].astype(str)
        
        # Nutrients named as ppm
        n_col = [c for c in df.columns if 'Nitrogen' in c][0]
        p_col = [c for c in df.columns if 'Phosphorus' in c or 'Phosphate' in c][0]
        k_col = [c for c in df.columns if 'Potassium' in c][0]
        processed['N'] = pd.to_numeric(df[n_col], errors='coerce')
        processed['P'] = pd.to_numeric(df[p_col], errors='coerce')
        processed['K'] = pd.to_numeric(df[k_col], errors='coerce')
        
        ph_col = [col for col in df.columns if 'pH' in col or col.lower() == 'ph'][0]
        processed['ph'] = pd.to_numeric(df[ph_col], errors='coerce')
        
        if 'Rainfall (mm)' in df.columns:
            processed['rainfall'] = pd.to_numeric(df['Rainfall (mm)'], errors='coerce')
        else:
            processed['rainfall'] = 150
        
        # Target fertilizer column name
        fert_col = 'Recommended Fertilizer' if 'Recommended Fertilizer' in df.columns else 'Expected Fertilizer'
        processed['fertilizer_name'] = df[fert_col].astype(str)
        
        # Drop rows missing critical values
        processed = processed.dropna(subset=['temperature','humidity','moisture','N','P','K','ph','fertilizer_name'])
        
        return processed

    def prepare_test_dataset(self, df):
        """Prepare fertilizer_test_data.csv dataset"""
        processed = pd.DataFrame()
        
        # Clean column names (handle encoding issues)
        df.columns = df.columns.str.strip()
        
        # Map columns - handle different column name formats
        temp_col = [col for col in df.columns if 'Temperature' in col or 'temp' in col.lower()][0]
        humidity_col = [col for col in df.columns if 'Humidity' in col][0]
        moisture_col = [col for col in df.columns if 'Moisture' in col][0]
        
        processed['temperature'] = pd.to_numeric(df[temp_col], errors='coerce')
        processed['humidity'] = pd.to_numeric(df[humidity_col], errors='coerce')
        processed['moisture'] = pd.to_numeric(df[moisture_col], errors='coerce')
        
        # Categorical features
        processed['soil_type'] = df['Soil Type'].astype(str)
        processed['crop_type'] = df['Crop Type'].astype(str)
        
        # Nutrient levels
        processed['N'] = pd.to_numeric(df['Nitrogen (N)'], errors='coerce')
        processed['P'] = pd.to_numeric(df['Phosphorus (P)'], errors='coerce')
        processed['K'] = pd.to_numeric(df['Potassium (K)'], errors='coerce')
        
        # Additional features
        ph_col = [col for col in df.columns if 'pH' in col or col.lower() == 'ph'][0]
        processed['ph'] = pd.to_numeric(df[ph_col], errors='coerce')
        
        rainfall_col = [col for col in df.columns if 'Rainfall' in col][0]
        processed['rainfall'] = pd.to_numeric(df[rainfall_col], errors='coerce')
        
        # Target: fertilizer name
        processed['fertilizer_name'] = df['Expected Fertilizer'].astype(str)
        
        # Remove rows with missing critical data
        processed = processed.dropna(subset=['temperature', 'humidity', 'moisture', 'N', 'P', 'K', 'fertilizer_name'])
        
        return processed
    
    def prepare_dataset1(self, df):
        """Prepare Fertilizer Prediction.csv dataset"""
        # Map columns
        processed = pd.DataFrame()
        
        # Climate features
        processed['temperature'] = df['Temparature']  # Note: typo in original
        processed['humidity'] = pd.to_numeric(df['Humidity '].str.strip() if 'Humidity ' in df.columns else df['Humidity'], errors='coerce')
        processed['moisture'] = df['Moisture']
        
        # Categorical features
        processed['soil_type'] = df['Soil Type']
        processed['crop_type'] = df['Crop Type']
        
        # Nutrient levels (these are the current soil levels in the dataset)
        processed['N'] = df['Nitrogen']
        processed['P'] = df['Phosphorous']
        processed['K'] = df['Potassium']
        
        # Add default values for missing features
        processed['ph'] = 6.5  # Default pH
        processed['rainfall'] = 150  # Default rainfall
        
        # Target: fertilizer name
        processed['fertilizer_name'] = df['Fertilizer Name']
        
        return processed
    
    def prepare_dataset2(self, df):
        """Convert dataset.csv to fertilizer recommendation format"""
        # This dataset has N, P, K, ph, EC, micronutrients, and crop
        # We need to infer fertilizer recommendations based on nutrient levels
        
        fertilizer_mapping = self.infer_fertilizer_from_nutrients(df)
        
        processed = pd.DataFrame()
        
        # Use average climate values (since dataset2 doesn't have climate)
        processed['temperature'] = 28  # Default
        processed['humidity'] = 60     # Default
        processed['moisture'] = 40     # Default
        processed['ph'] = df['ph'] if 'ph' in df.columns else 6.5
        processed['rainfall'] = 150  # Default
        
        # Soil type - infer from pH
        processed['soil_type'] = df['ph'].apply(lambda x: 'Clayey' if x < 6.0 else 'Loamy' if x < 7.0 else 'Sandy')
        
        # Crop type
        processed['crop_type'] = df['label']
        
        # Nutrients
        processed['N'] = df['N']
        processed['P'] = df['P']
        processed['K'] = df['K']
        
        # Inferred fertilizer
        processed['fertilizer_name'] = fertilizer_mapping
        
        return processed
    
    def infer_fertilizer_from_nutrients(self, df):
        """Infer fertilizer type based on nutrient deficiencies"""
        fertilizers = []
        
        for idx, row in df.iterrows():
            n, p, k = row['N'], row['P'], row['K']
            
            # Determine fertilizer based on what's needed
            if n < 20 and p < 20 and k < 20:
                fert = 'Urea'  # Primarily N
            elif p > 30 and n < 20:
                fert = 'DAP'  # Di-Ammonium Phosphate (N-P)
            elif n > 20 and p > 20 and k > 20:
                fert = '17-17-17'  # Balanced NPK
            elif n > 25 and p > 25:
                fert = '28-28'  # High N-P
            elif p > 30:
                fert = '14-35-14'  # High P
            elif n > 30:
                fert = 'Urea'
            else:
                fert = '20-20'  # Balanced
            
            fertilizers.append(fert)
        
        return fertilizers
    
    def standardize_fertilizer_names(self, df):
        """Standardize fertilizer names across datasets"""
        # Mapping to standardize fertilizer names
        fertilizer_mapping = {
            'NPK 10-26-26': '10-26-26',
            '10-26-26': '10-26-26',
            '14-35-14': '14-35-14',
            'DAP': 'DAP',
            'Urea': 'Urea',
            'MOP': 'MOP',
            'Potash': 'Potash',
            'NPK': 'NPK',
            '17-17-17': '17-17-17',
            '20-20': '20-20',
            '28-28': '28-28'
        }
        
        df['fertilizer_name'] = df['fertilizer_name'].str.strip()
        df['fertilizer_name'] = df['fertilizer_name'].map(fertilizer_mapping).fillna(df['fertilizer_name'])
        
        return df
    
    def train_models(self, test_size=0.2, random_state=42):
        """Train models for predicting fertilizer type"""
        df = self.load_and_prepare_data()
        
        # Encode categorical features
        df['soil_type_encoded'] = self.soil_encoder.fit_transform(df['soil_type'])
        df['crop_type_encoded'] = self.crop_encoder.fit_transform(df['crop_type'])
        
        # Feature selection - include pH and rainfall if available
        base_features = ['temperature', 'humidity', 'moisture', 'N', 'P', 'K', 
                        'soil_type_encoded', 'crop_type_encoded']
        
        # Add pH and rainfall if available
        if 'ph' in df.columns:
            df['ph'] = pd.to_numeric(df['ph'], errors='coerce').fillna(6.5)
            base_features.append('ph')
        
        if 'rainfall' in df.columns:
            df['rainfall'] = pd.to_numeric(df['rainfall'], errors='coerce').fillna(150)
            base_features.append('rainfall')
        
        self.feature_names = base_features
        X = df[self.feature_names].fillna(0)
        
        # Target: fertilizer name
        y = self.label_encoder.fit_transform(df['fertilizer_name'])
        
        print(f"\nFeatures: {self.feature_names}")
        print(f"Target classes: {self.label_encoder.classes_}")
        print(f"Number of samples: {len(X)}")
        
        # Split data with stratification - use smaller test size for more training data
        test_size_adjusted = 0.15  # Use more data for training
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_adjusted, random_state=random_state, stratify=y
            )
        except ValueError:
            # If stratification fails due to small classes, use regular split
            print("Warning: Stratification failed, using regular split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_adjusted, random_state=random_state
            )
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize models with improved hyperparameters
        models_config = {
            'RandomForest': RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                max_depth=25,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced',
                bootstrap=True,
                oob_score=True
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=300,
                random_state=random_state,
                max_depth=12,
                learning_rate=0.03,
                subsample=0.8,
                min_samples_split=2
            ),
            'SVM': SVC(
                kernel='rbf',
                probability=True,
                random_state=random_state,
                C=100,
                gamma='scale',
                class_weight='balanced'
            ),
            'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto')
        }
        
        print("\n" + "="*60)
        print("Training Fertilizer Prediction Models...")
        print("="*60)
        
        results = {}
        
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            if name in ['SVM', 'KNN']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} Accuracy: {accuracy:.4f}")
            
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
    
    def save_model(self, model_dir='models/fertilizer'):
        """Save fertilizer optimization models"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save best model
        model_path = os.path.join(model_dir, 'fertilizer_model.pkl')
        joblib.dump(self.best_model, model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")
        
        # Save encoders
        label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        joblib.dump(self.label_encoder, label_encoder_path)
        print(f"Label encoder saved to: {label_encoder_path}")
        
        soil_encoder_path = os.path.join(model_dir, 'soil_encoder.pkl')
        joblib.dump(self.soil_encoder, soil_encoder_path)
        print(f"Soil encoder saved to: {soil_encoder_path}")
        
        crop_encoder_path = os.path.join(model_dir, 'crop_encoder.pkl')
        joblib.dump(self.crop_encoder, crop_encoder_path)
        print(f"Crop encoder saved to: {crop_encoder_path}")
        
        # Save metadata
        metadata = {
            'model_name': self.best_model_name,
            'accuracy': float(self.models[self.best_model_name]['accuracy']),
            'feature_names': self.feature_names,
            'needs_scaling': self.models[self.best_model_name]['needs_scaling'],
            'training_date': datetime.now().isoformat(),
            'all_model_accuracies': {k: float(v['accuracy']) for k, v in self.models.items()},
            'fertilizer_classes': list(self.label_encoder.classes_),
            'soil_types': list(self.soil_encoder.classes_),
            'crop_types': list(self.crop_encoder.classes_),
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
        
        return model_dir

if __name__ == "__main__":
    print("="*60)
    print("Enhanced Fertilizer Optimization Model Training")
    print("Using Multiple Compatible Datasets")
    print("="*60)
    
    trainer = FertilizerOptimizationModel()
    best_model, results = trainer.train_models()
    trainer.save_model()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
