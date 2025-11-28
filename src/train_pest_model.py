import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime
import glob
from pathlib import Path
import shutil
import gc

class PestDetectionModel:
    """
    Pest and Disease Detection Model Trainer
    Uses CNN for image classification with memory-efficient data loading
    """
    
    def __init__(self, img_size=(224, 224), batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.label_encoder = LabelEncoder()
        self.class_names = []
        self.history = None
        self.temp_data_dir = None
        
    def prepare_data_structure(self, output_dir='temp_pest_data'):
        """
        Prepare a unified data structure for flow_from_directory
        This creates train/val splits in a temporary directory
        """
        print("="*60)
        print("Preparing Data Structure...")
        print("="*60)
        
        self.temp_data_dir = output_dir
        train_dir = os.path.join(output_dir, 'train')
        val_dir = os.path.join(output_dir, 'val')
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        all_classes = set()
        class_file_counts = {}
        
        # Collect all classes and files from all datasets
        # Using smaller datasets to avoid memory issues
        datasets_config = [
            {
                'path': 'DataSet For PestPrediction/DataSet/train',
                'max_per_class': None,  # Use all available images
                'name': 'DataSet_train'
            },
            {
                'path': 'DataSet For PestPrediction/farm_insects',
                'max_per_class': 100,
                'name': 'farm_insects'
            },
            {
                # Limit Pest_Dataset significantly to avoid memory issues
                'path': 'DataSet For PestPrediction/Datasets/Pest_Dataset',
                'max_per_class': 50,  # Reduced from 300 to keep dataset manageable
                'name': 'Pest_Dataset'
            }
        ]
        
        # First pass: collect all classes and file counts
        print("\nScanning datasets...")
        for config in datasets_config:
            base_dir = config['path']
            if not os.path.exists(base_dir):
                print(f"  Skipping {config['name']}: path not found")
                continue
                
            print(f"\n  Processing {config['name']}...")
            subdirs = [d for d in os.listdir(base_dir) 
                      if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith('.')]
            
            for class_name in subdirs:
                class_dir = os.path.join(base_dir, class_name)
                
                # Find all image files
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG', '*.gif', '*.GIF']:
                    image_files.extend(glob.glob(os.path.join(class_dir, '**', ext), recursive=True))
                    image_files.extend(glob.glob(os.path.join(class_dir, ext)))
                
                if len(image_files) == 0:
                    continue
                
                # Apply max per class limit (if specified)
                if config['max_per_class'] is not None and len(image_files) > config['max_per_class']:
                    np.random.seed(42)
                    image_files = np.random.choice(image_files, config['max_per_class'], replace=False).tolist()
                
                # Handle prefix for CCMT dataset
                final_class_name = class_name
                if 'prefix' in config:
                    final_class_name = f"{config['prefix']}{class_name}"
                
                if final_class_name not in class_file_counts:
                    class_file_counts[final_class_name] = []
                    all_classes.add(final_class_name)
                
                class_file_counts[final_class_name].extend(image_files)
                print(f"    {final_class_name}: {len(image_files)} images")
        
        print(f"\nTotal classes found: {len(all_classes)}")
        print(f"Total images: {sum(len(files) for files in class_file_counts.values())}")
        
        # Second pass: copy files to train/val directories with 80/20 split
        print("\nOrganizing data into train/val splits...")
        for class_name, image_files in class_file_counts.items():
            # Create class directories
            train_class_dir = os.path.join(train_dir, class_name)
            val_class_dir = os.path.join(val_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(val_class_dir, exist_ok=True)
            
            # Shuffle and split
            np.random.seed(42)
            np.random.shuffle(image_files)
            split_idx = int(len(image_files) * 0.8)
            train_files = image_files[:split_idx]
            val_files = image_files[split_idx:]
            
            # Copy files (with progress for large classes)
            copied = 0
            for img_path in train_files:
                try:
                    # Validate and convert image
                    img = Image.open(img_path)
                    img = img.convert('RGB')
                    
                    # Get filename
                    filename = os.path.basename(img_path)
                    # Ensure unique filename
                    dest_path = os.path.join(train_class_dir, filename)
                    counter = 1
                    while os.path.exists(dest_path):
                        name, ext = os.path.splitext(filename)
                        dest_path = os.path.join(train_class_dir, f"{name}_{counter}{ext}")
                        counter += 1
                    
                    # Save resized image
                    img_resized = img.resize(self.img_size)
                    img_resized.save(dest_path, 'JPEG', quality=95)
                    copied += 1
                except Exception as e:
                    continue
            
            for img_path in val_files:
                try:
                    img = Image.open(img_path)
                    img = img.convert('RGB')
                    
                    filename = os.path.basename(img_path)
                    dest_path = os.path.join(val_class_dir, filename)
                    counter = 1
                    while os.path.exists(dest_path):
                        name, ext = os.path.splitext(filename)
                        dest_path = os.path.join(val_class_dir, f"{name}_{counter}{ext}")
                        counter += 1
                    
                    img_resized = img.resize(self.img_size)
                    img_resized.save(dest_path, 'JPEG', quality=95)
                    copied += 1
                except Exception as e:
                    continue
            
            if copied > 0:
                print(f"  {class_name}: {len(train_files)} train, {len(val_files)} val")
        
        print(f"\nData preparation complete!")
        print(f"Train directory: {train_dir}")
        print(f"Validation directory: {val_dir}")
        
        return train_dir, val_dir, sorted(all_classes)
    
    def create_model(self, num_classes):
        """Create CNN model for pest detection"""
        print(f"\nCreating model with {num_classes} classes...")
        
        # Use a more efficient architecture with transfer learning potential
        model = models.Sequential([
            # First Conv Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.MaxPooling2D(2, 2),
            layers.BatchNormalization(),
            
            # Second Conv Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.BatchNormalization(),
            
            # Third Conv Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.BatchNormalization(),
            
            # Fourth Conv Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.BatchNormalization(),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, epochs=30, validation_split=0.2):
        """Train the pest detection model using flow_from_directory"""
        try:
            # Prepare data structure
            train_dir, val_dir, class_names = self.prepare_data_structure()
            
            self.class_names = np.array(class_names)
            num_classes = len(class_names)
            
            print(f"\nNumber of classes: {num_classes}")
            
            # Count samples
            train_samples = sum(len([f for f in os.listdir(os.path.join(train_dir, cls)) 
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) 
                              for cls in class_names)
            val_samples = sum(len([f for f in os.listdir(os.path.join(val_dir, cls)) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) 
                            for cls in class_names)
            
            print(f"Training samples: {train_samples}")
            print(f"Validation samples: {val_samples}")
            
            if train_samples == 0:
                raise ValueError("No training samples found!")
            
            # Data augmentation for training
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                fill_mode='nearest',
                brightness_range=[0.8, 1.2]
            )
            
            # Only rescale for validation
            val_datagen = ImageDataGenerator(rescale=1./255)
            
            # Create generators
            train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=True,
                seed=42
            )
            
            val_generator = val_datagen.flow_from_directory(
                val_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=False,
                seed=42
            )
            
            # Use actual number of classes from generator
            actual_num_classes = len(train_generator.class_indices)
            if actual_num_classes != num_classes:
                print(f"Adjusting class count: Expected {num_classes}, got {actual_num_classes} from generator")
                num_classes = actual_num_classes
                # Update class names to match generator
                class_names = sorted(train_generator.class_indices.keys())
                self.class_names = np.array(class_names)
            
            # Create label encoder mapping
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(class_names)
            
            # Store class indices for later use
            self.class_indices = train_generator.class_indices
            
            # Create model with correct number of classes
            self.model = self.create_model(num_classes)
            
            print("\n" + "="*60)
            print("Model Architecture:")
            print("="*60)
            self.model.summary()
            
            # Calculate steps
            steps_per_epoch = train_samples // self.batch_size
            validation_steps = val_samples // self.batch_size
            
            if steps_per_epoch == 0:
                steps_per_epoch = 1
            if validation_steps == 0:
                validation_steps = 1
            
            print(f"\nSteps per epoch: {steps_per_epoch}")
            print(f"Validation steps: {validation_steps}")
            
            # Create models/pest directory
            os.makedirs('models/pest', exist_ok=True)
            
            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=7, 
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.5, 
                    patience=3, 
                    min_lr=1e-7,
                    verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    'models/pest/best_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1,
                    save_weights_only=False
                ),
                keras.callbacks.CSVLogger('models/pest/training_log.csv')
            ]
            
            # Train model
            print("\n" + "="*60)
            print("Training Model...")
            print("="*60)
            
            self.history = self.model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=val_generator,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate
            print("\nEvaluating model...")
            val_loss, val_accuracy = self.model.evaluate(
                val_generator, 
                steps=validation_steps, 
                verbose=1
            )
            
            print(f"\nValidation Accuracy: {val_accuracy:.4f}")
            
            return val_accuracy
            
        except Exception as e:
            print(f"\nError during training: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Cleanup
            gc.collect()
    
    def save_model(self, model_dir='models/pest'):
        """Save the trained model and metadata"""
        try:
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(model_dir, 'pest_detection_model.h5')
            self.model.save(model_path)
            print(f"\nModel saved to: {model_path}")
            
            # Save label encoder
            import joblib
            encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
            joblib.dump(self.label_encoder, encoder_path)
            print(f"Label encoder saved to: {encoder_path}")
            
            # Save class indices mapping (for flow_from_directory compatibility)
            if hasattr(self, 'class_indices'):
                indices_path = os.path.join(model_dir, 'class_indices.json')
                with open(indices_path, 'w') as f:
                    json.dump(self.class_indices, f, indent=2)
                print(f"Class indices saved to: {indices_path}")
            
            # Save metadata
            metadata = {
                'model_type': 'CNN',
                'img_size': self.img_size,
                'num_classes': len(self.class_names),
                'class_names': self.class_names.tolist() if isinstance(self.class_names, np.ndarray) else list(self.class_names),
                'training_date': datetime.now().isoformat(),
                'val_accuracy': float(self.history.history['val_accuracy'][-1]) if self.history else None,
                'final_loss': float(self.history.history['val_loss'][-1]) if self.history else None
            }
            
            metadata_path = os.path.join(model_dir, 'model_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved to: {metadata_path}")
            
            return model_dir
            
        except Exception as e:
            print(f"Error saving model: {e}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == "__main__":
    print("="*60)
    print("Pest & Disease Detection Model Training")
    print("="*60)
    
    try:
        # Set memory growth to avoid OOM errors
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU memory growth enabled")
            except RuntimeError as e:
                print(f"GPU setup error: {e}")
        
        trainer = PestDetectionModel(img_size=(224, 224), batch_size=16)
        accuracy = trainer.train(epochs=25)
        trainer.save_model()
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Final Validation Accuracy: {accuracy:.4f}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
