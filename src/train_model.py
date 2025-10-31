"""
Car Brand Recognition Model Training Script
This script implements transfer learning using pre-trained CNN models.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

class CarBrandClassifier:
    def __init__(self, img_size=(224, 224), num_classes=8, model_name='MobileNetV2'):
        """
        Initialize the car brand classifier.
        
        Args:
            img_size (tuple): Input image size
            num_classes (int): Number of car brands to classify
            model_name (str): Base model architecture ('MobileNetV2' or 'ResNet50')
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = None
        self.history = None
        
    def create_model(self):
        """
        Create the car brand classification model using transfer learning.
        """
        print(f"Creating model with {self.model_name} architecture...")
        
        # Choose base model
        if self.model_name == 'MobileNetV2':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        elif self.model_name == 'ResNet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom classification head
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create the complete model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Model created successfully!")
        print(f"Total parameters: {self.model.count_params():,}")
        return self.model
    
    def train_model(self, train_generator, validation_generator, epochs=20):
        """
        Train the car brand classification model.
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs (int): Number of training epochs
        """
        print("Starting model training...")
        
        # Create callbacks
        callbacks = self._create_callbacks()
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return self.history
    
    def fine_tune_model(self, train_generator, validation_generator, epochs=10):
        """
        Fine-tune the model by unfreezing some layers of the base model.
        """
        print("Starting fine-tuning...")
        
        # Unfreeze the top layers of the base model
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = len(base_model.layers) // 2
        
        # Freeze all the layers before fine_tune_at
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Recompile with a lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001/10),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Continue training
        fine_tune_history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            callbacks=self._create_callbacks(prefix="fine_tuned_"),
            verbose=1
        )
        
        # Combine histories
        if self.history:
            for key in fine_tune_history.history:
                self.history.history[key].extend(fine_tune_history.history[key])
        
        print("Fine-tuning completed!")
        return fine_tune_history
    
    def _create_callbacks(self, prefix=""):
        """Create training callbacks."""
        # Ensure models directory exists
        Path("models").mkdir(exist_ok=True)
        
        callbacks = [
            ModelCheckpoint(
                filepath=f'models/{prefix}best_car_model_{self.model_name.lower()}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        return callbacks
    
    def evaluate_model(self, test_generator):
        """
        Evaluate the trained model on test data.
        
        Args:
            test_generator: Test data generator
        """
        print("Evaluating model on test data...")
        
        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(
            test_generator,
            steps=test_generator.samples // test_generator.batch_size,
            verbose=1
        )
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Get predictions for confusion matrix
        test_generator.reset()
        predictions = self.model.predict(
            test_generator,
            steps=test_generator.samples // test_generator.batch_size + 1,
            verbose=1
        )
        
        # Get true labels
        true_labels = test_generator.classes[:len(predictions)]
        predicted_labels = np.argmax(predictions, axis=1)
        
        # Create and save evaluation results
        evaluation_results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'evaluation_date': datetime.now().isoformat()
        }
        
        with open('models/evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        return test_loss, test_accuracy, predicted_labels, true_labels
    
    def plot_training_history(self):
        """Plot training history."""
        if not self.history:
            print("No training history available!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath=None):
        """Save the trained model."""
        if filepath is None:
            filepath = f'models/car_brand_classifier_{self.model_name.lower()}.h5'
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
        # Save model architecture and info
        model_info = {
            'model_name': self.model_name,
            'img_size': self.img_size,
            'num_classes': self.num_classes,
            'saved_date': datetime.now().isoformat()
        }
        
        info_path = filepath.replace('.h5', '_info.json')
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)

def main():
    """Main training function."""
    print("=== Car Brand Classification Training ===")
    
    # Import data preparation
    from data_preparation import CarDatasetPreprocessor
    
    # Initialize data preprocessor
    preprocessor = CarDatasetPreprocessor(batch_size=32)
    
    # Create data generators
    try:
        train_gen, val_gen, test_gen = preprocessor.create_data_generators()
        num_classes = train_gen.num_classes
        class_names = list(train_gen.class_indices.keys())
        
        print(f"Training with {num_classes} car brands: {class_names}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure you have properly organized dataset with images.")
        return
    
    # Initialize and create model
    classifier = CarBrandClassifier(
        img_size=(224, 224),
        num_classes=num_classes,
        model_name='MobileNetV2'  # You can change to 'ResNet50'
    )
    
    model = classifier.create_model()
    print(model.summary())
    
    # Train the model
    print("\n=== Phase 1: Initial Training ===")
    history1 = classifier.train_model(train_gen, val_gen, epochs=20)
    
    # Fine-tune the model
    print("\n=== Phase 2: Fine-tuning ===")
    history2 = classifier.fine_tune_model(train_gen, val_gen, epochs=10)
    
    # Evaluate the model
    print("\n=== Model Evaluation ===")
    test_loss, test_acc, pred_labels, true_labels = classifier.evaluate_model(test_gen)
    
    # Plot training history
    classifier.plot_training_history()
    
    # Save the final model
    classifier.save_model()
    
    print(f"\n=== Training Complete ===")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print("Model saved in the 'models' directory")

if __name__ == "__main__":
    main()