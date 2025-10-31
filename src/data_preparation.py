"""
Data Preparation and Preprocessing Script
This script handles image preprocessing, data augmentation, and dataset preparation.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import shutil
from PIL import Image
import cv2

class CarDatasetPreprocessor:
    def __init__(self, dataset_path="dataset", img_size=(224, 224), batch_size=32):
        """
        Initialize the data preprocessor.
        
        Args:
            dataset_path (str): Path to the dataset directory
            img_size (tuple): Target image size for preprocessing
            batch_size (int): Batch size for data loading
        """
        self.dataset_path = Path(dataset_path)
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = []
        
    def explore_dataset(self):
        """
        Explore and analyze the dataset structure and statistics.
        """
        print("=== Dataset Exploration ===")
        
        # Check if dataset exists
        if not self.dataset_path.exists():
            print(f"Dataset path {self.dataset_path} does not exist!")
            return False
            
        # Analyze train directory
        train_dir = self.dataset_path / "train"
        if train_dir.exists():
            self.class_names = [d.name for d in train_dir.iterdir() if d.is_dir()]
            print(f"Found {len(self.class_names)} car brands: {self.class_names}")
            
            # Count images per class
            class_counts = {}
            total_images = 0
            
            for class_name in self.class_names:
                class_path = train_dir / class_name
                if class_path.exists():
                    image_files = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png"))
                    class_counts[class_name] = len(image_files)
                    total_images += len(image_files)
                    print(f"  {class_name}: {len(image_files)} images")
            
            print(f"\nTotal training images: {total_images}")
            
            # Visualize class distribution
            self.plot_class_distribution(class_counts)
            
        return True
    
    def plot_class_distribution(self, class_counts):
        """Plot the distribution of images across different car brands."""
        plt.figure(figsize=(10, 6))
        brands = list(class_counts.keys())
        counts = list(class_counts.values())
        
        plt.bar(brands, counts, color='skyblue', alpha=0.7)
        plt.title('Distribution of Car Images by Brand')
        plt.xlabel('Car Brand')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('dataset_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def preprocess_images(self, input_dir, output_dir):
        """
        Preprocess images: resize, normalize, and save.
        
        Args:
            input_dir (Path): Input directory containing images
            output_dir (Path): Output directory for processed images
        """
        print(f"Preprocessing images from {input_dir} to {output_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for class_dir in input_dir.iterdir():
            if class_dir.is_dir():
                class_output_dir = output_dir / class_dir.name
                class_output_dir.mkdir(exist_ok=True)
                
                image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                
                for img_file in image_files:
                    try:
                        # Load and preprocess image
                        img = Image.open(img_file)
                        img = img.convert('RGB')  # Ensure RGB format
                        img = img.resize(self.img_size, Image.Resampling.LANCZOS)
                        
                        # Save preprocessed image
                        output_path = class_output_dir / img_file.name
                        img.save(output_path, 'JPEG', quality=95)
                        
                    except Exception as e:
                        print(f"Error processing {img_file}: {e}")
    
    def create_data_generators(self):
        """
        Create data generators with augmentation for training and validation.
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation and test
        validation_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.dataset_path / "train",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            self.dataset_path / "validation",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        
        test_generator = test_datagen.flow_from_directory(
            self.dataset_path / "test",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, validation_generator, test_generator
    
    def visualize_sample_images(self, generator, num_samples=9):
        """
        Visualize sample images from the dataset.
        """
        plt.figure(figsize=(12, 12))
        
        # Get a batch of images
        batch_images, batch_labels = next(generator)
        class_names = list(generator.class_indices.keys())
        
        for i in range(min(num_samples, len(batch_images))):
            plt.subplot(3, 3, i + 1)
            plt.imshow(batch_images[i])
            
            # Get class name from label
            label_idx = np.argmax(batch_labels[i])
            class_name = class_names[label_idx]
            plt.title(f'Car Brand: {class_name}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to run data preparation."""
    print("=== Car Dataset Preparation ===")
    
    # Initialize preprocessor
    preprocessor = CarDatasetPreprocessor()
    
    # Explore dataset
    if not preprocessor.explore_dataset():
        print("Please run src/download_data.py first to set up the dataset structure.")
        return
    
    # Create data generators
    print("\n=== Creating Data Generators ===")
    try:
        train_gen, val_gen, test_gen = preprocessor.create_data_generators()
        print(f"Training samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Test samples: {test_gen.samples}")
        print(f"Number of classes: {train_gen.num_classes}")
        print(f"Class names: {list(train_gen.class_indices.keys())}")
        
        # Visualize sample images
        print("\n=== Visualizing Sample Images ===")
        preprocessor.visualize_sample_images(train_gen)
        
    except Exception as e:
        print(f"Error creating data generators: {e}")
        print("Make sure you have images in the dataset/train directory organized by brand.")

if __name__ == "__main__":
    main()