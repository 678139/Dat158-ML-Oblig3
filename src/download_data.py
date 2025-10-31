"""
Data Download Script for Stanford Cars Dataset
This script downloads and prepares the Stanford Cars dataset from Kaggle.
"""

import os
import zipfile
import requests
from pathlib import Path
import shutil

def download_stanford_cars_dataset():
    """
    Download Stanford Cars dataset from Kaggle.
    Note: You need to setup Kaggle API credentials first.
    """
    print("Setting up dataset directory...")
    
    # Create dataset directories
    base_dir = Path("dataset")
    base_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different car brands
    car_brands = ["bmw", "tesla", "toyota", "mercedes", "audi", "ford", "honda", "nissan"]
    
    for split in ["train", "test", "validation"]:
        split_dir = base_dir / split
        split_dir.mkdir(exist_ok=True)
        
        for brand in car_brands:
            brand_dir = split_dir / brand
            brand_dir.mkdir(exist_ok=True)
    
    print("Dataset directory structure created successfully!")
    print("\nTo download the actual dataset:")
    print("1. Install Kaggle API: pip install kaggle")
    print("2. Setup Kaggle credentials (kaggle.json in ~/.kaggle/)")
    print("3. Run: kaggle datasets download -d jessicali9530/stanford-cars-dataset")
    print("4. Extract and organize images into the created folder structure")
    
    return True

def setup_sample_structure():
    """
    Create a sample structure to demonstrate the expected organization.
    """
    sample_readme = """
# Dataset Organization

Place your car images in the following structure:

dataset/
├── train/
│   ├── bmw/        # BMW car images for training
│   ├── tesla/      # Tesla car images for training
│   ├── toyota/     # Toyota car images for training
│   ├── mercedes/   # Mercedes car images for training
│   ├── audi/       # Audi car images for training
│   ├── ford/       # Ford car images for training
│   ├── honda/      # Honda car images for training
│   └── nissan/     # Nissan car images for training
├── test/
│   ├── bmw/        # BMW car images for testing
│   ├── tesla/      # Tesla car images for testing
│   └── ...         # Same structure as train
└── validation/
    ├── bmw/        # BMW car images for validation
    ├── tesla/      # Tesla car images for validation
    └── ...         # Same structure as train

## Data Sources
- Stanford Cars Dataset: https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset
- Manual collection from various sources (ensure proper licensing)

## Image Requirements
- Format: JPG or PNG
- Minimum resolution: 224x224 pixels
- Clear view of the car
- Properly labeled by brand
"""
    
    with open("dataset/README.md", "w") as f:
        f.write(sample_readme)

if __name__ == "__main__":
    print("=== Car Dataset Setup ===")
    download_stanford_cars_dataset()
    setup_sample_structure()
    print("\nSetup complete! Check dataset/README.md for next steps.")