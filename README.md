# Car Brand Recognition - DAT158 ML Assignment 3

## Project Overview
This project implements a machine learning model for car brand recognition using deep learning techniques. The model can classify car images into different brands (BMW, Tesla, Toyota, etc.).

## Project Structure
```
Dat158-ML-Oblig3/
├── dataset/
│   ├── train/           # Training images organized by brand
│   ├── test/            # Test images for evaluation
│   └── validation/      # Validation images for model tuning
├── models/              # Saved trained models
├── notebooks/           # Jupyter notebooks for exploration and analysis
├── src/                 # Source code for data processing and training
├── webapp/              # Web application for deployment
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Project Phases

### 1. Problem Understanding ✅
- **Goal**: Build a car brand recognition system
- **Input**: Car images
- **Output**: Car brand classification (BMW, Tesla, Toyota, etc.)
- **Approach**: Image classification using deep learning

### 2. Data Collection
- **Dataset**: Stanford Cars Dataset from Kaggle
- **Size**: 16,185 images of 196 car classes
- **Format**: JPG images with varying sizes

### 3. Data Exploration & Preparation
- Image preprocessing (resize, normalize)
- Data augmentation for better generalization
- Train/test/validation split

### 4. Modeling
- **Base Model**: Transfer learning with pre-trained CNN (MobileNetV2/ResNet50)
- **Framework**: TensorFlow/Keras
- **Technique**: Fine-tuning on car dataset

### 5. Evaluation
- Accuracy metrics
- Confusion matrix
- Error analysis

### 6. Deployment
- **Web Interface**: Streamlit/Gradio web app
- **Functionality**: Upload image → Get car brand prediction

### 7. Documentation
- Project report
- Code documentation
- GitHub repository

## Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. **Data Preparation**: Run `src/data_preparation.py`
2. **Model Training**: Run `src/train_model.py`
3. **Web App**: Run `streamlit run webapp/app.py`

## Results
(To be updated after model training)

## Authors
- [Your Name] - DAT158 Machine Learning Course

## License
This project is for educational purposes as part of DAT158 course.