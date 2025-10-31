"""
Car Brand Recognition Web Application
A Streamlit web app for car brand classification using the trained model.
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Car Brand Recognition",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """Load the trained car brand classification model."""
    model_path = "../models"
    
    # Try to find the latest model
    model_files = list(Path(model_path).glob("*.h5"))
    
    if not model_files:
        st.error("No trained model found! Please train a model first.")
        return None, None
    
    # Load the most recent model
    latest_model = max(model_files, key=os.path.getctime)
    
    try:
        model = tf.keras.models.load_model(latest_model)
        
        # Try to load model info
        info_file = str(latest_model).replace('.h5', '_info.json')
        model_info = {}
        if Path(info_file).exists():
            with open(info_file, 'r') as f:
                model_info = json.load(f)
        
        return model, model_info
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the uploaded image for model prediction.
    
    Args:
        image: PIL Image object
        target_size: Target size for resizing
    
    Returns:
        Preprocessed image array
    """
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_car_brand(model, image, class_names):
    """
    Predict the car brand from an image.
    
    Args:
        model: Trained Keras model
        image: Preprocessed image array
        class_names: List of class names
    
    Returns:
        Predicted class and confidence scores
    """
    # Make prediction
    predictions = model.predict(image)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    # Get all probabilities
    class_probabilities = {
        class_names[i]: float(predictions[0][i]) 
        for i in range(len(class_names))
    }
    
    # Sort by confidence
    sorted_predictions = sorted(
        class_probabilities.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    return class_names[predicted_class_idx], confidence, sorted_predictions

def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title("🚗 Car Brand Recognition")
    st.markdown("""
    This application uses a deep learning model to classify car brands from images.
    Upload an image of a car, and the model will predict the brand!
    """)
    
    # Load model
    with st.spinner("Loading model..."):
        model, model_info = load_model()
    
    if model is None:
        st.stop()
    
    # Display model information in sidebar
    st.sidebar.header("Model Information")
    if model_info:
        st.sidebar.json(model_info)
    else:
        st.sidebar.write("Model loaded successfully!")
    
    # Default class names (can be updated based on your dataset)
    default_classes = ["audi", "bmw", "ford", "honda", "mercedes", "nissan", "tesla", "toyota"]
    
    # File uploader
    st.header("Upload Car Image")
    uploaded_file = st.file_uploader(
        "Choose a car image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a car (JPG, JPEG, or PNG format)"
    )
    
    if uploaded_file is not None:
        # Create two columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            # Make prediction
            with st.spinner("Analyzing image..."):
                try:
                    # Preprocess image
                    processed_image = preprocess_image(image)
                    
                    # Predict
                    predicted_brand, confidence, all_predictions = predict_car_brand(
                        model, processed_image, default_classes
                    )
                    
                    # Display results
                    st.success(f"Predicted Car Brand: **{predicted_brand.upper()}**")
                    st.write(f"Confidence: **{confidence:.2%}**")
                    
                    # Show confidence meter
                    st.progress(confidence)
                    
                    # Display all predictions
                    st.subheader("All Predictions:")
                    for brand, prob in all_predictions:
                        st.write(f"**{brand.capitalize()}**: {prob:.2%}")
                        st.progress(prob)
                
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
    
    # Example images section
    st.header("Example Usage")
    st.markdown("""
    ### Tips for best results:
    - Use clear, well-lit images of cars
    - Ensure the car is the main subject in the image
    - Avoid heavily cropped or distorted images
    - The model works best with side or front views of cars
    
    ### Supported Car Brands:
    """)
    
    # Display supported brands in a nice format
    cols = st.columns(4)
    brands = ["Audi", "BMW", "Ford", "Honda", "Mercedes", "Nissan", "Tesla", "Toyota"]
    for i, brand in enumerate(brands):
        with cols[i % 4]:
            st.write(f"🚗 {brand}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Created for DAT158 Machine Learning Course**
    
    This application demonstrates the practical deployment of a machine learning model
    for car brand recognition using transfer learning and deep neural networks.
    """)

if __name__ == "__main__":
    main()