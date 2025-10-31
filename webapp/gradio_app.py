"""
Car Brand Recognition with Gradio
An alternative web interface using Gradio for car brand classification.
"""

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from pathlib import Path

def load_model():
    """Load the trained car brand classification model."""
    model_path = "../models"
    
    # Try to find the latest model
    model_files = list(Path(model_path).glob("*.h5"))
    
    if not model_files:
        print("No trained model found! Please train a model first.")
        return None, None
    
    # Load the most recent model
    latest_model = max(model_files, key=lambda x: x.stat().st_ctime)
    
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
        print(f"Error loading model: {e}")
        return None, None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess the uploaded image for model prediction."""
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

def predict_car_brand(image):
    """
    Predict the car brand from an uploaded image.
    
    Args:
        image: PIL Image object
    
    Returns:
        Dictionary with brand predictions and confidence scores
    """
    if image is None:
        return "Please upload an image of a car."
    
    # Load model (you might want to cache this globally)
    model, model_info = load_model()
    if model is None:
        return "Model not available. Please train a model first."
    
    # Default class names (update based on your trained model)
    class_names = ["audi", "bmw", "ford", "honda", "mercedes", "nissan", "tesla", "toyota"]
    
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Get probabilities for all classes
        class_probabilities = {
            class_names[i].capitalize(): float(predictions[0][i]) 
            for i in range(len(class_names))
        }
        
        return class_probabilities
    
    except Exception as e:
        return f"Error during prediction: {str(e)}"

def create_gradio_interface():
    """Create and configure the Gradio interface."""
    
    # Define the interface
    interface = gr.Interface(
        fn=predict_car_brand,
        inputs=gr.Image(type="pil", label="Upload Car Image"),
        outputs=gr.Label(num_top_classes=8, label="Car Brand Predictions"),
        title="🚗 Car Brand Recognition",
        description="""
        Upload an image of a car and get predictions for the car brand!
        
        This model uses deep learning to classify cars into different brands:
        Audi, BMW, Ford, Honda, Mercedes, Nissan, Tesla, Toyota
        
        **Tips for best results:**
        - Use clear, well-lit images
        - Ensure the car is the main subject
        - Side or front views work best
        """,
        examples=[
            # You can add example images here once you have some
            # ["path/to/example1.jpg"],
            # ["path/to/example2.jpg"],
        ],
        theme=gr.themes.Soft(),
        allow_flagging="never"
    )
    
    return interface

def main():
    """Main function to launch the Gradio app."""
    print("🚗 Starting Car Brand Recognition App with Gradio...")
    
    # Check if model exists
    model, model_info = load_model()
    if model is None:
        print("⚠️  No trained model found!")
        print("Please run 'python src/train_model.py' first to train a model.")
        return
    
    print("✅ Model loaded successfully!")
    if model_info:
        print(f"Model: {model_info.get('model_name', 'Unknown')}")
        print(f"Classes: {model_info.get('num_classes', 'Unknown')}")
    
    # Create and launch interface
    interface = create_gradio_interface()
    
    # Launch the app
    interface.launch(
        share=False,  # Set to True if you want a public link
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )

if __name__ == "__main__":
    main()