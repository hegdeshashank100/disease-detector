import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Load the pre-trained model
MODEL_PATH = 'model/skin_model.h5'  # Update the path as necessary
model = tf.keras.models.load_model(MODEL_PATH)

# Image preprocessing function
def prepare_image(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))  # Resize the image to match the model input
    img = np.array(img) / 255.0   # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict disease
def predict_disease(image_path):
    # Prepare image
    img = prepare_image(image_path)
    
    # Make prediction
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    
    # Class mapping (update with your actual classes)
    disease_map = {0: "Eczema", 1: "Healthy Skin", 2: "Other Disease"}  # Add more diseases as necessary
    result = disease_map.get(predicted_class, "Unknown")
    confidence = float(np.max(predictions))

    return result, confidence

# Main function to execute prediction
if __name__ == "__main__":
    # Replace 'path_to_your_image.jpg' with the actual image path you want to test
    image_path = r'download.jpeg'  
    
    if os.path.exists(image_path):
        prediction, confidence = predict_disease(image_path)
        print(f"Predicted Disease: {prediction}, Confidence: {confidence:.2f}")
    else:
        print(f"Error: The image file '{image_path}' does not exist.")
