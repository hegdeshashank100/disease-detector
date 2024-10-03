from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import logging

# Initialize Flask app
app = Flask(__name__)

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model
MODEL_PATH = r'C:\Users\hegde\Desktop\diseas detector\model\skin_model.keras'  # Updated to .keras
model = tf.keras.models.load_model(MODEL_PATH)

# Folder to save uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image preprocessing function with exception handling
def prepare_image(img_path):
    try:
        img = Image.open(img_path)

        # Ensure the image has 3 channels (convert to RGB)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = img.resize((224, 224))  # Resize the image to match the model input
        img = np.array(img) / 255.0   # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return None  # Return None if there's an error

# Class mapping for predictions
disease_map = {
    0: "Acne and Rosacea ",
    1: "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    2: "Atopic Dermatitis",
    3: "Cellulitis Impetigo and other Bacterial Infections",
    4: "Eczema",
    5: "Exanthems and Drug Eruptions",
    6: "Herpes HPV and other STDs",
    7: "Light Diseases and Disorders of Pigmentation",
    8: "Lupus and other Connective Tissue diseases",
    9: "Melanoma Skin Cancer Nevi and Moles",
    10: "Poison Ivy Photos and other Contact Dermatitis",
    11: "Psoriasis pictures Lichen Planus and related diseases",
    12: "Seborrheic Keratoses and other Benign Tumors",
    13: "Systemic Disease",
    14: "Tinea Ringworm Candidiasis and other Fungal Infections",
    15: "Urticaria Hives",
    16: "Vascular Tumors",
    17: "Vasculitis",
    18: "Warts Molluscum and other Viral Infections",
}

# Define the root route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Define API route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        logging.warning("No file part in request.")
        return redirect(url_for('index'))  # Redirect to index if no file

    file = request.files['file']

    if file.filename == '':
        logging.warning("No file selected.")
        return redirect(url_for('index'))  # Redirect to index if no file selected

    if file and allowed_file(file.filename):
        # Save uploaded image
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        try:
            file.save(file_path)
        except Exception as e:
            logging.error(f"Error saving file: {e}")
            return redirect(url_for('index'))

        # Prepare image and make prediction
        img = prepare_image(file_path)
        if img is None:
            return redirect(url_for('index'))

        try:
            predictions = model.predict(img)
            predicted_class = np.argmax(predictions)
            result = disease_map.get(predicted_class, "Unknown")
            confidence = float(np.max(predictions))
        except Exception as e:
            logging.error(f"Error making prediction: {e}")
            return redirect(url_for('index'))

        # Cleanup uploaded files after processing (optional)
        os.remove(file_path)  # Remove the uploaded file after prediction

        # Render the result in a new tab
        return render_template('result.html', prediction=result, confidence=confidence)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
