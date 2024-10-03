# a.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# Define image dimensions
img_height = 224  # Set the height of your input images
img_width = 224   # Set the width of your input images

# Define your image data generator
train_datagen = ImageDataGenerator(
    rescale=1.0/255,  # Rescale pixel values to [0, 1]
    rotation_range=20,  # Randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # Randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # Randomly shift images vertically (fraction of total height)
    shear_range=0.2,  # Shear angle in counter-clockwise direction in degrees
    zoom_range=0.2,  # Randomly zoom into images
    horizontal_flip=True,  # Randomly flip images
    fill_mode='nearest'  # Fill in new pixels after a transformation
)

# Define your directory paths
train_dir = 'path_to_train_directory'  # Replace with your train directory path
val_dir = 'path_to_validation_directory'  # Replace with your validation directory path

# Create generators for training and validation data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),  # Use the defined height and width
    batch_size=32,  # Adjust batch size as necessary
    class_mode='categorical'  # Use categorical for multi-class classification
)

val_generator = train_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),  # Use the defined height and width
    batch_size=32,
    class_mode='categorical'
)

# Add your model training code here
