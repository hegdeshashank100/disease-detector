import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import EfficientNetB0  # Using a pre-trained model
from tensorflow.keras import layers, models
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages

# Define dataset directories
train_dir = r'C:\Users\hegde\Desktop\diseas detector\dataset\train'
validation_dir = r'C:\Users\hegde\Desktop\diseas detector\dataset\validation'

# Define parameters
batch_size = 32  # Increased batch size for better training efficiency
img_height = 224
img_width = 224
epochs = 50  # Increased number of epochs for better training

# Image data generators with advanced augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,  # Increased rotation range
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],  # Random brightness adjustment
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load training and validation datasets
train_data_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # Use categorical since there are multiple classes
    shuffle=True,  # Shuffle training data
)

val_data_gen = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # Use categorical since there are multiple classes
    shuffle=False,  # Validation data should not be shuffled
)

# Build the model using transfer learning with EfficientNetB0
base_model = EfficientNetB0(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')

# Freeze the base model layers
base_model.trainable = False

# Create a new model on top of the base model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),  # Dropout to reduce overfitting
    layers.Dense(256, activation='relu'),
    layers.Dense(19, activation='softmax')  # Updated to 19 classes
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate for fine-tuning
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint('model/best_skin_model.keras', monitor='val_accuracy', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Early stopping

# Train the model
history = model.fit(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // batch_size,
    validation_data=val_data_gen,
    validation_steps=val_data_gen.samples // batch_size,
    epochs=epochs,
    callbacks=[checkpoint, early_stopping]  # Save the best model and stop early if no improvement
)

# Save the final trained model
model.save('model/skin_model.keras')  # Use .keras format for saving
