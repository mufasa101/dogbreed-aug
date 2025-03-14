

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import random




import kagglehub

# Download latest version
path = kagglehub.dataset_download("jessicali9530/stanford-dogs-dataset")

print("Path to dataset files:", path)

image_dir = os.path.join(path, 'images/Images')

# Verify the corrected path
print("Updated Images directory path:", image_dir)
print("Images available:", os.listdir(image_dir)[:5])  # Show first 5 images







import shutil
from sklearn.model_selection import train_test_split

import shutil
from sklearn.model_selection import train_test_split

# Define paths for train and validation datasets
train_dir = os.path.join(path, 'train')
val_dir = os.path.join(path, 'val')

# Ensure directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get all breed folders
breed_folders = os.listdir(image_dir)

for breed in breed_folders:
    breed_path = os.path.join(image_dir, breed)
    if not os.path.isdir(breed_path):
        continue  # Skip non-folder files

    # Create corresponding breed subfolders in train and val
    train_breed_path = os.path.join(train_dir, breed)
    val_breed_path = os.path.join(val_dir, breed)
    os.makedirs(train_breed_path, exist_ok=True)
    os.makedirs(val_breed_path, exist_ok=True)

    # Get all images in the breed folder
    image_files = [f for f in os.listdir(breed_path) if f.endswith(('.jpg', '.png'))]

    # Split into train (80%) and val (20%)
    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

    # Move images into their respective folders
    for file in train_files:
        shutil.copy(os.path.join(breed_path, file), os.path.join(train_breed_path, file))

    for file in val_files:
        shutil.copy(os.path.join(breed_path, file), os.path.join(val_breed_path, file))

print("Dataset successfully split into training and validation sets!")






# Image augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1.0/255,  # Normalize pixel values
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# No augmentation for validation data (just rescale)
val_datagen = ImageDataGenerator(rescale=1.0/255)

# Load the images from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Resize images
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Get class indices
class_indices = train_generator.class_indices
print("Class indices:", class_indices)


import matplotlib.pyplot as plt

# Function to display original and augmented images
def visualize_augmentation(generator):
    images, labels = next(generator)  # Get a batch from the generator
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    for i in range(5):
        # Original Image
        axes[0, i].imshow(images[i])
        axes[0, i].axis('off')
        axes[0, i].set_title("Original")

        # Get another batch of augmented images
        aug_images, _ = next(generator)  # Use `next(generator)` instead of `.next()`
        axes[1, i].imshow(aug_images[i])
        axes[1, i].axis('off')
        axes[1, i].set_title("Augmented")

    plt.show()

# Show the images
visualize_augmentation(train_generator)




import tensorflow as tf
from tensorflow.keras import layers, models

# Define CNN Model
def build_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(train_generator.class_indices), activation='softmax')  # Output layer
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Create Model
cnn_model = build_cnn()
cnn_model.summary()



history_original = cnn_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10  # Train for 10 epochs
)

model_augmented = build_cnn(num_classes)

history_augmented = model_augmented.fit(
    augmented_train_generator,
    epochs=10,
    validation_data=val_generator
)


import matplotlib.pyplot as plt

# Function to plot training results
def plot_training(history, title):
    plt.figure(figsize=(12, 4))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Plot original dataset training results
plot_training(history_original, "Original Data")

# Plot augmented dataset training results
plot_training(history_augmented, "Augmented Data")
