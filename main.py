import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import random
import cv2


from google.colab import files
files.upload()  # Upload your kaggle.json when prompted


import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("jessicali9530/stanford-dogs-dataset")

print("Path to dataset files:", path)

# Let's say we only want 10 specific breeds for a smaller dataset
selected_breeds = [
    "n02085936-Maltese_dog",
    "n02086079-Pekinese",
    "n02086240-Shih-Tzu",
    "n02086910-papillon",
    "n02087046-toy_terrier",
    "n02087394-Rhodesian_ridgeback",
    "n02088094-Afghan_hound",
    "n02088238-basset",
    "n02088364-beagle",
    "n02088632-bluetick"
]

image_dir = os.path.join(path, 'images/Images')

# Verify the corrected path
print("Updated Images directory path:", image_dir)
print("Images available:", os.listdir(image_dir)[:10])  # Show first 10 images

# 3) Create a smaller dataset directory
smaller_dataset_dir = os.path.join(path, "smaller_dataset")
os.makedirs(smaller_dataset_dir, exist_ok=True)

# ** Copy only the selected breed folders to smaller_dataset_dir **
for breed in selected_breeds:
    source_breed_dir = os.path.join(image_dir, breed)
    dest_breed_dir = os.path.join(smaller_dataset_dir, breed)
    shutil.copytree(source_breed_dir, dest_breed_dir) 
    # copytree copies an entire folder recursively



# Define paths for train and validation datasets
train_small_dir = os.path.join(path, 'train_small')
val_small_dir   = os.path.join(path, 'val_small')


# Ensure directories exist
os.makedirs(train_small_dir, exist_ok=True)
os.makedirs(val_small_dir, exist_ok=True)






import shutil
from sklearn.model_selection import train_test_split


# 5) Split each breed’s images into train (80%) and val (20%)
breed_folders = [
    f for f in os.listdir(smaller_dataset_dir)
    if os.path.isdir(os.path.join(smaller_dataset_dir, f))
]

for breed in breed_folders:
    source_breed_path = os.path.join(smaller_dataset_dir, breed)
    images = [img for img in os.listdir(source_breed_path) if img.endswith(('.jpg','.jpeg','.png'))]

    # 80% train, 20% val
    train_files, val_files = train_test_split(images, test_size=0.2, random_state=42)

    train_breed_path = os.path.join(train_small_dir, breed)
    val_breed_path = os.path.join(val_small_dir, breed)
    os.makedirs(train_breed_path, exist_ok=True)
    os.makedirs(val_breed_path, exist_ok=True)

    # Copy train images
    for img_file in train_files:
        shutil.copy(os.path.join(source_breed_path, img_file),
                    os.path.join(train_breed_path, img_file))

    # Copy val images
    for img_file in val_files:
        shutil.copy(os.path.join(source_breed_path, img_file),
                    os.path.join(val_breed_path, img_file))







import os
import kagglehub

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split



# 3) Define your data generators
original_train_datagen = ImageDataGenerator(rescale=1.0/255)
augmented_train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

seed_val = 42
original_train_generator = original_train_datagen.flow_from_directory(
    train_small_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
    seed=seed_val
)

augmented_train_generator = augmented_train_datagen.flow_from_directory(
    train_small_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
    seed=seed_val
)

val_generator = val_datagen.flow_from_directory(
    val_small_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
print("\n")
print("Class indices:", original_train_generator.class_indices)
print("original_train_generator classes:", original_train_generator.class_indices)
print("augmented_train_generator classes:", augmented_train_generator.class_indices)

print("Val classes:", val_generator.class_indices)










import matplotlib.pyplot as plt

def visualize_augmentation(orig_gen, aug_gen, num_images=5):
    """
    Displays num_images from orig_gen (top row) and the corresponding
    augmented images from aug_gen (bottom row).
    Both generators should be aligned (same seed, shuffle=False).
    """
    # Fetch one batch from each generator
    orig_images, orig_labels = next(orig_gen)
    aug_images, aug_labels = next(aug_gen)

    # Create a figure with 2 rows and num_images columns
    fig, axes = plt.subplots(2, num_images, figsize=(5*num_images, 8))
    fig.suptitle('Original (top) vs. Augmented (bottom)', fontsize=16)

    for i in range(num_images):
        # Top row: original
        axes[0, i].imshow(orig_images[i])
        axes[0, i].axis('off')

        # Bottom row: augmented
        axes[1, i].imshow(aug_images[i])
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

# IMPORTANT: Because we just called next() on each generator,
# they have advanced by one batch. For repeated visualizations
# with the same images, re-initialize the generators or
# set up a separate “preview” generator. For a quick demo:
visualize_augmentation(original_train_generator, augmented_train_generator, num_images=5)







from tensorflow.keras import layers, models

def build_cnn(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model





num_classes = len(original_train_generator.class_indices)

model_original = build_cnn(num_classes)
model_original.summary()

history_original = model_original.fit(
    original_train_generator,
    epochs=10,
    validation_data=val_generator
)



model_augmented = build_cnn(num_classes)

history_augmented = model_augmented.fit(
    augmented_train_generator,
    epochs=10,
    validation_data=val_generator
)






import matplotlib.pyplot as plt

def plot_training(history, title):
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_training(history_original, "Original Data")
plot_training(history_augmented, "Augmented Data")
