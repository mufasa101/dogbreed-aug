class ImageAugmentationApp:
    def __init__(self, parent_folder):
        self.parent_folder = parent_folder
        self.train_images = []
        self.train_labels = []
        self.test_images = []
        self.test_labels = []
        self.zoom_factor = 1.0
        self.cancel_requested = False
        self.datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    
    def load_images(self):
        # Clear current lists
        self.train_images, self.train_labels = [], []
        self.test_images, self.test_labels = [], []
        # Traverse directories and load images with proper error handling
        # Implement cancellation check inside the loop
        # Return results or store in instance variables
        pass
    
    def visualize_comparison(self, num_samples=5):
        # Use fixed random indices to ensure consistency between original and augmented images
        # Apply the current zoom factor to visualization
        pass
    
    def update_zoom(self, delta):
        self.zoom_factor = max(0.4, self.zoom_factor + delta)
    
    def cancel(self):
        self.cancel_requested = True








from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Define an ImageDataGenerator with the required augmentations
datagen = ImageDataGenerator(
    rotation_range=30,            # Rotate images up to 30 degrees
    width_shift_range=0.2,        # Shift images horizontally by 20%
    height_shift_range=0.2,       # Shift images vertically by 20%
    shear_range=0.2,              # Shear transformation by 20%
    zoom_range=0.2,               # Zoom in/out by 20%
    horizontal_flip=True,         # Flip images horizontally
    fill_mode='nearest'           # Fill empty areas after transformation
)

# Function to visualize a single image's augmentations
def visualize_augmentations(image, num_examples=5):
    # Convert image to 4D tensor for the generator
    image = np.expand_dims(image, axis=0)
    plt.figure(figsize=(15, 3))
    for i, batch in enumerate(datagen.flow(image, batch_size=1)):
        plt.subplot(1, num_examples, i + 1)
        plt.imshow(batch[0])
        plt.title(f'Augmentation {i+1}')
        plt.axis('off')
        if i >= num_examples - 1:
            break
    plt.show()

# Example usage: Load an image (ensure it is normalized and resized as needed)
# For demonstration, we assume 'sample_image' is a numpy array with shape (height, width, 3)
# sample_image = ... (load your image here)
# visualize_augmentations(sample_image)
