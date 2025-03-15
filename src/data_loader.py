import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def load_images(parent_folder, image_size=(128, 128), max_folders=None):
    """
    Loads images from the dataset.
    Checks for a nested 'Images' folder and iterates over each breed folder.
    Optionally limits the number of breed folders processed (if max_folders is provided).
    Splits the dataset into training and testing sets (80/20 split).

    Returns:
        train_images, train_labels, test_images, test_labels as numpy arrays.
    """
    # Start at the 'images' folder inside parent_folder.
    images_dir = os.path.join(parent_folder, 'images')
    # Check for a nested "Images" folder.
    nested_folder = os.path.join(images_dir, 'Images')
    if os.path.exists(nested_folder) and os.path.isdir(nested_folder):
        images_dir = nested_folder

    all_images, all_labels = [], []

    # Get list of breed folders
    breed_folders = os.listdir(images_dir)
    # Limit the number of folders if max_folders is provided
    if max_folders is not None:
        breed_folders = breed_folders[:max_folders]

    for breed in breed_folders:
        breed_path = os.path.join(images_dir, breed)
        if os.path.isdir(breed_path):
            print(f"Loading images for breed: {breed}")
            for filename in tqdm(os.listdir(breed_path), desc=f"Loading {breed} images"):
                file_path = os.path.join(breed_path, filename)
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        img = Image.open(file_path).convert('RGB')
                        img = img.resize(image_size)
                        img_array = np.array(img) / 255.0  # Normalize pixel values
                        if img_array.shape == (image_size[0], image_size[1], 3):
                            all_images.append(img_array)
                            all_labels.append(breed)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

    print(f"Total images loaded: {len(all_images)}")
    
    if len(all_images) == 0:
        raise ValueError("No images loaded. Please check your dataset path and structure.")
    
    # Split data into training and testing sets (80/20 split)
    train_images, test_images, train_labels, test_labels = train_test_split(
        np.array(all_images), np.array(all_labels), test_size=0.2, random_state=42, stratify=all_labels
    )
    
    return train_images, train_labels, test_images, test_labels
