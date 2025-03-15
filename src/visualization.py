# src/visualization.py (custom snippet for custom augmentation)
import matplotlib.pyplot as plt
import numpy as np
import cv2
from src.custom_augmentation import get_custom_augmentation, apply_custom_augmentation

def visualize_custom_comparison(images, num_samples=5, zoom_factor=1.0):
    """
    Visualizes a subset of images alongside their augmented versions using Albumentations.
    """
    import random
    transform = get_custom_augmentation()
    indices = random.sample(range(len(images)), num_samples)
    plt.figure(figsize=(15 * zoom_factor, 6 * zoom_factor))
    
    for i, idx in enumerate(indices):
        # Original image (assumed RGB)
        orig_image = images[idx]
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(orig_image)
        plt.title("Original", fontsize=10 * zoom_factor)
        plt.axis("off")
        
        # Convert image to uint8 then to BGR for Albumentations, apply transform, then convert back to RGB.
        orig_image_uint8 = (orig_image * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(orig_image_uint8, cv2.COLOR_RGB2BGR)
        augmented_bgr = apply_custom_augmentation(image_bgr, transform)
        augmented_rgb = cv2.cvtColor(augmented_bgr, cv2.COLOR_BGR2RGB)
        
        # Show augmented image
        plt.subplot(2, num_samples, i + num_samples + 1)
        plt.imshow(augmented_rgb)
        plt.title("Augmented", fontsize=10 * zoom_factor)
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()



