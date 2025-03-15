# src/custom_augmentation.py
import albumentations as A
import cv2

def get_custom_augmentation():
    """
    Returns an Albumentations Compose object configured with advanced augmentations.
    """
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.MotionBlur(blur_limit=3, p=0.2),
        # You can add more transformations here...
    ], p=1.0)
    
    return transform

def apply_custom_augmentation(image, transform):
    """
    Applies the given Albumentations transformation to an image.
    Expects image in numpy array format (BGR for OpenCV).
    Returns the augmented image.
    """
    # Albumentations works with images as numpy arrays (BGR order)
    augmented = transform(image=image)
    return augmented['image']
