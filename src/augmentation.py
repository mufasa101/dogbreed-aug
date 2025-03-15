from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_augmentation_generator():
    """
    Returns an ImageDataGenerator configured with advanced augmentations.
    """
    datagen = ImageDataGenerator(
        rotation_range=30,            # Rotate images up to 30 degrees
        width_shift_range=0.2,        # Shift horizontally up to 20%
        height_shift_range=0.2,       # Shift vertically up to 20%
        shear_range=0.2,              # Shear transformation up to 20%
        zoom_range=0.2,               # Zoom in/out by up to 20%
        horizontal_flip=True,         # Flip images horizontally
        fill_mode='nearest'           # Fill missing pixels with nearest neighbor
    )
    return datagen
