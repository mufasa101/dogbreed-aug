import ipywidgets as widgets
from IPython.display import display, clear_output
import os

from src.data_loader import load_images
from src.augmentation import get_augmentation_generator  # May still be used for other purposes
from src.visualization import visualize_custom_comparison

# Global state variables (ideally should be encapsulated in a class)
zoom_factor = 1.0
cancel_requested = False
train_images, train_labels, test_images, test_labels = [], [], [], []
datagen = get_augmentation_generator()  # May be used elsewhere if needed
output = widgets.Output()

# Slider to limit the number of folders (for presentation purposes)
max_folders_slider = widgets.IntSlider(value=10, min=1, max=50, 
                                         description='Max Folders:', 
                                         style={'description_width': 'initial'})

def load_data_action(b):
    """
    Load the dataset from the specified directory, applying a limit on the number of folders.
    """
    global train_images, train_labels, test_images, test_labels
    with output:
        clear_output(wait=True)
        print("Loading dataset...")
        # Use a relative path: from notebooks folder, go up one level to project root and then into "data"
        parent_folder = os.path.abspath(os.path.join(os.getcwd(), "..", "data"))
        max_folders = max_folders_slider.value  # Limit number of breed folders processed
        try:
            train_images, train_labels, test_images, test_labels = load_images(parent_folder, max_folders=max_folders)
            print(f"Loaded {len(train_images)} training images and {len(test_images)} testing images.")
        except Exception as e:
            print(f"Error: {e}")

def visualize_action(b):
    """
    Visualize a subset of the loaded images with the applied custom augmentations.
    """
    global zoom_factor, train_images
    with output:
        clear_output(wait=True)
        if len(train_images) == 0:
            print("Please load the dataset first!")
        else:
            num_samples = num_images_slider.value
            print(f"Visualizing {num_samples} images with zoom factor {zoom_factor}...")
            visualize_custom_comparison(train_images, num_samples=num_samples, zoom_factor=zoom_factor)

def zoom_in_action(b):
    """
    Increase the zoom factor for the visualization.
    """
    global zoom_factor
    zoom_factor += 0.2
    with output:
        print(f"Zoom factor increased to {zoom_factor}")

def zoom_out_action(b):
    """
    Decrease the zoom factor for the visualization, ensuring it doesn't go below a minimum level.
    """
    global zoom_factor
    if zoom_factor > 0.4:
        zoom_factor -= 0.2
        with output:
            print(f"Zoom factor decreased to {zoom_factor}")
    else:
        with output:
            print("Minimum zoom level reached!")

# UI Widgets
load_data_button = widgets.Button(description="Load Dataset", button_style='success')
visualize_button = widgets.Button(description="Visualize Images", button_style='info')
zoom_in_button = widgets.Button(description="Zoom In", button_style='primary')
zoom_out_button = widgets.Button(description="Zoom Out", button_style='primary')
num_images_slider = widgets.IntSlider(value=5, min=1, max=15, description='Images:', style={'description_width': 'initial'})

# Bind button actions to their respective functions
load_data_button.on_click(load_data_action)
visualize_button.on_click(visualize_action)
zoom_in_button.on_click(zoom_in_action)
zoom_out_button.on_click(zoom_out_action)

# Arrange controls in a horizontal box layout
controls = widgets.HBox([load_data_button, visualize_button, zoom_in_button, zoom_out_button, num_images_slider, max_folders_slider])

# Display the controls and the output widget
display(controls, output)



