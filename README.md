## Group 6

# Multimedia Applications: Image Augmentation for Enhanced Machine Learning

## Table of Contents

- [Overview](#overview)
- [Dataset Selection and Justification](#dataset-selection-and-justification)
- [Installation and Environment Setup](#installation-and-environment-setup)
- [Project Structure](#project-structure)
- [Data Preprocessing and Augmentation](#data-preprocessing-and-augmentation)
- [Model Architecture and Training](#model-architecture-and-training)
- [Results and Analysis](#results-and-analysis)
- [Usage Instructions](#usage-instructions)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project demonstrates a suite of **advanced image augmentation techniques** using a real-world **Kaggle dog breed dataset**. Our goal is to highlight how augmentation can significantly improve model robustness when dealing with high-resolution images that exhibit varying lighting, orientation, and scale conditions.

---

## Dataset Selection and Justification

- **Dataset Chosen:** [Kaggle Dog Breeds Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)
- **Why Dog Breeds?**
  - **Realistic Complexity:** Large, high-resolution images with diverse backgrounds and poses.
  - **Fine-Grained Classification:** Forces the model to distinguish subtle differences among visually similar classes.
  - **Multimedia Relevance:** Reflects authentic use cases where image variety is common.

---

## Installation and Environment Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/mufasa101/dogbreed-aug.git
   cd image_aug
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

### Data Setup

We use the **Stanford Dogs Dataset** from Kaggle. Follow these steps to download and set up the dataset:

1. **Install the Kaggle API:**
   ```bash
   pip install kaggle
   ```
2. **Obtain Kaggle API Credentials:**
   - Log in to your [Kaggle account](https://www.kaggle.com/).
   - Under your profile icon, go to **Account**.
   - Click **Create New API Token**, which downloads `kaggle.json`.
3. **Place the Credentials File:**
   - **Linux/macOS:**
     ```bash
     mkdir -p ~/.kaggle
     mv ~/Downloads/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```
   - **Windows:**
     1. Create a `.kaggle` folder under `%USERPROFILE%`.
     2. Move `kaggle.json` there.
     3. Adjust file permissions so only you can read the file.
4. **Download the Dataset:**
   ```bash
   kaggle datasets download -d jessicali9530/stanford-dogs-dataset
   ```
5. **Extract the Dataset:**
   ```bash
   mkdir -p data
   unzip stanford-dogs-dataset.zip -d data/
   ```

After this, your `data/` folder will contain the necessary images for training, validation, and testing.

---

## Project Structure

```bash
image_aug/
├── data/                           # Contains downloaded or processed datasets
├── notebooks/                      # Jupyter notebooks for exploration & demos
│   ├── data.ipynb                 # Notebook for exploring & preparing the dataset
│   └── training_experiments.ipynb # Notebook demonstrating model training on original & augmented data
├── src/                            # Source code
│   ├── data_loader.py             # Functions for loading & preprocessing data
│   ├── augmentation.py            # Basic ImageDataGenerator setup
│   ├── custom_augmentation.py     # Albumentations-based augmentations
│   ├── model.py                   # CNN model definition & training routines
│   ├── visualization.py           # Scripts for visualizing original & augmented images
│   └── ui.py                      # Interactive UI components using ipywidgets
├── README.md                      # This file
├── requirements.txt               # Python dependencies
└── LICENSE                        # License information
```

---

## Data Preprocessing and Augmentation

### Preprocessing

- **Loading & Splitting:** Our `load_images` function splits data into training and test sets, optionally limiting the number of folders for quicker demos.
- **Resizing & Normalizing:** Images are resized (128×128) and normalized to [0,1].
- **Error Handling:** Skips corrupt files gracefully to avoid pipeline crashes.

### Augmentation Techniques

We demonstrate **two** augmentation approaches:

1. **Keras/TensorFlow `ImageDataGenerator`:**

   - Rotation (±30°), width/height shifts, shear (±20%), zoom (±20%), and horizontal flips.
   - Ideal for quick, on-the-fly augmentation in model training.

2. **Albumentations (Custom Pipeline):**
   - Used primarily for visualization.
   - Offers more advanced transformations (e.g., brightness, contrast, random noise), giving us deeper insight into how each transformation alters an image.

---

## Model Architecture and Training

### Architecture

A **Convolutional Neural Network** with:

- **Convolution + Pooling Blocks:** Incrementally deeper layers to learn complex features.
- **Dropout (50%):** Reduces overfitting by randomly dropping neurons.
- **Fully Connected Layer:** Adjusted to the number of classes, concluding with a softmax activation for classification.

### Training

1. **Original Data Model:**

   - Baseline training on unaugmented images.
   - Typically reveals signs of overfitting or lower generalization.

2. **Augmented Data Model:**
   - Trains on data generated via `ImageDataGenerator` (or custom Albumentations, if desired).
   - Often yields better generalization and reduced overfitting.

We track training/validation accuracy and loss over epochs to compare performance.

---

## Results and Analysis

1. **Performance Curves:**

   - Plots show how the **original** vs. **augmented** models learn differently.
   - Augmentation usually narrows the gap between training and validation accuracy.

2. **Confusion Matrix:**

   - Highlights class-by-class performance, revealing which breeds are often confused.
   - Demonstrates whether augmentation helps improve recognition for challenging classes.

3. **Visual Comparisons:**
   - Original vs. augmented images displayed side-by-side.
   - Shows how transformations like rotation, shift, and zoom enrich the training set.

**Key Takeaway:** Data augmentation reduces overfitting and boosts the model’s ability to handle real-world variations, making it a critical step for high-resolution, fine-grained classification tasks like dog breed identification.

---

## Usage Instruction

1. **Data Acquisition:**  
   Follow the [Data Setup](#data-setup) instructions to download and organize the Stanford Dogs Dataset.

Below is an example of how you might **introduce and document** a separate notebook called **`data.ipynb`** in your project. This notebook can serve as a dedicated space for **data exploration, cleaning, and preprocessing** steps, keeping your workflow organized and easy to follow.

---

## **`data.ipynb`** – Data Exploration and Preprocessing

This notebook is designed to **explore** and **prepare** the dataset before training any models. It allows you to visually inspect images, check label distributions, and apply basic preprocessing or cleaning steps.

### **Notebook Overview**

1. **Dataset Inspection:**

   - Examine folder structure and verify that images are organized as expected.
   - Print sample file paths, display random images, and confirm they’re properly labeled.
   - Check for potential issues like missing or corrupt files.

2. **EDA (Exploratory Data Analysis):**

   - Plot the distribution of classes (e.g., how many images per breed).
   - Identify imbalances or underrepresented classes.
   - Possibly visualize basic statistics (image dimensions, aspect ratios, etc.).

3. **Preprocessing:**

   - Resize images to a consistent shape (e.g., 128×128).
   - Normalize pixel values to a 0–1 range or other standardization.
   - (Optional) Crop or remove unnecessary margins if needed.

4. **Splitting or Reorganizing:**

   - If your dataset isn’t already split into training and testing sets, do so here (using `train_test_split` or a custom approach).
   - Move or copy files into `train/` and `val/` directories, or handle them with code in `src/data_loader.py`.

5. **Documentation of Findings:**
   - Record any peculiarities discovered, such as corrupt images or mislabeled samples.
   - Suggest strategies for dealing with heavily imbalanced classes or low-quality images.

### **Why Have a Separate `data.ipynb`?**

- **Cleaner Workflow:** By isolating data exploration and cleaning tasks in one notebook, you keep your main training notebook (`training_experiments.ipynb`) focused on model-related steps.
- **Reproducibility:** Anyone can open `data.ipynb` to understand how you prepared the dataset before training.
- **Debugging:** If there are discrepancies in the data (e.g., class distribution doesn’t match expectations), you can revisit this notebook to pinpoint where the process might have gone awry.

### **Usage Instructions**

1. **Open `data.ipynb`:**
   ```bash
   jupyter notebook notebooks/data.ipynb
   ```
2. **Run Cells in Order:**  
   Start from the top to load images, examine distributions, and perform any required cleaning or splitting.
3. **Confirm Outputs:**  
   Check that the final distribution of images aligns with your project’s needs. If you’re limiting folders or images for demonstration, verify that the code handles these cases gracefully.

### **Next Steps**

Once you’ve verified and preprocessed the data in `data.ipynb`, you can move on to:

- **`training_experiments.ipynb`:** For model training on original vs. augmented data.
- **`ui.py` / Interactive Visualization:** For real-time augmentation demos and image comparisons.

---

### **Training & Experiments:**

Open and run the notebook in `notebooks/training_experiments.ipynb` to:

- Train the model on **original data**.
- Train the model on **augmented data**.
- Compare the two training runs with performance plots and confusion matrices.

### **Visualization:**

- Use `src/visualization.py` or the interactive UI (`src/ui.py`) to visualize original vs. augmented images.
- Experiment with `custom_augmentation.py` (Albumentations) for a more advanced augmentation pipeline.

### **Interactive UI (Optional):**

- Run `%run -i src/ui.py` in a Jupyter cell to load the dataset, visualize images, and adjust augmentation parameters on the fly.

---

## Contributing

We welcome feedback and contributions! Whether you’d like to refine the augmentation pipeline, experiment with new model architectures, or improve documentation, feel free to open an issue or submit a pull request. Please ensure your changes are well-documented and tested.

---

## License

This project is distributed under the [MIT License](LICENSE). You’re free to use, modify, and distribute this code, provided you include proper attribution.

---

**Group 6** thanks you for exploring **Image Augmentation for Enhanced Machine Learning**. If you encounter any issues or have suggestions for improvement, please open an issue or reach out via our discussion boards. We look forward to your contributions!
