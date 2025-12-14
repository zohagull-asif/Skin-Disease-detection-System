"""
Data Preprocessing Module
Handles image loading, augmentation, and dataset preparation
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2


IMG_SIZE = 224
BATCH_SIZE = 32


def create_data_generators(train_dir, validation_split=0.2, batch_size=BATCH_SIZE):
    """
    Create training and validation data generators with augmentation.

    Args:
        train_dir: Directory containing subdirectories for each class
        validation_split: Fraction of data to use for validation
        batch_size: Batch size for training

    Returns:
        train_generator, validation_generator
    """
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=validation_split
    )

    # Training generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # Validation generator (no augmentation)
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_generator, validation_generator


def create_test_generator(test_dir, batch_size=BATCH_SIZE):
    """
    Create test data generator without augmentation.

    Args:
        test_dir: Directory containing test images
        batch_size: Batch size for testing

    Returns:
        test_generator
    """
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return test_generator


def preprocess_single_image(image_path):
    """
    Preprocess a single image for prediction.

    Args:
        image_path: Path to the image file

    Returns:
        Preprocessed image array ready for model prediction
    """
    # Load image
    img = Image.open(image_path)

    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize
    img = img.resize((IMG_SIZE, IMG_SIZE))

    # Convert to numpy array
    img_array = np.array(img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def preprocess_uploaded_image(uploaded_file):
    """
    Preprocess an uploaded image file (for Streamlit).

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        Preprocessed image array ready for model prediction
    """
    # Load image from uploaded file
    img = Image.open(uploaded_file)

    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize
    img = img.resize((IMG_SIZE, IMG_SIZE))

    # Convert to numpy array
    img_array = np.array(img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, img


def apply_hair_removal(image):
    """
    Apply hair removal preprocessing using morphological operations.
    Useful for dermoscopy images with hair artifacts.

    Args:
        image: Input image (numpy array)

    Returns:
        Image with hair artifacts removed
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply blackhat morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Apply thresholding
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Inpaint the hair regions
    result = cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)

    return result


def enhance_image(image):
    """
    Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        image: Input image (numpy array)

    Returns:
        Enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Split channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Merge channels
    lab = cv2.merge([l, a, b])

    # Convert back to RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return enhanced


def segment_lesion(image):
    """
    Segment the skin lesion from the background.

    Args:
        image: Input image (numpy array)

    Returns:
        Segmented image with lesion highlighted
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Apply mask to original image
    result = cv2.bitwise_and(image, image, mask=thresh)

    return result, thresh


def get_class_weights(train_generator):
    """
    Calculate class weights for imbalanced datasets.

    Args:
        train_generator: Training data generator

    Returns:
        Dictionary of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight

    # Get all labels
    labels = train_generator.classes

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )

    # Convert to dictionary
    class_weight_dict = dict(enumerate(class_weights))

    return class_weight_dict


def load_dataset_info(data_dir):
    """
    Load and display dataset information.

    Args:
        data_dir: Directory containing the dataset

    Returns:
        Dictionary with dataset statistics
    """
    stats = {}
    total_images = 0

    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return stats

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            num_images = len([f for f in os.listdir(class_path)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            stats[class_name] = num_images
            total_images += num_images
            print(f"  {class_name}: {num_images} images")

    print(f"\nTotal images: {total_images}")
    stats['total'] = total_images

    return stats


if __name__ == "__main__":
    # Test preprocessing functions
    print("Data Preprocessing Module")
    print("=" * 50)
    print("\nAvailable functions:")
    print("- create_data_generators(train_dir)")
    print("- create_test_generator(test_dir)")
    print("- preprocess_single_image(image_path)")
    print("- preprocess_uploaded_image(uploaded_file)")
    print("- apply_hair_removal(image)")
    print("- enhance_image(image)")
    print("- segment_lesion(image)")
    print("- get_class_weights(train_generator)")
    print("- load_dataset_info(data_dir)")
