"""
    This module provides functions to load and preprocess the Chest X-ray dataset for Scikit-learn models (1 dimension (flattened)) (SKDataset).
"""
import numpy as np
from numpy import float32
from numpy.typing import NDArray
from src.datasets.datasetHelper import load_dataset, split_dataset, resize_images, categorize_labels
from config import IS_DATA_AUGMENTED

def load_data_with_validation() -> tuple[tuple[NDArray[float32], NDArray[float32]], tuple[NDArray[float32], NDArray[float32]], tuple[NDArray[float32], NDArray[float32]]]:
    """
    Load and preprocess the dataset, splitting it into training, validation, and test sets.

    Returns:
        tuple: ((x_train, y_train), (x_val, y_val), (x_test, y_test))
                Preprocessed training, validation, and test data with their labels.
    """
    # Load training data
    print("Loading training data...")
    x_train_raw, y_train_raw = load_dataset(
        'train', flatten=True, print_repartition=True)

    # Load test data
    print("Loading test data...")
    x_test_raw, y_test_raw = load_dataset(
        'test', flatten=True, print_repartition=True)

    # Split training data to create validation set
    print("Creating validation set...")
    x_train, x_val, y_train, y_val = split_dataset(x_train_raw, y_train_raw)

    # Resize and normalize images
    print("Preprocessing images...")
    x_train, x_val, x_test = resize_images(
        train=x_train, val=x_val, test=x_test_raw)

    # Convert labels to categorical (one-hot encoding)
    print("Processing labels...")
    y_train, y_val, y_test = categorize_labels(
        train=y_train, val=y_val, test=y_test_raw)

    print("Data loading complete!")

    if IS_DATA_AUGMENTED:
        print("Applying data augmentation...")
        x_train = np.append(x_train, augment_data(x_train), axis=0)
        y_train = np.append(y_train, y_train, axis=0)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def load_data() -> tuple[tuple[NDArray[float32], NDArray[float32]], tuple[NDArray[float32], NDArray[float32]]]:
    """
    Load and preprocess the dataset, splitting it into training, and test sets.

    Returns:
        tuple: ((x_train, y_train), (x_test, y_test))
                Preprocessed training and test data with their labels.
    """
    # Load training data
    print("Loading training data...")
    x_train, y_train = load_dataset(
        'train', flatten=True, print_repartition=True)

    # Load test data
    print("Loading test data...")
    x_test, y_test = load_dataset(
        'test', flatten=True, print_repartition=True)

    print("Data loading complete!")

    if IS_DATA_AUGMENTED:
        print("Applying data augmentation...")
        x_train = np.append(x_train, augment_data(x_train), axis=0)
        y_train = np.append(y_train, y_train, axis=0)

    return (x_train, y_train), (x_test, y_test)

def augment_data(x_train: NDArray[float32]) -> NDArray[float32]:
    """
    Apply data augmentation techniques to the input images for a scikit-learn model.
    Args:
        image (numpy.ndarray): Input images to be augmented.
    Returns:
        numpy.ndarray: Augmented images.
    """
    return np.array([__data_augmentation(image) for image in x_train])

def __data_augmentation(image: NDArray[float32]) -> NDArray[float32]:
    """
    Apply data augmentation techniques to the input image for a scikit-learn model without reshaping image to its original dimensions.
    Args:
        image (numpy.ndarray): Input image to be augmented with shape (flattened).
    Returns:
        numpy.ndarray: Augmented image.
    """
    # todo :)
    return image
