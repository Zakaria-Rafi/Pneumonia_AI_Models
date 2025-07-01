from config import IS_CROSS_VALIDATION
from numpy import float64
from numpy.typing import NDArray
from numpy import float32
from numpy import dtype
from typing import Any
from numpy import ndarray
from typing import Dict, Generator, List, Literal, Tuple
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
import numpy as np
from sklearn.utils import compute_class_weight
import tensorflow as tf
from config import RESIZE_DIM, IS_DATA_AUGMENTED
from src.models.common.ModelInterface import ModelInterface


class CNNModel(ModelInterface):
    """ Defines the neural network architecture and prepares it for training.

    - Sets up data augmentation to artificially expand the training dataset.
    - Creates a multi-layer CNN with 4 convolutional blocks (32→64→128→256 filters).
    - Uses BatchNormalization and Dropout layers to prevent overfitting.
    - Compiles the model with Adam optimizer and categorical cross-entropy loss.
    - Calculates class weights to handle class imbalance in the dataset.

    The model is designed to classify chest X-rays into NORMAL, BACTERIA, and VIRUS categories using the grayscales images.

    It is a Keras Sequential model so it needs to use the TFDataset (3 dimensions).
    """

    model: Sequential
    name = "CNNModel"

    def __init__(self):
        """
        Initializes the CNN model.

        """
        self.input_shape = RESIZE_DIM + \
            (1,)  # Adding channel dimension for grayscale images
        self.num_classes = 3
        self.model = self.__create_cnn_model()

    def __create_cnn_model(self) -> Sequential:
        """ Define the CNN architecture.

        Returns:
            Sequential: The compiled CNN model.
        """
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu',
                   padding='same', input_shape=self.input_shape),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Fourth convolutional block
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Flatten and fully connected layers
            Flatten(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            # Output layer with softmax for classification
            Dense(self.num_classes, activation='softmax')
        ])

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=1e-4),  # type: ignore
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def __get_callbacks(self) -> List:
        """ Define callbacks for better training performance.

        Returns:
            list: A list of callbacks including EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau.
        """
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            restore_best_weights=True
        )

        checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.0001,
            verbose=1
        )

        return [early_stopping, checkpoint, reduce_lr]

    def __get_class_weights(self, y_train: NDArray[float32]) -> Dict[int, float]:
        """ Calculate class weights to handle imbalance.

        Args:
            y_train (NDArray[float32]): Training labels in one-hot encoded format.

        Returns:
            dict: A dictionary mapping class indices to their respective weights.
        """
        y_integers = np.argmax(y_train, axis=1)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_integers),
            y=y_integers
        )
        return dict(enumerate(class_weights))

    def fit(self, x_train: NDArray[float32], y_train: NDArray[float32], epochs: int = 30) -> Dict[Literal["accuracy", "val_accuracy", "loss", "val_loss"], List[float]]:
        """ Train the model with data augmentation and class weights.
        Apply cross-validation if enabled.

        Args:
            x_train (NDArray[float32]): Training data.
            y_train (NDArray[float32]): Training labels in one-hot encoded format.
            epochs (int): Number of training epochs (default: 30).

        Returns:
            dict: Training history containing metrics like accuracy and loss.
        """

        print(x_train.shape)
        # Get callbacks
        callbacks = self.__get_callbacks()

        # Calculate class weights
        class_weights = self.__get_class_weights(y_train)

        train = None

        if (IS_CROSS_VALIDATION):
            data = self.__cross_validate_data(x_train, y_train)
        else:
            data = [(x_train, y_train, None, None)]

        # Train the model
        for (x, y, x_val, y_val) in data:
            train = self.model.fit(
                x,
                y,
                validation_data=(x_val, y_val),
                epochs=epochs,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )

        return train.history  # type: ignore

    def __cross_validate_data(self, x: NDArray[float32], y: NDArray[float32], k: int = 3) -> Generator[Tuple[NDArray[float32], NDArray[float32], NDArray[float32], NDArray[float32]], None, None]:
        """ Perform k-fold cross-validation on the dataset.
        Args:
            x (NDArray[float32]): Input data.
            y (NDArray[float32]): Labels corresponding to the input data.
            k (int): Number of folds for cross-validation (default: 3).
        Yields:
            Generator[Tuple[NDArray[float32], NDArray[float32], NDArray[float32], NDArray[float32]]]: Training and validation data for each fold.
        """
        if k <= 1:
            raise ValueError("k must be greater than 1 for cross-validation.")
        x_splitted = np.array_split(x, k)
        y_splitted = np.array_split(y, k)
        for i in range(k):
            x_train = np.concatenate(x_splitted[:i] + x_splitted[i + 1:])
            y_train = np.concatenate(y_splitted[:i] + y_splitted[i + 1:])
            x_val = x_splitted[i]
            y_val = y_splitted[i]
            yield x_train, y_train, x_val, y_val

    def predict(self, x_test: NDArray[float32]) -> NDArray[float64]:
        """ Predict the model.

        Args:
            x_test (NDArray[float32]): Test data.

        Returns:
            NDArray[float64]: Predicted class indices for the test data.
        """
        predictions = self.model.predict(x_test)
        return np.argmax(predictions, axis=1)

    def summary(self) -> None:
        """ Prints the summary of the model. """
        self.model.summary()
