from numpy import argmax, float64
from numpy import float32
from numpy.typing import NDArray
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.models.common.SKModelInterface import SKModelInterface


class KNNModel(SKModelInterface):
    """ K-Nearest Neighbors model for chest X-ray classification.

    - Applies dimensionality reduction using PCA before classification
    - Uses standardization of features to improve KNN performance
    - Implements K-Nearest Neighbors classifier
    - Provides evaluation metrics compatible with CNN model

    The model is designed to classify chest X-rays into NORMAL, BACTERIA, and VIRUS categories.

    It is a sklearn KNeighborsClassifier so it needs to use the SKDataset (flattened images).
    """
    name = "KNN"

    def __init__(self, n_neighbors: int = 3, n_components: int = 100, 
                 algorithm: str = 'auto', p: int = 1, weights: str = 'distance'):
        """
        Initialize the KNN model.

        Args:
            n_neighbors (int): Number of neighbors to use for classification
            n_components (int): Number of PCA components for dimensionality reduction
            algorithm (str): Algorithm used to compute the nearest neighbors
            p (int): Power parameter for the Minkowski metric
            weights (str): Weight function used in prediction
        """
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.algorithm = algorithm
        self.p = p
        self.weights = weights
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            p=p,
            weights=weights
        )
        self.is_fitted = False
        self.num_classes = 3

    def fit(self, x_train: NDArray[float32], y_train: NDArray[float32]) -> None:
        """Train the KNN model.

        Args:
            x_train (NDArray[float32]): Training data
            y_train (NDArray[float32]): Training labels in one-hot encoded format
        """
        # Convert one-hot encoded labels to class indices
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            y_train_indices = argmax(y_train, axis=1)
        else:
            y_train_indices = y_train

        # Standardize features
        X_scaled = self.scaler.fit_transform(x_train)

        # Apply PCA for dimension reduction
        X_pca = self.pca.fit_transform(X_scaled)
        print(
            f"PCA explained variance ratio sum: {sum(self.pca.explained_variance_ratio_):.4f}")

        # Fit the KNN model
        self.model.fit(X_pca, y_train_indices)
        self.is_fitted = True

        self.model.predict(X_pca)

    def predict(self, x_test: NDArray[float32]) -> NDArray[float64]:
        """Predict classes for the given data.

        Args:
            x_test (NDArray[float32]): Test data

        Returns:
            NDArray[float64]: Predicted class indices for the test data
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")

        # Standardize features
        X_scaled = self.scaler.transform(x_test)

        # Apply PCA
        X_pca = self.pca.transform(X_scaled)

        # Return predictions
        return self.model.predict(X_pca)

    def summary(self) -> None:
        print("KNN Model Summary:")
        params = self.model.get_params()
        for param, value in params.items():
            print(f"  {param}: {value}")
