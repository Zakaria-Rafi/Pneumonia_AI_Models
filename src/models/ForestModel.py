from numpy import float64
from numpy import float32
from numpy.typing import NDArray
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from src.models.common.SKModelInterface import SKModelInterface


class ForestModel(SKModelInterface):
    """
    A class to represent a Random Forest model for classification tasks.

    Hyperparameters was chosen base on the grid search results

    It is a sklearn RandomForestClassifier so it needs to use the SKDataset (flattened images)
    """

    model: RandomForestClassifier
    name = "Random Forest"

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=175, random_state=42, min_samples_split=5, max_depth=None)

    def fit(self, x_train: NDArray[float32], y_train: NDArray[float32]) -> None:
        """
        Train the Random Forest model on the training data.

        Args:
            x_train (NDArray[float32]): Training data features.
            y_train (NDArray[float32]): Training data labels.
        """
        self.model.fit(x_train, y_train)

    def predict(self, x_test: NDArray[float32]) -> NDArray[float64]:
        """
        Predict the labels for the test data.

        Args:
            x_test (NDArray[float32]): Test data features.

        Returns:
            NDArray[float32]: Predicted labels for the test data.
        """
        return self.model.predict(x_test)

    def summary(self) -> None:
        """
        Print the summary of the Random Forest model including its parameters.
        """
        print("Random Forest Model Summary:")
        params = self.model.get_params()
        for param, value in params.items():
            print(f"  {param}: {value}")
