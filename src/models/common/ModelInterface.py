from numpy import float64
from numpy import float32
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from typing import Literal, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class ModelInterface(ABC):
    name: str

    @abstractmethod
    def predict(self, x_test: NDArray[float32]) -> NDArray[float64]:
        pass

    @abstractmethod
    def summary(self):
        pass

    def evaluate(self, x_test: NDArray[float32], y_test: NDArray[float32]) -> Tuple[dict[Literal["accuracy", "precision", "recall", "f1_score"], float | np.floating | NDArray[float64]], dict[Literal["predictions", "y_true"], NDArray[float32 | float64]]]:
        pred = self.predict(x_test)

        if len(y_test.shape) > 1 and y_test.shape[1] > 1: # fix for tf datasets
            y_test = np.argmax(y_test, axis=1)

        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, average='weighted')
        recall = recall_score(y_test, pred, average='weighted')
        f1 = f1_score(y_test, pred, average='weighted')

        return ({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score":  f1,
        }, {
            "predictions": pred,
            "y_true": y_test
        })