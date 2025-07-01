from numpy import float32
from numpy.typing import NDArray
from abc import ABC, abstractmethod
import numpy as np
from src.models.common.ModelInterface import ModelInterface


class SKModelInterface(ModelInterface, ABC):

    @abstractmethod
    def fit(self, x_train: NDArray[float32], y_train: NDArray[float32]) -> None:
        pass
