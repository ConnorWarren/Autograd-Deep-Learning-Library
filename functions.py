import numpy as np
import numpy.typing as npt

def relu(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.maximum(0, x)

def relu_prime(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.where(x < 0, 0, 1)

def softmax(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    exp = np.exp(x)
    return exp / np.sum(exp)

def cross_entropy_loss(
        expected: npt.NDArray[np.float64], actual: npt.NDArray[np.float64]
    ) -> np.float64:
    return -np.sum(expected * np.log(actual))
