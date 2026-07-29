import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

from functions import relu, relu_prime, softmax, cross_entropy_loss

@dataclass
class LayerCache:
    z: npt.NDArray[np.float64]
    a: npt.NDArray[np.float64]

class Layer:
    def __init__(self, in_size: int, out_size: int, activation_func) -> None:
        self.cache = None
        self.in_size = in_size
        self.out_size = out_size
        self.activation_func = activation_func
        self.weights = np.random.randn(out_size, in_size) * 0.01
        self.biases = np.random.randn(out_size) * 0.01

    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        z = x @ self.weights.T + self.biases
        a = self.activation_func(z)
        self.cache = LayerCache(z, a)
        return a

class NeuralNetwork:
    def __init__(self) -> None:
        self.lr = .1
        self.layers = [
            Layer(28*28, 128, relu),
            Layer(128, 64, relu),
            Layer(64, 10, softmax)
        ]
        self.input_layer = None

    def forward(self, x) -> npt.NDArray[np.float64]:
        self.input_layer = x
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, prediction, label):
        if self.input_layer is None or not all([layer.cache for layer in self.layers]):
            raise Exception("Run forward pass before calling backward")

        dZ_3 = prediction - label
        dW_3 = dZ_3.T @ self.layers[1].cache.a # type: ignore

        dZ_2 = dZ_3 @ self.layers[2].weights * relu_prime(self.layers[1].cache.z)
        dW_2 = dZ_2.T @ self.layers[0].cache.a #type: ignore

        dZ_1 = dZ_2 @ self.layers[1].weights * relu_prime(self.layers[0].cache.z)
        dW_1 = dZ_1.T @ self.input_layer

        batch_size = self.input_layer.shape[0]

        self.layers[2].weights -= self.lr * dW_3 / batch_size
        self.layers[1].weights -= self.lr * dW_2 / batch_size
        self.layers[0].weights -= self.lr * dW_1 / batch_size
