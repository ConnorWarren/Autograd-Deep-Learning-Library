import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod

from tensor import Tensor, Parameter

class Layer(ABC):
    def __init__(self, in_size: int, out_size: int, use_params: bool = True) -> None:
        self.in_size = in_size
        self.out_size = out_size
        self.weights = Parameter()
        self.biases = Parameter()

        if use_params:
            self.weights = Parameter(np.random.randn(out_size, in_size) * 0.01, label="W")
            self.biases = Parameter(np.random.randn(out_size) * 0.01, label="b")

        self.use_params = use_params

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ...

    def backward(self) -> Tensor:
        ...
    
    def parameters(self) -> tuple[Parameter | None, ...]:
        return self.weights, self.biases

class Linear(Layer):
    def __init__(self, in_size: int, out_size: int) -> None:
        super().__init__(in_size, out_size)

    def forward(self, x: Tensor) -> Tensor:
        z = x @ self.weights.transpose() + self.biases
        return z

class ReLU(Layer):
    def __init__(self, in_size: int) -> None:
        super().__init__(in_size, in_size, use_params=False)

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

class NeuralNetwork:
    def __init__(self) -> None:
        self.layers = [
            Linear(28*28, 128),
            ReLU(128),
            Linear(128, 64),
            ReLU(64),
            Linear(64, 10)
        ]
        self.input_layer = None

    def forward(self, x) -> Tensor:
        x = Tensor(x)
        self.input_layer = x
        for layer in self.layers:
            x = layer.forward(x)

        return x
    """
    def backward(self, prediction, label):
        if self.input_layer is None or not all([layer.cache for layer in self.layers]):
            raise Exception("Run forward pass before calling backward")

        batch_size = self.input_layer.shape[0]

        dZ_3 = (prediction - label) / batch_size
        dW_3 = dZ_3.T @ self.layers[1].cache.a # type: ignore

        dZ_2 = dZ_3 @ self.layers[2].weights * relu_prime(self.layers[1].cache.z)
        dW_2 = dZ_2.T @ self.layers[0].cache.a #type: ignore

        dZ_1 = dZ_2 @ self.layers[1].weights * relu_prime(self.layers[0].cache.z)
        dW_1 = dZ_1.T @ self.input_layer

        self.layers[2].weights -= self.lr * dW_3
        self.layers[1].weights -= self.lr * dW_2
        self.layers[0].weights -= self.lr * dW_1

        self.layers[2].biases -= self.lr * np.sum(dZ_3, axis=0)
        self.layers[1].biases -= self.lr * np.sum(dZ_2, axis=0)
        self.layers[0].biases -= self.lr * np.sum(dZ_1, axis=0)
    """

    def parameters(self):
        """Return a list of all parameters in the network"""
        return [param for layer in self.layers for param in layer.parameters()]

    def save(self, path: str) -> None:
        layer_data = {}
        for i, layer in enumerate(self.layers):
            if not layer.use_params:
                continue
            layer_data[f"W_{i}"] = layer.weights.array
            layer_data[f"b_{i}"] = layer.biases.array

        np.savez_compressed(path, **layer_data)

    def load(self, path: str) -> None:
        with np.load(path) as data:
            for i, layer in enumerate(self.layers):
                if not layer.use_params:
                    continue
                layer.weights = Parameter(data[f"W_{i}"])
                layer.biases = Parameter(data[f"b_{i}"])
