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

class Model(ABC):
    def __init__(self, *layers):
        self.layers = layers

    @abstractmethod
    def forward(self, x) -> Tensor:
        ...

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

class Sequential(Model):
    def __init__(self, *layers):
        self.layers = layers
        super().__init__(*self.layers)

    def forward(self, x) -> Tensor:
        x = Tensor(x, label="x")
        for layer in self.layers:
            x = layer.forward(x)
        return x
