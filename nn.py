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
        self.weights = np.random.rand(out_size, in_size) * 0.01
        self.biases = np.random.rand(out_size) * 0.01

    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if np.size(x) != self.in_size:
            raise Exception("Forward input must match input size")
        
        z = self.weights @ x + self.biases
        a = self.activation_func(z)
        self.cache = LayerCache(z, a)
        return a

class NeuralNetwork:
    def __init__(self) -> None:
        self.lr = 0.001
        self.layers = [
            Layer(28*28, 128, relu),
            Layer(128, 64, relu),
            Layer(64, 10, softmax)
        ]
        self.loss_func = cross_entropy_loss
        self.input_layer = None

    def forward(self, x) -> npt.NDArray[np.float64]:
        self.input_layer = x
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, prediction, label):
        if not all([layer.cache for layer in self.layers]):
            raise Exception("Run forward pass before calling backward")

        err_sig3 = prediction - label
        grad_w3 = np.outer(err_sig3, self.layers[1].cache.a) # type: ignore

        err_sig2 = err_sig3 @ self.layers[2].weights * relu_prime(self.layers[1].cache.z)
        grad_w2 = np.outer(err_sig2, self.layers[0].cache.a) #type: ignore

        err_sig1 = err_sig2 @ self.layers[1].weights * relu_prime(self.layers[0].cache.z)
        grad_w1 = np.outer(err_sig1, self.input_layer)

        self.layers[2].weights -= self.lr * grad_w3
        self.layers[1].weights -= self.lr * grad_w2
        self.layers[0].weights -= self.lr * grad_w1
