import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

from functions import relu, softmax, relu_prime, softmax_prime, cel_prime

class LinearLayer:
    def __init__(self, in_size: int, out_size: int) -> None:
        self.in_size = in_size
        self.out_size = out_size
        self.weights = np.random.rand(out_size, in_size)
        self.biases = np.random.rand(out_size)

    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if np.size(x) != self.in_size:
            raise Exception("Forward input must match input size")

        return np.linalg.matmul(self.weights, x) + self.biases

@dataclass
class LayerCache:
    z: npt.NDArray[np.float64]
    a: npt.NDArray[np.float64]

class NeuralNetwork:
    def __init__(self):
        self.lr = 0.001
        self.cache: list[LayerCache] = []
        self.fc1 = LinearLayer(28*28, 128)
        self.fc2 = LinearLayer(128, 64)
        self.fc3 = LinearLayer(64, 10)

    def forward(self, x) -> npt.NDArray[np.float64]:
        self.cache.append(LayerCache(x, x))
        
        z = self.fc1.forward(x)
        a = relu(z)
        self.cache.append(LayerCache(z, a))

        z = self.fc2.forward(a)
        a = relu(z)
        self.cache.append(LayerCache(z, a))
        
        z = self.fc3.forward(a)
        a = softmax(z)
        self.cache.append(LayerCache(z, a))
        
        return a
    
    def clear_cache(self):
        self.cache = []

    def backward(self, expected, actual):
        loss_w3 = np.outer(cel_prime(expected, actual) \
            * softmax_prime(self.cache[3].a), self.cache[2].a)
      
        loss_w2 = np.outer(loss_w3 * relu_prime(self.cache[2].a), self.cache[1].a)

        loss_w1 = np.outer(loss_w2 * relu_prime(self.cache[1].a), self.cache[0].a)
        
        self.fc3.weights -= loss_w3 * self.lr
        self.fc2.weights -= loss_w2 * self.lr
        self.fc1.weights -= loss_w1 * self.lr

        """del_z3 = softmax_prime(self.cache[3]) * relu_prime(self.cache[2]) \
            * relu_prime(self.cache[1]) * self.cache[0]
        grad_3 = loss_derivative(expected, actual) * del_z3
        self.fc3.weights -= grad_3 * self.lr
        
        del_z2 = relu_prime(self.cache[2]) * relu_prime(self.cache[1]) * self.cache[0]
        grad_2 = loss_derivative(expected, actual) * del_z2
        self.fc2.weights -= grad_2 * self.lr

        del_z1 = relu_prime(self.cache[1]) * self.cache[0]
        grad_1 = loss_derivative(expected, actual) * del_z1
        self.fc1.weights -= grad_1 * self.lr
        """