import numpy as np
import numpy.typing as npt
from tensor import Tensor
from functions import cross_entropy_loss, softmax

def cel_softmax(expected: Tensor, predicted: Tensor) -> Tensor:
    probs = softmax(predicted.array)
    output = Tensor(
        np.array(cross_entropy_loss(expected.array, probs)),
        inputs=(expected, Tensor(probs, inputs=(predicted,)))
    )
    def backward():
        predicted.gradient = predicted.array - expected.array

    output.backward = backward
    return output
