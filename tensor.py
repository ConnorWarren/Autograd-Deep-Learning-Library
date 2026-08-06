import numpy as np
import numpy.typing as npt

class Tensor:
    def __init__(self, array: npt.NDArray, inputs = None):
        self.array = array
        self.inputs = inputs
        self.gradient = np.zeros(self.array.shape)

    def __add__(self, other: "Tensor") -> "Tensor":
        output = Tensor(self.array + other.array, (self, other))

        def backward():
            self.gradient += 1
            other.gradient += 1

        self.backward = backward

        return output

    def __mul__(self, other: "Tensor"):
        result = Tensor(self.array * other.array, (self, other))

        def backward():
            temp = self.gradient
            self.gradient += other.gradient
            other.gradient += temp

        self.backward = backward

        return result

    def __matmul__(self, other: "Tensor"):
        return Tensor(self.array @ other.array, (self, other))

    def zero_gradients(self):
        ...

    def topological_sort(self) -> list["Tensor"]:
        order = []
        visited = set()
        stack = [(self, True), (self, False)]
        while stack:
            top, processed = stack.pop()
            if processed:
                order.append(top)
                continue
            if not top.inputs:
                continue
            for parent in top.inputs:
                if parent not in visited:
                    stack.extend([(parent, True), (parent, False)])
                    visited.add(parent)

        return order

    def backward(self) -> None:
        for tensor in reversed(self.topological_sort()):
            tensor.gradient = tensor.backward()