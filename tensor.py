import numpy as np
import numpy.typing as npt

class Tensor:
    def __init__(self, array: npt.NDArray, inputs: tuple[Tensor, ...] = (), label=""):
        self.array = array
        self.inputs = inputs
        self.gradient = np.zeros(self.array.shape)
        self.backward = None
        self.label = label

    def __repr__(self):
        return f"{self.label}"

    def __add__(self, other: "Tensor") -> "Tensor":
        output = Tensor(self.array + other.array, (self, other))

        def backward():
            self.gradient += output.gradient
            other.gradient += output.gradient

        output.backward = backward

        return output

    def __mul__(self, other: "Tensor"):
        output = Tensor(self.array * other.array, (self, other))

        def backward():
            self.gradient += output.gradient * other.array
            other.gradient += output.gradient * self.array

        output.backward = backward

        return output

    def __matmul__(self, other: "Tensor"):
        output = Tensor(self.array @ other.array, (self, other))
    
    def sum(self) -> "Tensor":
        output = Tensor(self.array.sum(), inputs=(self,))

        def backward():
            self.gradient = np.ones_like(self.array)

        output.backward = backward
        return output

    def zero_gradients(self) -> None:
        """Zero all gradients in the computational graph."""
        self.gradient = 0
        for tensor in self.inputs:
            tensor.zero_gradients()

    def topological_sort(self) -> list["Tensor"]:
        """Generate a valid toplogical ordering of the computational graph"""
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

    def backward_all(self) -> None:
        if self.array.size != 1:
            raise Exception("Backward can only be called on a scalar value")
        self.gradient = 1
        for tensor in reversed(self.topological_sort()[1:]):
            if tensor.backward is None:
                continue
            tensor.backward()
