import numpy as np
import numpy.typing as npt

from functions import relu, relu_prime

class Tensor:
    """Represents a tensor node in a computational graph."""
    def __init__(self, array: npt.NDArray, inputs: tuple["Tensor", ...] = (), label=""):
        self.array = array
        self.inputs = inputs
        self._gradient: npt.NDArray = np.zeros_like(self.array)
        self.backward = None
        self.label = label

    def __repr__(self):
        return f"Tensor({self.label + "," if self.label else ""}" \
              f"{"x".join([str(dim) for dim in self.array.shape])})"

    def __add__(self, other: "Tensor") -> "Tensor":
        output = Tensor(self.array + other.array, (self, other))

        def backward():
            self.gradient += output.gradient
            other.gradient += np.sum(output.gradient, axis=0)

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

        def backward():
            self.gradient += output.gradient @ other.array.T
            other.gradient += self.array.T @ output.gradient

        output.backward = backward
    
        return output
    
    @property
    def gradient(self):
        return self._gradient
    
    @gradient.setter
    def gradient(self, new_grad: npt.NDArray):
        if new_grad.shape != self.array.shape:
            raise Exception(f"Gradient shape {new_grad.shape} does not match tensor shape {self.array.shape}")
    
    def zero_gradient(self) -> None:
        self.gradient = np.zeros_like(self.array)

    def sum(self) -> "Tensor":
        output = Tensor(self.array.sum(), inputs=(self,))

        def backward():
            self.gradient += np.ones_like(self.array)

        output.backward = backward
        return output

    def relu(self) -> "Tensor":
        output = Tensor(relu(self.array), inputs=(self, ))

        def backward():
            self.gradient += relu_prime(self.array)
        
        output.backward = backward

        return output
    
    def transpose(self) -> "Tensor":
        output = Tensor(self.array.T, inputs=(self, ))

        def backward():
            self.gradient = output.gradient.T

        self.backward = backward

        return output

    def topological_sort(self) -> list["Tensor"]:
        """Generate a valid toplogical ordering of the computational graph."""
        order = []
        visited = set()
        stack: list[tuple[Tensor, bool]] = [(self, True), (self, False)]
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
        """
        Performs the backward pass on all nodes in the computational
        graph in reverse topological order. Can only be called on a scalar
        valued tensor.
        """
        if self.array.size != 1:
            raise Exception("Backward can only be called on a scalar valued tensor.")
        
        self.gradient = np.ones_like(self.array)
        for tensor in reversed(self.topological_sort()[1:]):
            if tensor.backward is None:
                continue
            tensor.backward()

class Parameter(Tensor):
    def __init__(self, array: npt.NDArray = np.array(0), label: str = ""):
        super().__init__(array, label=label)

    def __repr__(self) -> str:
        return f"Parameter({self.label})"