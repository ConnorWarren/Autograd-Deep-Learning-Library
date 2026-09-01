import numpy as np
import numpy.typing as npt

from functions import relu, relu_prime

class Tensor:
    """Represents a tensor node in a computational graph."""
    def __init__(self, array: npt.NDArray, inputs: tuple[Tensor, ...] = (), label=""):
        self.array = array
        self.inputs = inputs
        self.gradient: npt.NDArray = np.zeros(self.array.shape)
        self.backward = None
        self.label = label

    def __repr__(self):
        return f"Tensor({self.label})"

    def __add__(self, other: "Tensor") -> "Tensor":
        output = Tensor(self.array + other.array, (self, other))

        def backward():
            self.gradient = self.gradient + output.gradient
            other.gradient = other.gradient + output.gradient

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
            self.gradient = self.gradient + output.gradient @ other.array.T
            other.gradient = other.gradient.T + output.gradient.T @ self.array

        output.backward = backward
        
        return output
    
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
            self.gradient = self.array.T

        self.backward = backward

        return output

    def zero_gradients(self) -> None:
        """Zero all gradients in the computational graph."""
        self.gradient = np.zeros(self.array.shape)
        for tensor in self.inputs:
            tensor.zero_gradients()

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
        
        self.gradient = np.ones_like(self.array.shape)
        for tensor in reversed(self.topological_sort()[1:]):
            if tensor.backward is None:
                continue
            tensor.backward()

class Parameter(Tensor):
    def __init__(self, array: npt.NDArray = np.array(0), label: str = ""):
        super().__init__(array, label=label)

    def __repr__(self) -> str:
        return f"Parameter({self.label})"