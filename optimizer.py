from tensor import Parameter

class SGD:
    """Optimizer for batch stochastic gradient descent."""
    def __init__(self, parameters: list[Parameter], learning_rate: float, batch_size):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def step(self):
        for param in self.parameters:
            param.array = param.array - self.learning_rate * param.gradient / self.batch_size
    
    def zero_gradients(self):
        for param in self.parameters:
            param.zero_gradient()
