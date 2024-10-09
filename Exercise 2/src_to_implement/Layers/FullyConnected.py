import numpy as np
from .Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input, output):
        super().__init__()  # call base class constructor
        self.trainable = True

        self.weights = np.random.rand(input + 1, output) # extra input for b
        self._optimiser = None
        self._gradient_weights = None

    def forward(self, in_tensor):
        # append column of ones to the input_tensor for b
        batch = in_tensor.shape[0]
        ones_col = np.ones((batch, 1))
        bias_input = np.hstack((in_tensor, ones_col))

        out_tensor = np.dot(bias_input, self.weights)
        self.bias_input = bias_input # for backpropagation

        return out_tensor

    def backward(self, error_tensor):
        self._gradient_weights = np.dot(self.bias_input.T, error_tensor)

        # propagate error to the previous layer minus b
        error_tensor_prev = np.dot(error_tensor, self.weights[:-1].T)

        if self._optimiser is not None:
            self.weights = self._optimiser.calculate_update(self.weights, self._gradient_weights)

        return error_tensor_prev

    @property
    def gradient_weights(self): return self._gradient_weights # w gradient

    @property
    def optimiser(self): return self._optimiser # optimiser getter

    @optimiser.setter
    def optimiser(self, optimiser): self._optimiser = optimiser # optimiser setter
