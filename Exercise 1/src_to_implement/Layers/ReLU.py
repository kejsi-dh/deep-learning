import numpy as np
from .Base import BaseLayer

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, in_tensor):
        self.in_tensor = in_tensor

        out_tensor = np.maximum(0, in_tensor)
        return out_tensor

    def backward(self, error_tensor):
        relu_gradient = self.in_tensor > 0

        error_tensor_prev = error_tensor * relu_gradient
        return error_tensor_prev