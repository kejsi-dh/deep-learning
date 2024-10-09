import numpy as np
from .Base import BaseLayer

class Flatten(BaseLayer):
    def forward(self, in_tensor):
        self.in_shape = in_tensor.shape
        flatten = in_tensor.reshape(in_tensor.shape[0], -1)
        return flatten

    def backward(self, error_tensor):
        return error_tensor.reshape(self.in_shape)