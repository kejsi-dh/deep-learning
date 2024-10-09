import numpy as np
from .Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, in_tensor):
        # subtract max value per row for numerical stability
        shifted_logits = in_tensor - np.max(in_tensor, axis=1, keepdims=True)

        exp_vals = np.exp(shifted_logits)
        sum_exp_vals = np.sum(exp_vals, axis=1, keepdims=True)
        self.out_tensor = exp_vals / sum_exp_vals

        return self.out_tensor

    def backward(self, error_tensor):
        batch, num_classes = error_tensor.shape

        error_tensor_prev = np.empty_like(error_tensor)

        for i in range(batch):
            softmax_output = self.out_tensor[i].reshape(-1, 1)
            jacobian_matrix = np.diagflat(softmax_output) - np.dot(softmax_output, softmax_output.T)
            error_tensor_prev[i] = np.dot(jacobian_matrix, error_tensor[i])

        return error_tensor_prev
