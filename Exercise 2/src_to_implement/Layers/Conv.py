import numpy as np
from .Base import BaseLayer
from scipy.signal import correlate, convolve
import copy

class Conv(BaseLayer):
    def __init__(self, stride_shape, conv_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.conv_shape = conv_shape
        self.num_kernels = num_kernels

        self.weights = np.random.uniform(0, 1, size=(num_kernels, *conv_shape))
        self.bias = np.random.uniform(0, 1, num_kernels)

        self._gradient_weights = None
        self._gradient_bias = None
        self._optimizer_weights = None
        self._optimizer_bias = None

    def forward(self, in_tensor):
        self.in_tensor = in_tensor
        output = np.zeros((in_tensor.shape[0], self.num_kernels, *in_tensor.shape[2:]))
        for batch in range(in_tensor.shape[0]):
            for kernel in range(self.num_kernels):
                for channel in range(in_tensor.shape[1]):
                    output[batch, kernel] += correlate(in_tensor[batch, channel],
                                                       self.weights[kernel, channel], 'same')
                output[batch, kernel] += self.bias[kernel]

        if len(self.stride_shape) > 1:
            return output[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]
        else: return output[:, :, ::self.stride_shape[0]]

    def backward(self, error_tensor):
        output = np.zeros_like(self.in_tensor)

        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)

        # get shape of input
        if len(self.in_tensor.shape) == 4:
            batch_size, num_channels, in_height, in_width = self.in_tensor.shape
        elif len(self.in_tensor.shape) == 3:
            batch_size, num_channels, in_length = self.in_tensor.shape
            in_height, in_width = 1, in_length  # 1D data has height 1
        else: raise ValueError("Input tensor must have 3 or 4 dimensions.")

        # get shape of error
        if len(self.in_tensor.shape) == 4:
            _, num_kernels, out_height, out_width = error_tensor.shape
        elif len(self.in_tensor.shape) == 3:
            _, num_kernels, out_length = error_tensor.shape
            out_height, out_width = 1, out_length  # 1D data has height 1
        else: raise ValueError("Output tensor must have 3 or 4 dimensions.")

        if len(self.stride_shape) > 1:
            up_error = np.zeros((batch_size, num_kernels, in_height, in_width))
            up_error[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor
        else:
            up_error = np.zeros((batch_size, num_kernels, in_length))
            up_error[:, :, ::self.stride_shape[0]] = error_tensor

        padded_in = self.padding(self.in_tensor, self.weights) # valid conv

        for batch in range(batch_size):
            for kernel in range(num_kernels):
                for channel in range(num_channels):
                    output[batch, channel] += \
                        convolve(up_error[batch, kernel],
                                 self.weights[kernel, channel], 'same')
                    self._gradient_weights[kernel, channel] += \
                        correlate(padded_in[batch, channel], up_error[batch, kernel], 'valid')

                self._gradient_bias[kernel] += np.sum(up_error[batch, kernel])

        if self._optimizer_weights and self._optimizer_bias:
            self.weights = \
                self._optimizer_weights.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)

        return output

    def initialize(self, weights_initializer, bias_initializer):
        self.fan_in = np.prod(self.conv_shape)
        self.fan_out = np.prod(self.conv_shape[1:]) * self.num_kernels
        self.weights = weights_initializer.initialize(self.weights.shape, self.fan_in, self.fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape)

    @staticmethod
    def padding(in_matrix, k):
        flag = (len(k.shape) == 3)
        if flag:
            k = k[:, :, np.newaxis, :]
            in_matrix = in_matrix[:, :, np.newaxis, :]

        padding = np.array(k.shape[2:]) - 1
        bottom_pad, right_pad = np.ceil(padding / 2).astype(int)
        top_pad, left_pad = padding - [bottom_pad, right_pad]

        padded_matrix = np.pad(in_matrix,
                               ((0, 0), (0, 0), (top_pad, bottom_pad), (left_pad, right_pad)))

        if flag: return np.squeeze(padded_matrix, axis=2)

        return padded_matrix

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @property
    def optimizer(self):
        return self._optimizer_weights

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer_weights = optimizer
        self._optimizer_bias = copy.deepcopy(optimizer)