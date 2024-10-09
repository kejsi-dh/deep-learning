import numpy as np
from .Base import BaseLayer


class Conv(BaseLayer):
    def __init__(self, stride_shape, conv_shape, num_kernels, padding='valid'):
        super().__init__()
        self.trainable = True
        self.num_kernels = num_kernels
        self.conv_shape = conv_shape
        self.stride_shape = stride_shape
        self.padding = padding
        self.weights = np.random.uniform(0, 1, size=(num_kernels, *conv_shape))
        self.bias = np.random.uniform(0, 1, size=num_kernels)

        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)
        self.in_tensor = None

        # handle stride shape
        '''
        if isinstance(stride_shape, int):
            self.stride_shape = (stride_shape,) * (len(conv_shape) - 1)
        else: self.stride_shape = tuple(stride_shape)
        '''

        if len(conv_shape) == 2:
            # 1D convolution
            # self.weights = np.random.uniform(0, 1, (num_kernels, *conv_shape))
            self.is_2d = False
        elif len(conv_shape) == 3:
            # 2D convolution
            # self.weights = np.random.uniform(0, 1, (num_kernels, *conv_shape))
            self.is_2d = True

    def _apply_padding(self, input_tensor):
        if self.padding == 'same':
            # same padding
            pad_height = (self.conv_shape[1] - 1) // 2
            pad_width = (self.conv_shape[2] - 1) // 2
            return np.pad(input_tensor, ((0, 0), (0, 0), (pad_height, pad_height),
                                         (pad_width, pad_width)), mode='constant')
        elif isinstance(self.padding, int):
            # integer padding
            return np.pad(input_tensor, ((0, 0), (0, 0), (self.padding, self.padding),
                                         (self.padding, self.padding)), mode='constant')
        else: return input_tensor # no padding

    def forward(self, in_tensor):
        self.in_tensor = in_tensor
        if self.is_2d: return self._conv2d_forward(in_tensor)
        else: return self._conv1d_forward(in_tensor)

    def _conv1d_forward(self, in_tensor):
        # 1D convolution zero-padding and stride handling
        '''
        batch_size, in_channels, in_length = in_tensor.shape
        kernel_size = self.conv_shape[1]

        out_length = (in_length - kernel_size) // self.stride_shape[0] + 1
        output = np.zeros((batch_size, self.num_kernels, out_length), dtype=np.float32)

        for i in range(out_length):
            start = i * self.stride_shape[0]
            end = start + kernel_size
            input_slice = in_tensor[:, :, i * self.stride_shape[0]:start + kernel_size]
            output[:, :, i] = np.tensordot(input_slice, self.weights,
                                           axes=([1, 2], [1, 2])) + self.bias

        return output
        '''

        in_tensor = self._apply_padding(in_tensor)

        batch_size, in_channels, in_length = in_tensor.shape
        kernel_size = self.conv_shape[1]
        stride_size = self.stride_shape

        out_length = (in_length - kernel_size) // stride_size + 1
        output = np.zeros((batch_size, self.num_kernels, out_length))

        for i in range(0, in_length - kernel_size + 1, stride_size):
            start = i * self.stride_shape[0]
            end = start + kernel_size
            input_slice = in_tensor[:, :, i * self.stride_shape[0]:start + kernel_size]
            output[:, :, i] = np.tensordot(input_slice, self.weights,
                                           axes=([1, 2], [1, 2])) + self.bias

    def _conv2d_forward(self, in_tensor):
        # 2D convolution zero-padding and stride handling
        '''
        batch_size, in_channels, in_height, in_width = in_tensor.shape
        kernel_height, kernel_width = self.conv_shape[:1]

        out_height = (in_height - kernel_height) // self.stride_shape[0] + 1
        out_width = (in_width - kernel_width) // self.stride_shape[1] + 1
        output = np.zeros((batch_size, self.num_kernels, out_height, out_width), dtype=np.float32)

        for i in range(out_height):
            for j in range(out_width):
                start_y = i * self.stride_shape[0]
                end_y = start_y + kernel_height
                start_x = j * self.stride_shape[1]
                end_x = start_x + kernel_width

                input_slice = in_tensor[:, :, start_y:end_y, start_x:end_x]
                output[:, :, i, j] = np.tensordot(input_slice, self.weights,
                                                  axes=([1, 2, 3], [1, 2, 3])) + self.bias

        return output
        '''

        batch_size, in_channels, in_height, in_width = in_tensor.shape
        kernel_h, kernel_w = self.conv_shape[1], self.conv_shape[2]
        stride_h, stride_w = self.stride_shape

        input_tensor = self._apply_padding(in_tensor)
        out_height = (in_height - kernel_h) // stride_h + 1
        out_width = (in_width - kernel_w) // stride_w + 1

        output_tensor = np.zeros((batch_size, self.num_kernels, out_height, out_width))

        for batch in range(batch_size):
            for kernel in range(self.num_kernels):
                for i in range(0, in_height - kernel_h + 1, stride_h):
                    for j in range(0, in_width - kernel_w + 1, stride_w):
                        region = input_tensor[batch, :, i:i + kernel_h, j:j + kernel_w]
                        output_tensor[batch, kernel, i // stride_h, j // stride_w] = np.sum(
                            region * self.weights[kernel]) + self.bias[kernel]

        return output_tensor

    def backward(self, error_tensor):
        if self.is_2d: return self._conv2d_backward(error_tensor)
        else: return self._conv1d_backward(error_tensor)

    def _conv1d_backward(self, error_tensor):
        """
        b, num_kernels, output_y = error_tensor.shape
        _, c, y = self.in_tensor.shape

        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)

        for i in range(output_y):
            start = i * self.stride_shape[0]
            end = start + self.conv_shape[1]
            self._gradient_weights += np.tensordot(error_tensor[:, :, i],
                                                   self.in_tensor[:, :, start:end], axes=([0], [0]))

        self._gradient_bias = np.sum(error_tensor, axis=(0, 2))

        propagated_error = np.zeros_like(self.in_tensor)
        padded_error = np.pad(propagated_error, ((0, 0), (0, 0),
                                                 (self.conv_shape[1] // 2, self.conv_shape[1] // 2)))

        for i in range(output_y):
            start = i * self.stride_shape[0]
            end = start + self.conv_shape[1]
            for j in range(self.num_kernels):
                padded_error[:, :, start:end] += np.tensordot(error_tensor[:, j, i],
                                                              self.weights[j, :, :], axes=0)

        propagated_error = padded_error[:, :, self.conv_shape[1] // 2:-self.conv_shape[1] // 2]

        if self._optimiser_weights:
            self.weights = self._optimiser_weights.update(self.weights, self.gradient_weights)
        if self._optimiser_bias:
            self.bias = self._optimiser_bias.update(self.bias, self.gradient_bias)

        return propagated_error
        """
        batch_size, num_kernels, out_length = error_tensor.shape
        _, in_channels, in_length = self.in_tensor.shape
        kernel_size = self.conv_shape[:1]

        propagated_error = np.zeros_like(self.in_tensor, dtype=np.float32)
        self._gradient_weights = np.zeros_like(self.weights, dtype=np.float32)
        self._gradient_bias = np.zeros_like(self.bias, dtype=np.float32)

        for i in range(out_length):
            start = i * self.stride_shape[0]
            end = start + kernel_size
            input_slice = self.in_tensor[:, :, start:end]

            for b in range(batch_size):
                self._gradient_weights += error_tensor[b, :, i][:, np.newaxis, np.newaxis] \
                                          * input_slice[b]
                propagated_error[b, :, start:end] += \
                    np.tensordot(error_tensor[b, :, i], self.weights, axes=(0, 0))

        self._gradient_bias = np.sum(error_tensor, axis=(0, 2))

        return propagated_error

    def _conv2d_backward(self, error_tensor):
        """
        b, num_kernels, output_y, output_x = error_tensor.shape
        _, c, y, x = self.in_tensor.shape

        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)

        for i in range(output_y):
            for j in range(output_x):
                start_y = i * self.stride_shape[0]
                start_x = j * self.stride_shape[1]
                end_y = start_y + self.conv_shape[1]
                end_x = start_x + self.conv_shape[2]
                self._gradient_weights += np.tensordot(error_tensor[:, :, i, j],
                                                       self.in_tensor[:, :, start_y:end_y, start_x:end_x],
                                                       axes=([0], [0]))

        self._gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))

        propagated_error = np.zeros_like(self.in_tensor)
        padded_error = np.pad(propagated_error, ((0, 0), (0, 0),
                                                 (self.conv_shape[1] // 2, self.conv_shape[1] // 2),
                                                 (self.conv_shape[2] // 2, self.conv_shape[2] // 2)))

        for i in range(output_y):
            for j in range(output_x):
                start_y = i * self.stride_shape[0]
                end_y = start_y + self.conv_shape[1]
                start_x = j * self.stride_shape[1]
                end_x = start_x + self.conv_shape[2]
                for k in range(self.num_kernels):
                    padded_error[:, :, start_y:end_y, start_x:end_x] += \
                        np.tensordot(error_tensor[:, k, i, j], self.weights[k, :, :, :], axes=0)

        propagated_error = padded_error[:, :, self.conv_shape[1] // 2:-self.conv_shape[1] // 2,
                           self.conv_shape[2] // 2:-self.conv_shape[2] // 2]

        if self._optimizer_weights:
            self.weights = self._optimizer_weights.update(self.weights, self._gradient_weights)
        if self._optimizer_bias:
            self.bias = self._optimizer_bias.update(self.bias, self._gradient_bias)

        return propagated_error
        """
        batch_size, num_kernels, out_height, out_width = error_tensor.shape
        _, in_channels, in_height, in_width = self.in_tensor.shape
        kernel_height, kernel_width = self.conv_shape[1]

        propagated_error = np.zeros_like(self.in_tensor, dtype=np.float32)
        self._gradient_weights = np.zeros_like(self.weights, dtype=np.float32)
        self._gradient_bias = np.zeros_like(self.bias, dtype=np.float32)

        for i in range(out_height):
            for j in range(out_width):
                start_y = i * self.stride_shape[0]
                end_y = start_y + kernel_height
                start_x = j * self.stride_shape[1]
                end_x = start_x + kernel_width
                input_slice = self.in_tensor[:, :, start_y:end_y, start_x:end_x]

                for b in range(batch_size):
                    self._gradient_weights += error_tensor[b, :, i, j][:, np.newaxis,
                                              np.newaxis, np.newaxis] * input_slice[b]
                    propagated_error[b, :, start_y:end_y, start_x:end_x] += \
                        np.tensordot(error_tensor[b, :, i, j], self.weights, axes=(0, 0))

        self._gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))

        return propagated_error

    def initialize(self, initializer, bias_initializer):
        fan_in = np.prod(self.conv_shape)
        fan_out = np.prod(self.conv_shape[1:]) * self.num_kernels
        self.weights = initializer.initialize(self.conv_shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize((self.num_kernels,))

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias
