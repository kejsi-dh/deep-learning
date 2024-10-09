import numpy as np
from .Base import BaseLayer
from scipy.signal import convolve, convolve2d, correlate

class Conv(BaseLayer):
    def __init__(self, stride_shape, num_kernels, conv_shape, is_1d=True):
        super().__init__()
        self.trainable = True
        self.num_kernels = num_kernels
        self.conv_shape = conv_shape
        self.input_tensor = None
        self.stride_shape = stride_shape
        self.is_1d = is_1d

        # self.weights = np.random.randn(num_kernels, *conv_shape)
        # self.bias = np.random.randn(num_kernels)
        self.weights = np.random.uniform(0, 1, size=(num_kernels, *conv_shape))
        self.bias = np.random.uniform(0, 1, num_kernels)

        # self._gradient_weights = None
        # self._gradient_bias = None
        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)

        # self._optimizer_weights = None
        # self._optimizer_bias = None

    def forward(self, input_tensor):
        '''
        input_size = np.prod(input_tensor.shape)

        # Target shape for reshaping (y=10, x=14)
        target_y, target_x = 10, 14
        target_size = target_y * target_x

        if input_size != target_size:
            if input_size < target_size:
                pad_size = target_size - input_size
                input_tensor = np.pad(input_tensor.flatten(), (0, pad_size), mode='constant').reshape(
                    (target_y, target_x))
            else:
                input_tensor = input_tensor.flatten()[:target_size].reshape((target_y, target_x))
        else: input_tensor = input_tensor.reshape((target_y, target_x))
        '''
        self.input_tensor = input_tensor  # Save for the backward pass
        # b, c = input_tensor.shape[:2]  # Extract batch size and channels

        if self.is_1d:
            '''
            if len(input_tensor.shape) == 2:
                input_tensor = np.expand_dims(input_tensor, axis=0)

            b, c, y = input_tensor.shape
            m = self.conv_shape[1]
            stride_y = self.stride_shape[0]

            if (y - m) % stride_y != 0:
                raise ValueError(f"Invalid stride {stride_y} "
                                 f"for input dimension {y} and kernel size {m}.")

            output_shape = (((y - m) // stride_y) + 1,)
            output_tensor = np.zeros((b, self.num_kernels, *output_shape))

            
            for batch in range(b):
                for kernel in range(self.num_kernels):
                    for i in range(output_shape[0]):
                        y_start = i * stride_y
                        y_end = y_start + m

                        if y_end > y:
                            raise ValueError(f"Invalid input slicing: y_end {y_end} "
                                             f"exceeds input dimension {y}.")

                        conv_region = input_tensor[batch, :, y_start:y_end]
                        if conv_region.shape != self.weights[kernel].shape:
                            raise ValueError(
                                f"Shape mismatch between input slice {conv_region.shape} "
                                f"and weights {self.weights[kernel].shape}")

                        output_tensor[batch, kernel, i] = \
                            np.sum(conv_region * self.weights[kernel]) + self.bias[kernel]
            
            for batch in range(b):
                for kernel in range(self.num_kernels):
                    for channel in range(c):
                        conv_result = convolve(input_tensor[batch, channel],
                                               self.weights[kernel, channel], mode='valid')
                        output_tensor[batch, kernel] += conv_result[::stride_y]
                    output_tensor[batch, kernel] += self.bias[kernel]
            '''
            '''
            y = input_tensor.shape[2]  # Spatial dimension (y-axis)
            m = self.conv_shape[1]  # Kernel size
            stride_y = self.stride_shape[0]

            # Calculate output length
            output_length = (y - m) // stride_y + 1
            output_tensor = np.zeros((b, self.num_kernels, output_length))

            for batch in range(b):
                for kernel in range(self.num_kernels):
                    output = np.zeros(output_length)
                    for channel in range(c):
                        # Perform 1D convolution for each channel
                        conv_result = convolve(input_tensor[batch, channel],
                                               self.weights[kernel, channel], mode='valid')
                        output += conv_result[::stride_y]  # Apply stride

                    # Add bias
                    output_tensor[batch, kernel] = output + self.bias[kernel]
            '''
            b, c, length = input_tensor.shape
            kernel_size = self.conv_shape[-1]
            stride_y = self.stride_shape

            # Calculate output shape
            output_length = (length - kernel_size) // stride_y + 1
            output_tensor = np.zeros((b, self.num_kernels, output_length))

            for batch in range(b):
                for kernel in range(self.num_kernels):
                    output = np.zeros(output_length)
                    for channel in range(c):
                        # Correlate (apply filter without flipping the kernel)
                        conv_result = correlate(input_tensor[batch, channel],
                                                self.weights[kernel, channel],
                                                mode='valid')

                        # Apply stride and accumulate results
                        conv_result_strided = conv_result[::stride_y]
                        output += conv_result_strided

                    # Add bias after the correlation
                    output_tensor[batch, kernel] = output + self.bias[kernel]
        else:
            '''
            if len(input_tensor.shape) == 2:
                input_tensor = np.expand_dims(input_tensor, axis=0)
                input_tensor = np.expand_dims(input_tensor, axis=1)

            b, c, y, x = input_tensor.shape
            m, n = self.conv_shape[1], self.conv_shape[2]
            stride_y, stride_x = self.stride_shape

            output_shape_y = ((y - m) // stride_y) + 1
            output_shape_x = ((x - n) // stride_x) + 1

            if output_shape_y <= 0 or output_shape_x <= 0:
                stride_y = max(1, (y - m) // (1 if output_shape_y <= 0 else output_shape_y))
                stride_x = max(1, (x - n) // (1 if output_shape_x <= 0 else output_shape_x))

                output_shape_y = (y - m) // stride_y + 1
                output_shape_x = (x - n) // stride_x + 1

                if output_shape_y <= 0 or output_shape_x <= 0:
                    raise ValueError(f"Invalid stride or kernel size for input dimensions y={y}, "
                                     f"x={x}, kernel sizes m={m}, n={n}, and strides "
                                     f"stride_y={stride_y}, stride_x={stride_x}.")

            output_tensor = np.zeros((b, self.num_kernels, output_shape_y, output_shape_x))
            
            for batch in range(b):
                for kernel in range(self.num_kernels):
                    for i in range(output_shape_y):  # Loop over the output y-axis
                        for j in range(output_shape_x):  # Loop over the output x-axis
                            y_start, x_start = i * stride_y, j * stride_x  # Start positions
                            y_end, x_end = y_start + m, x_start + n  # End positions

                            # Ensure correct input slicing
                            if y_end > y or x_end > x:
                                raise ValueError(f"Invalid input slicing: y_end {y_end}, "
                                                 f"x_end {x_end} exceeds "
                                                 f"input dimensions y={y}, x={x}.")

                            conv_region = input_tensor[batch, :, y_start:y_end, x_start:x_end]
                            if conv_region.shape != self.weights[kernel].shape:
                                raise ValueError(
                                    f"Shape mismatch between input slice {conv_region.shape} "
                                    f"and weights {self.weights[kernel].shape}")

                            output_tensor[batch, kernel, i, j] = \
                                np.sum(conv_region * self.weights[kernel]) + self.bias[kernel]
            
            for batch in range(b):
                for kernel in range(self.num_kernels):
                    for channel in range(c):
                        conv_result = convolve2d(input_tensor[batch, channel],
                                                 self.weights[kernel, channel], mode='valid')
                        for i in range(output_shape_y):
                            for j in range(output_shape_x):
                                # Get the starting points for the current stride
                                y_start = i * stride_y
                                x_start = j * stride_x
                                output_tensor[batch, kernel, i, j] += conv_result[y_start, x_start]

                    output_tensor[batch, kernel] += self.bias[kernel]
            '''
            '''
            y, x = input_tensor.shape[2:4]  # Spatial dimensions (y, x)
            m, n = self.conv_shape[1], self.conv_shape[2]  # Kernel sizes
            stride_y, stride_x = self.stride_shape

            # Calculate output height and width
            output_height = (y - m) // stride_y + 1
            output_width = (x - n) // stride_x + 1

            # Initialize output tensor
            output_tensor = np.zeros((b, self.num_kernels, output_height, output_width))

            for batch in range(b):
                for kernel in range(self.num_kernels):
                    output = np.zeros((output_height, output_width))
                    for channel in range(c):
                        # Perform 2D convolution for each channel
                        input_2d = np.squeeze(input_tensor[batch, channel])
                        conv_result = convolve2d(input_2d, self.weights[kernel, channel],
                                                 mode='valid')

                        # Apply strides
                        output += conv_result[::stride_y, ::stride_x]

                    # Add bias
                    output_tensor[batch, kernel] = output + self.bias[kernel]
            '''
            b, c, height, width = input_tensor.shape
            kernel_height, kernel_width = self.conv_shape[-2], self.conv_shape[-1]
            stride_y, stride_x = self.stride_shape

            # Calculate output shape
            output_height = (height - kernel_height) // stride_y + 1
            output_width = (width - kernel_width) // stride_x + 1
            output_tensor = np.zeros((b, self.num_kernels, output_height, output_width))

            for batch in range(b):
                for kernel in range(self.num_kernels):
                    output = np.zeros((output_height, output_width))
                    for channel in range(c):
                        # Correlate in 2D (apply filter without flipping the kernel)
                        conv_result = correlate(input_tensor[batch, channel],
                                                self.weights[kernel, channel],
                                                mode='valid')

                        # Apply strides and accumulate results
                        conv_result_strided = conv_result[::stride_y, ::stride_x]
                        output += conv_result_strided

                    # Add bias after the correlation
                    output_tensor[batch, kernel] = output + self.bias[kernel]

        return output_tensor

    def backward(self, grad_output):
        # self._gradient_weights = np.zeros_like(self.weights)
        # self._gradient_bias = np.zeros_like(self.bias)
        # b, c = self.input_tensor.shape[:2]

        if self.is_1d:
            '''
            b, c, y = self.input_tensor.shape
            m = self.conv_shape[1]
            stride_y = self.stride_shape[0]
            output_shape = error_tensor.shape[2]  # Error tensor shape

            # Calculate gradients w.r.t. weights, bias and propagate the error
            propagated_error = np.zeros_like(self.input_tensor)
            for batch in range(b):
                for kernel in range(self.num_kernels):
                    for i in range(output_shape):
                        y_start = i * stride_y
                        y_end = y_start + m

                        self._gradient_weights[kernel] += self.input_tensor[batch, :, y_start:y_end] \
                                                          * error_tensor[batch, kernel, i]
                        self._gradient_bias[kernel] += error_tensor[batch, kernel, i]

                        propagated_error[batch, :, y_start:y_end] \
                            += self.weights[kernel] * error_tensor[batch, kernel, i]
            '''
            '''
            y = self.input_tensor.shape[2]  # Spatial dimension (y-axis)
            m = self.conv_shape[1]  # Kernel size
            stride_y = self.stride_shape[0]

            # Initialize gradient tensors
            grad_input = np.zeros_like(self.input_tensor)
            self._gradient_weights = np.zeros_like(self.weights)
            self._gradient_bias = np.sum(error_tensor, axis=(0, 2))

            for batch in range(b):
                for kernel in range(self.num_kernels):
                    for channel in range(c):
                        # Perform full convolution to calculate gradients for input
                        grad_input[batch, channel] += convolve(error_tensor[batch, kernel],
                                                               self.weights[kernel, channel],
                                                               mode='valid')

                        # Gradient w.r.t. weights (flip error and input for cross-correlation)
                        flipped_input = np.flip(self.input_tensor[batch, channel])
                        self._gradient_weights[kernel, channel] += \
                            convolve(flipped_input, error_tensor[batch, kernel], mode='valid')
            '''
            b, c, length = self.input_tensor.shape
            kernel_size = self.conv_shape[-1]
            stride_y = self.stride_shape

            grad_input = np.zeros_like(self.input_tensor)
            grad_weights = np.zeros_like(self.weights)
            grad_bias = np.zeros_like(self.bias)

            for batch in range(b):
                for kernel in range(self.num_kernels):
                    # Calculate bias gradient (sum over the output)
                    grad_bias[kernel] += np.sum(grad_output[batch, kernel])

                    for channel in range(c):
                        # Convolve for gradient of weights
                        grad_w = convolve(self.input_tensor[batch, channel],
                                          grad_output[batch, kernel], mode='valid')
                        grad_weights[kernel, channel] += grad_w

                        # Convolve for gradient of input
                        grad_input_conv = convolve(grad_output[batch, kernel],
                                                   self.weights[kernel, channel],
                                                   mode='full')
                        grad_input[batch, channel] += grad_input_conv[::stride_y]

        else:
            '''
            b, c, y, x = self.input_tensor.shape
            m, n = self.conv_shape[1], self.conv_shape[2]
            stride_y, stride_x = self.stride_shape
            output_shape = (error_tensor.shape[2], error_tensor.shape[3])  # Error tensor shape

            # Calculate gradients and propagate the error for 2D
            propagated_error = np.zeros_like(self.input_tensor)
            for batch in range(b):
                for kernel in range(self.num_kernels):
                    for i in range(output_shape[0]):
                        for j in range(output_shape[1]):
                            y_start, x_start = i * stride_y, j * stride_x
                            y_end, x_end = y_start + m, x_start + n

                            # Accumulate gradients
                            self._gradient_weights[kernel] += \
                                self.input_tensor[batch, :, y_start:y_end, x_start:x_end] \
                                * error_tensor[batch, kernel, i, j]
                            self._gradient_bias[kernel] += error_tensor[batch, kernel, i, j]

                            # Propagate error to the previous layer
                            propagated_error[batch, :, y_start:y_end, x_start:x_end] += \
                                self.weights[kernel] * error_tensor[batch, kernel, i, j]

        if self._optimizer_weights and self._optimizer_bias:
            self.weights = self._optimizer_weights.update(self.weights, self._gradient_weights)
            self.bias = self._optimizer_bias.update(self.bias, self._gradient_bias)
        '''
            '''
            y, x = self.input_tensor.shape[2:4]  # Spatial dimensions (y, x)
            m, n = self.conv_shape[1], self.conv_shape[2]  # Kernel sizes
            stride_y, stride_x = self.stride_shape

            # Initialize gradient tensors
            grad_input = np.zeros_like(self.input_tensor)
            self._gradient_weights = np.zeros_like(self.weights)
            self._gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))

            for batch in range(b):
                for kernel in range(self.num_kernels):
                    for channel in range(c):
                        # Perform full convolution to calculate gradients for input
                        grad_input[batch, channel] += convolve2d(error_tensor[batch, kernel],
                                                                 self.weights[kernel, channel],
                                                                 mode='valid')

                        # Gradient w.r.t. weights (flip error and input for cross-correlation)
                        flipped_input = np.flip(self.input_tensor[batch, channel])
                        self._gradient_weights[kernel, channel] += \
                            convolve2d(flipped_input, error_tensor[batch, kernel], mode='valid')
            '''
            b, c, height, width = self.input_tensor.shape
            kernel_height, kernel_width = self.conv_shape[-2], self.conv_shape[-1]
            stride_y, stride_x = self.stride_shape

            grad_input = np.zeros_like(self.input_tensor)
            grad_weights = np.zeros_like(self.weights)
            grad_bias = np.zeros_like(self.bias)

            for batch in range(b):
                for kernel in range(self.num_kernels):
                    # Calculate bias gradient (sum over the output)
                    grad_bias[kernel] += np.sum(grad_output[batch, kernel])

                    for channel in range(c):
                        # Convolve for gradient of weights
                        grad_w = convolve(self.input_tensor[batch, channel],
                                          grad_output[batch, kernel], mode='valid')
                        grad_weights[kernel, channel] += grad_w

                        # Convolve for gradient of input
                        grad_input_conv = convolve(grad_output[batch, kernel],
                                                   self.weights[kernel, channel],
                                                   mode='full')
                        grad_input[batch, channel] += grad_input_conv[::stride_y, ::stride_x]

        # return propagated_error
        return grad_input, grad_weights, grad_bias

    '''
    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.conv_shape)
        fan_out = np.prod(self.conv_shape[1:]) * self.num_kernels
        self.weights = weights_initializer.initialize(self.conv_shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape)
    '''
    def initialize(self, weights_initializer, bias_initializer):
        self.fan_in = np.prod(self.conv_shape)
        self.fan_out = np.prod(self.conv_shape[1:]) * self.num_kernels
        self.weights = weights_initializer.initialize(self.weights.shape, self.fan_in, self.fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, self.fan_in, self.fan_out)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    '''
    @property
    def optimizer(self):
        return self._optimizer_weights

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer_weights
        self._optimizer_bias

    def update_parameters(self, learning_rate):
        if self.trainable:
            self.weights -= learning_rate * self._gradient_weights
            self.bias -= learning_rate * self._gradient_bias
    '''