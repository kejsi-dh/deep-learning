import numpy as np
from .Base import *

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        if isinstance(stride_shape, int):
            self.stride_shape = (stride_shape, stride_shape)
        else: self.stride_shape = stride_shape

        if isinstance(pooling_shape, int):
            self.pooling_shape = (pooling_shape, pooling_shape)
        else: self.pooling_shape = pooling_shape

        self.max_indices = None
        self.input_shape = None

    # input tensor --> columns where each corresponds to a pooling region
    def im2col(self, in_tensor, filter_size, stride):
        batch, channels, height, width = in_tensor.shape
        fheight, fwidth = filter_size
        sy, sx = stride

        out_h = (height - fheight) // sy + 1
        out_w = (width - fwidth) // sx + 1

        col = np.zeros((batch, channels, fheight, fwidth, out_h, out_w))

        for y in range(fheight):
            y_max = y + sy * out_h
            for x in range(fwidth):
                x_max = x + sx * out_w
                col[:, :, y, x, :, :] = in_tensor[:, :, y:y_max:sy, x:x_max:sx]

        col = col.reshape(batch, channels, fheight * fwidth, out_h * out_w)
        return col

    def forward(self, in_tensor):
        self.in_shape = in_tensor.shape
        self.col = self.im2col(in_tensor, self.pooling_shape, self.stride_shape)

        # max pooling on cols
        self.max_indices = np.argmax(self.col, axis=2)
        out_tensor = np.max(self.col, axis=2).reshape(self.in_shape[0], self.in_shape[1], -1)

        # reshape
        output_y = (self.in_shape[2] - self.pooling_shape[0]) // self.stride_shape[0] + 1
        output_x = (self.in_shape[3] - self.pooling_shape[1]) // self.stride_shape[1] + 1
        out_tensor = out_tensor.reshape(self.in_shape[0], self.in_shape[1], output_y, output_x)

        return out_tensor

    # max indices --> input tensor
    def col2im(self, cols, in_shape, filter_size, stride):
        batch, channels, height, width = in_shape
        fheight, fwidth = filter_size
        sy, sx = stride

        out_h = (height - fheight) // sy + 1
        out_w = (width - fwidth) // sx + 1
        cols_reshaped = cols.reshape(batch, channels, fheight, fwidth, out_h, out_w)

        out = np.zeros(in_shape)

        for y in range(fheight):
            y_max = y + sy * out_h
            for x in range(fwidth):
                x_max = x + sx * out_w
                out[:, :, y:y_max:sy, x:x_max:sx] += cols_reshaped[:, :, y, x, :, :]

        return out

    def backward(self, error_tensor):
        flat_error_tensor = error_tensor.reshape(self.input_shape[0], self.input_shape[1], -1)
        error_col = np.zeros_like(self.col)

        batch = self.col.shape[0]
        channels = self.col.shape[1]

        # scatter error into the locations of the max values
        for b in range(batch):
            for c in range(channels):
                error_col[b, c, self.max_indices[b, c], np.arange(self.max_indices.shape[2])] = \
                    flat_error_tensor[b, c, :]

        propagated_error = self.col2im(error_col, self.input_shape,
                                       self.pooling_shape, self.stride_shape)

        return propagated_error
