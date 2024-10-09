import numpy as np

class Constant:
    def __init__(self, const_val=0.1):
        self.const_val = const_val

    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        return np.full(weights_shape, self.const_val)

class UniformRandom:
    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        return np.random.uniform(0, 1, size=weights_shape)

class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        lim = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, lim, size=weights_shape)

class He:
    def initialize(self, weights_shape, fan_in, fan_out=None):
        std = np.sqrt(2 / fan_in)
        return np.random.randn(*weights_shape) * std
