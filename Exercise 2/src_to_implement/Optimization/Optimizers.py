import numpy as np

# sgd
class Sgd:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        updated_weights = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weights

# sgd with momentum
class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.velocity is None:
            self.velocity = np.zeros_like(weight_tensor)

        self.velocity = self.momentum_rate * self.velocity - self.learning_rate * gradient_tensor
        weight_tensor += self.velocity
        return weight_tensor

# adam
class Adam:
    def __init__(self, learning_rate, mu=0.9, rho=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.mu = mu # beta1
        self.rho = rho # beta2
        self.epsilon = epsilon
        self.grad_mean = None # mean of the gradients; first moment
        self.squared_mean = None # mean of squared gradients; second moment
        self.t = 0 # time step counter

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.grad_mean is None:
            self.grad_mean = np.zeros_like(weight_tensor)
        if self.squared_mean is None:
            self.squared_mean = np.zeros_like(weight_tensor)

        self.t += 1

        self.grad_mean = self.mu * self.grad_mean + (1 - self.mu) * gradient_tensor
        self.squared_mean = self.rho * self.squared_mean + (1 - self.rho) * np.square(gradient_tensor)

        # bias-corrected moments
        bias_gm = self.grad_mean / (1 - self.mu**self.t)
        bias_sm = self.squared_mean / (1 - self.rho**self.t)

        weight_tensor -= self.learning_rate * bias_gm / (np.sqrt(bias_sm) + self.epsilon)
        return weight_tensor
