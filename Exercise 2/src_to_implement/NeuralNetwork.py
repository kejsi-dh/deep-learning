import copy
import numpy as np

class NeuralNetwork:
    def __init__(self, optimiser):
        self.optimiser = optimiser # for w updates
        self.loss = []  # for loss values for each iteration
        self.layers = []  # for holding the architecture
        self.data_layer = None  # data layer for input data and labels
        self.loss_layer = None  # loss layer for computing loss and predictions

    def forward(self):
        in_tensor, self.label_tensor = self.data_layer.next()
        for layer in self.layers: in_tensor = layer.forward(in_tensor)

        loss_val = self.loss_layer.forward(in_tensor, self.label_tensor)
        return loss_val

    def backward(self):
        if self.loss_layer is not None:
            error_tensor = self.loss_layer.backward(self.label_tensor)

        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable: layer.optimiser = copy.deepcopy(self.optimiser)
        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            loss_val = self.forward()
            self.loss.append(loss_val)
            self.backward()

    def test(self, in_tensor):
        for layer in self.layers: in_tensor = layer.forward(in_tensor)
        return in_tensor
