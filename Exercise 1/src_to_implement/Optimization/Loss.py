import numpy as np

class CrossEntropyLoss:
    def forward(self, pred_tensor, label_tensor):
        self.pred_tensor = pred_tensor
        return np.sum(-np.log(pred_tensor[label_tensor == 1] + np.finfo(float).eps))

    def backward(self, label_tensor):
        return -label_tensor / (self.pred_tensor + np.finfo(float).eps)