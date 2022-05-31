import torch

class BCE:
    def __init__(self):
        self.criterion = torch.nn.BCELoss()

    def compute(self, prediction, ground_truth):
        return self.criterion(prediction, ground_truth)