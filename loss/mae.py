import torch

class L1:
    def __init__(self):
        self.criterion = torch.nn.L1Loss()

    def compute(self, prediction, ground_truth):
        return self.criterion(prediction, ground_truth)