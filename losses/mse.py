import torch

class MSE:
    def __init__(self):
        self.criterion = torch.nn.MSELoss()
    
    def compute(self, prediction, ground_truth):
        return self.criterion(prediction, ground_truth)