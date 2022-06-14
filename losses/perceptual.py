import torch
from torchvision import transforms
from .mae import MAE

class Perceptual:
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    perceptual_layer = ['4', '9', '14', '19']

    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
        self.model.to(self.device)
        self.criterion = MAE()
    
    def compute(self, prediction, ground_truth):
        prediction = self.preprocess(prediction)
        ground_truth = self.preprocess(ground_truth)
        
        loss = 0
        for layer, module in self.model.features._modules.items():
            prediction = module(prediction)
            ground_truth = module(ground_truth)
            if layer in self.perceptual_layer:
                loss += self.criterion.compute(prediction, ground_truth)
        return loss