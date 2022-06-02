import torch.nn as nn
import torch.nn.functional as F

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, equal=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ConvTrans2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convT = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.convT(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = x + residual
        return x