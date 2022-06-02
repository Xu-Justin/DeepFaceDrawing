import torch
import torch.nn as nn
from . import block

class Encoder(nn.Module):
    def __init__(self, channels, input_dimension, latent_dimension):
        super().__init__()
        
        self.channels = channels
        self.input_dimension = input_dimension
        self.conv_dimension = input_dimension
        self.latent_dimension = latent_dimension
        
        self.encoder_list = nn.ModuleList()
        for i in range(1, len(self.channels)):
            self.encoder_list.append(block.Conv2D(self.channels[i-1], self.channels[i]))
            self.encoder_list.append(block.ResNet(self.channels[i]))
            self.conv_dimension = self.conv_dimension // 2
        
        self.latent = nn.Linear(self.channels[-1] * self.conv_dimension * self.conv_dimension, self.latent_dimension)
        
    def forward(self, x):
        for encoder in self.encoder_list:
            x = encoder(x)
        x = torch.flatten(x, 1)
        x = self.latent(x)
        return x
    
class Decoder(nn.Module):
    
    def ReflectionPad2d(self, source_dimension, target_dimension):
        dif = target_dimension - source_dimension
        padding_left = padding_right = padding_top = padding_bottom = dif // 2
        if dif%2: padding_right = padding_bottom = (dif // 2) + 1
        return nn.ReflectionPad2d((padding_left, padding_right, padding_top, padding_bottom))
    
    def __init__(self, channels, input_dimension, output_dimension, latent_dimension):
        super().__init__()
        
        self.channels = channels
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.conv_dimension = input_dimension
        self.latent_dimension = latent_dimension
        
        self.decoder_list = nn.ModuleList()
        for i in range(1, len(self.channels)-1):
            self.decoder_list.append(block.ResNet(self.channels[i-1]))
            self.decoder_list.append(block.ConvTrans2D(self.channels[i-1], self.channels[i]))
            self.conv_dimension = self.conv_dimension * 2
        
        self.latent = nn.Linear(self.latent_dimension, self.channels[0] * self.input_dimension * self.input_dimension)
        
        self.output = nn.Sequential(
            block.ResNet(self.channels[-2]),
            self.ReflectionPad2d(self.conv_dimension, self.output_dimension),
            nn.Conv2d(self.channels[-2], self.channels[-1], kernel_size=4, stride=1, padding='same', padding_mode='reflect')
        )
        
    def forward(self, x):
        x = self.latent(x)
        x = torch.reshape(x, (x.shape[0], self.channels[0], self.input_dimension, self.input_dimension))
        for decoder in self.decoder_list:
            x = decoder(x)
        x = self.output(x)
        return x