import os
import torch
import torch.nn as nn
from . import block

class Generator(nn.Module):
    def __init__(self, dimension, spatial_channel):
        super().__init__()
        
        self.dimension = dimension
        self.spatial_channel = spatial_channel
        
        self.encoder_channels = [self.spatial_channel, 56, 112, 224, 448]
        self.encoder = nn.ModuleList()
        for i in range(1, len(self.encoder_channels)):
            self.encoder.append(block.Conv2D(self.encoder_channels[i-1], self.encoder_channels[i]))
        
        self.resnet = nn.ModuleList()
        self.n_resnet = 9
        for i in range(self.n_resnet):
            self.resnet.append(block.ResNet(self.encoder_channels[-1]))
        
        self.decoder_channels = [self.encoder_channels[-1], 224, 112, 56, 3]
        self.decoder = nn.ModuleList()
        for i in range(1, len(self.decoder_channels)):
            self.decoder.append(block.ConvTrans2D(self.decoder_channels[i-1], self.decoder_channels[i]))
        
    def forward(self, x):
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
        for i in range(len(self.resnet)):
            x = self.resnet[i](x)
        for i in range(len(self.decoder)):
            x = self.decoder[i](x)
        x = torch.sigmoid(x)
        return x
        
class Discriminator(nn.Module):
    def __init__(self, dimension, spatial_channel, avgpool):
        super().__init__()
        
        self.dimension = dimension
        self.spatial_channel = spatial_channel
        self.avgpool = avgpool
        
        self.pool = nn.ModuleList()
        for i in range(avgpool):
            self.pool.append(nn.AvgPool2d(kernel_size=4, stride=2, padding=1))
            
        self.dis_channels = [self.spatial_channel + 3, 64, 128, 256, 512]
        self.dis = nn.ModuleList()
        for i in range(1, len(self.dis_channels)):
            self.dis.append(block.Conv2D(self.dis_channels[i-1], self.dis_channels[i]))
            
    def forward(self, x):
        for i in range(len(self.pool)):
            x = self.pool[i](x)
        for i in range(len(self.dis)):
            x = self.dis[i](x)
        x = torch.sigmoid(x)
        return x

class Module(nn.Module):
    
    def __init__(self, generator=True, discriminator=False):
        super().__init__()
        self.G = None
        self.D1 = None
        self.D2 = None
        self.D3 = None
        
        self.dimension = 512
        self.spatial_channel = 32
        
        if generator:
            self.G = Generator(self.dimension, self.spatial_channel)
        
        if discriminator:
            self.D1 = Discriminator(self.dimension, self.spatial_channel, avgpool=0)
            self.D2 = Discriminator(self.dimension, self.spatial_channel, avgpool=1)
            self.D3 = Discriminator(self.dimension, self.spatial_channel, avgpool=2)
            self.label_real = 1
            self.label_fake = 0
            
    def forward(self, x):
        return self.generate(x)
    
    def generate(self, spatial_map):
        assert spatial_map.shape[1:] == (self.spatial_channel, self.dimension, self.dimension), f'[Image Synthesis : generate] Expected input spatial_map shape {(-1, self.spatial_channel, self.dimension, self.dimension)}, but received {spatial_map.shape}.'
        photo = self.G(spatial_map)
        assert photo.shape[1:] == (3, self.dimension, self.dimension), f'[Image Synthesis : generate] Expected output shape {(1, 3, self.dimension, self.dimension)}, but yield {photo.shape}.'
        return photo
    
    def discriminate(self, spatial_map, photo):
        assert spatial_map.shape[0] == photo.shape[0], f'[Image Synthesis : discriminate] Input spatial_map has {spatial_map.shape[0]} batch(es), but photo has {photo.shape[0]} batch(es).'
        assert spatial_map.shape[1:] == (self.spatial_channel, self.dimension, self.dimension), f'[Image Synthesis : discriminate] Expected input spatial_map shape {(-1, self.spatial_channel, self.dimension, self.dimension)}, but received {spatial_map.shape}.'
        assert photo.shape[1:] == (3, self.dimension, self.dimension), f'[Image Synthesis : discriminate] Expected input photo shape {(-1, 3, self.dimension, self.dimension)}, but received {photo.dimension}.'
        
        spatial_map_photo = torch.cat((spatial_map, photo), 1)
        patch_D1 = self.D1(spatial_map_photo)
        patch_D2 = self.D2(spatial_map_photo)
        patch_D3 = self.D3(spatial_map_photo)
        return patch_D1, patch_D2, patch_D3
    
    path_dict = {
        'G' : 'generator.pth',
        'D1' : 'discriminator_1.pth',
        'D2' : 'discriminator_2.pth',
        'D3' : 'discriminator_3.pth'
    }
    
    def get_path(self, path, key):
        return os.path.join(path, self.path_dict[key])
    
    def save_G(self, path):
        torch.save(self.G.state_dict(), path)
        print(f'Saved Image Synthesis : G to {path}')
    
    def save_D1(self, path):
        torch.save(self.D1.state_dict(), path)
        print(f'Saved Image Synthesis : D1 to {path}')
    
    def save_D2(self, path):
        torch.save(self.D2.state_dict(), path)
        print(f'Saved Image Synthesis : D2 to {path}')
    
    def save_D3(self, path):
        torch.save(self.D3.state_dict(), path)
        print(f'Saved Image Synthesis : D3 to {path}')
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        if self.G: self.save_G(self.get_path(path, 'G'))
        if self.D1: self.save_D1(self.get_path(path, 'D1'))
        if self.D2: self.save_D2(self.get_path(path, 'D2'))
        if self.D3: self.save_D3(self.get_path(path, 'D3'))
        
    def load_G(self, path, map_location=torch.device('cpu')):
        self.G.load_state_dict(torch.load(path, map_location=map_location))
        print(f'Loaded Image Synthesis : G from {path}')
    
    def load_D1(self, path, map_location=torch.device('cpu')):
        self.D1.load_state_dict(torch.load(path, map_location=map_location))
        print(f'Loaded Image Synthesis : D1 from {path}')
    
    def load_D2(self, path, map_location=torch.device('cpu')):
        self.D2.load_state_dict(torch.load(path, map_location=map_location))
        print(f'Loaded Image Synthesis : D2 from {path}')
    
    def load_D3(self, path, map_location=torch.device('cpu')):
        self.D3.load_state_dict(torch.load(path, map_location=map_location))
        print(f'Loaded Image Synthesis : D3 from {path}')
    
    def load(self, path, map_location=torch.device('cpu')):
        if self.G: self.load_G(self.get_path(path, 'G'), map_location=map_location)
        if self.D1: self.load_D1(self.get_path(path, 'D1'), map_location=map_location)
        if self.D2: self.load_D2(self.get_path(path, 'D2'), map_location=map_location)
        if self.D3: self.load_D3(self.get_path(path, 'D3'), map_location=map_location)