import torch
import torch.nn as nn
import torch.nn.functional as F

def Conv2D_Block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1)     
    )

def ConvTrans2D_Block(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1)
    )

class Resnet_Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = x + residual
        return x  

class Encoder(nn.Module):
    def __init__(self, channels, input_dimension, latent_dimension):
        super().__init__()
        
        self.channels = channels
        self.input_dimension = input_dimension
        self.output_dimension = input_dimension
        self.latent_dimension = latent_dimension
        
        self.encoder = nn.ModuleList()
        for i in range(1, len(channels)):
            self.encoder.append(Conv2D_Block(self.channels[i-1], self.channels[i]))
            self.encoder.append(Resnet_Block(self.channels[i]))
            self.output_dimension = int(self.output_dimension / 2)
            
        self.latent = nn.Linear(self.channels[-1] * self.output_dimension * self.output_dimension, self.latent_dimension)
    
    def forward(self, x):
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
        x = torch.flatten(x, 1)
        x = self.latent(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, channels, input_dimension, latent_dimension):
        super().__init__()
        
        self.channels = channels
        self.input_dimension = input_dimension
        self.output_dimension = input_dimension
        self.latent_dimension = latent_dimension
        
        self.decoder = nn.ModuleList()
        for i in range(1, len(channels)):
            self.decoder.append(Resnet_Block(self.channels[i-1]))
            self.decoder.append(ConvTrans2D_Block(self.channels[i-1], self.channels[i]))
            self.output_dimension = int(self.output_dimension * 2)
            
        self.latent = nn.Linear(self.latent_dimension, self.channels[0] * self.input_dimension * self.input_dimension)
    
    def forward(self, x):
        x = self.latent(x)
        x = torch.reshape(x, (x.shape[0], self.channels[0], self.input_dimension, self.input_dimension))
        for i in range(len(self.decoder)):
            x = self.decoder[i](x)
        return x

def ReflectionPad2d(source_dimension, target_dimension):
    dif = target_dimension - source_dimension
    padding_left = padding_right = padding_top = padding_bottom = int(dif / 2)
    if dif%2: padding_right = padding_bottom = int(dif / 2) + 1
    return nn.ReflectionPad2d((padding_left, padding_right, padding_top, padding_bottom))
    
class ComponentEmbedding(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        
        self.dimension = dimension
        self.latent_dimension = 512
        
        self.encoder_channels = [1, 32, 64, 128, 256, 512]
        self.decoder_channels = [512, 256, 128, 64, 32, 32]
        
        self.encoder = Encoder(self.encoder_channels, self.dimension, self.latent_dimension)
        self.decoder = Decoder(self.decoder_channels, self.encoder.output_dimension, self.latent_dimension)
        self.output = nn.Sequential(
            Resnet_Block(self.decoder_channels[-1]),
            ReflectionPad2d(self.decoder.output_dimension, self.dimension + 4),
            nn.Conv2d(self.decoder_channels[-1], 1, kernel_size=5),
        )
        
    def forward(self, x):
        x = self.Encode(x)
        x = self.Decode(x)
        return x
    
    def Encode(self, sketch):
        assert(sketch.shape[1:] == (1, self.dimension, self.dimension))
        latent = self.encoder(sketch)
        return latent
    
    def Decode(self, latent):
        assert(latent.shape[1:] == (self.latent_dimension,))
        sketch = self.decoder(latent)
        sketch = self.output(sketch)
        return sketch

class FeatureMapping(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        
        self.dimension = dimension
        self.latent_dimension = 512
        self.spatial_channel = 32
        
        self.encoder_channels = [1, 32, 64, 128, 256, 512]
        self.decoder_channels = [512, 256, 256, 128, 64, 64]
        
        self.encoder = Encoder(self.encoder_channels, self.dimension, self.latent_dimension)
        self.decoder = Decoder(self.decoder_channels, self.encoder.output_dimension, self.latent_dimension)
        self.output = nn.Sequential(
            Resnet_Block(self.decoder_channels[-1]),
            ReflectionPad2d(self.decoder.output_dimension, self.dimension + 4),
            nn.Conv2d(self.decoder_channels[-1], self.spatial_channel, kernel_size=5),
        )
        
        del self.encoder, self.encoder_channels
        
    def forward(self, x):
        x = self.Decode(x)
        return x
    
    def Decode(self, latent):
        assert(latent.shape[1:] == (self.latent_dimension,))
        sketch = self.decoder(latent)
        sketch = self.output(sketch)
        return sketch

class Generator(nn.Module):
    def __init__(self, dimension, spatial_channel):
        super().__init__()
        
        self.dimension = dimension
        self.spatial_channel = spatial_channel
        
        self.encoder_channels = [self.spatial_channel, 56, 112, 224, 448]
        self.encoder = nn.ModuleList()
        for i in range(1, len(self.encoder_channels)):
            self.encoder.append(Conv2D_Block(self.encoder_channels[i-1], self.encoder_channels[i]))
        
        self.resnet = nn.ModuleList()
        self.resnet_n = 9
        for i in range(self.resnet_n):
            self.resnet.append(Resnet_Block(self.encoder_channels[-1]))
        
        self.decoder_channels = [self.encoder_channels[-1], 224, 112, 56, 3]
        self.decoder = nn.ModuleList()
        for i in range(1, len(self.decoder_channels)):
            self.decoder.append(ConvTrans2D_Block(self.decoder_channels[i-1], self.decoder_channels[i]))
        
    def forward(self, x):
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
        for i in range(len(self.resnet)):
            x = self.resnet[i](x)
        for i in range(len(self.decoder)):
            x = self.decoder[i](x)
        return x
        
class Discriminator(nn.Module):
    def __init__(self, dimension, spatial_channel, avgpool=0):
        super().__init__()
        
        self.dimension = dimension
        self.spatial_channel = spatial_channel
        self.avgpool = avgpool
        
        self.input_dimension = self.dimension
        self.output_dimension = self.dimension
        
        self.pool = nn.ModuleList()
        for i in range(avgpool):
            self.pool.append(nn.AvgPool2d(kernel_size=4, stride=2, padding=1))
            self.output_dimension = int(self.output_dimension / 2)
        
        self.dis_channels = [self.spatial_channel + 3, 64, 128, 256, 512]
        self.dis = nn.ModuleList()
        for i in range(1, len(self.dis_channels)):
            self.dis.append(Conv2D_Block(self.dis_channels[i-1], self.dis_channels[i]))
            self.output_dimension = int(self.output_dimension / 2)
    
    def forward(self, x):
        for i in range(len(self.pool)):
            x = self.pool[i](x)
        for i in range(len(self.dis)):
            x = self.dis[i](x)
        return x

class ImageSynthesis(nn.Module):
    def __init__(self, generator=True, discriminator=False):
        super().__init__()
        
        self.dimension = 512
        self.spatial_channel = 32
        
        if generator:
            self.G = Generator(self.dimension, self.spatial_channel)
        
        if discriminator:
            self.D1 = Discriminator(self.dimension, self.spatial_channel, avgpool=0)
            self.D2 = Discriminator(self.dimension, self.spatial_channel, avgpool=1)
            self.D3 = Discriminator(self.dimension, self.spatial_channel, avgpool=2)
            
    def forward(self, x):
        return self.Generate(x)
    
    def Generate(self, spatial_map):
        assert(spatial_map.shape[1:] == (self.spatial_channel, self.dimension, self.dimension))
        photo = self.G(spatial_map)
        return photo
    
    def Discriminate(self, spatial_map, photo):
        assert(spatial_map.shape[0] == photo.shape[0])
        assert(spatial_map.shape[1:] == (self.spatial_channel, self.dimension, self.dimension))
        assert(photo.shape[1:] == (3, self.dimension, self.dimension))
        
        spatial_map_photo = torch.cat((spatial_map, photo), 1)
        patch_D1 = self.D1(spatial_map_photo)
        patch_D2 = self.D2(spatial_map_photo)
        patch_D3 = self.D3(spatial_map_photo)
        
        return patch_D1, patch_D2, patch_D3
        
def main():
    CE = ComponentEmbedding(512)
    FM = FeatureMapping(512)
    IM = ImageSynthesis(generator=True, discriminator=True)
    
    sketch = torch.rand(8, 1, 512, 512)
    print("CE Forward", CE(sketch).shape)
    
    latent = CE.Encode(sketch)
    print("CE Encode", latent.shape)
    
    spatial_map = FM.Decode(latent)
    print("FM Decode", spatial_map.shape)
    
    photo = IM.Generate(spatial_map)
    print("IM Generate", photo.shape)
    
    patch_D1, patch_D2, patch_D3 = IM.Discriminate(spatial_map, photo)
    print("IM Patch D1", patch_D1.shape)
    print("IM Patch D2", patch_D2.shape)
    print("IM Patch D3", patch_D3.shape)
    
    print()
    
    dim = 168  # dim: 128, 168, 192, 512
    
    CEs = ComponentEmbedding(dim)
    FMs = FeatureMapping(dim)
    
    print(f"CE and FM with dim={dim}")
    
    sketch = torch.rand(8, 1, dim, dim)
    print("CEs Forward", CEs(sketch).shape)
    
    latent = CEs.Encode(sketch)
    print("CEs Encode", latent.shape)
    
    spatial_map = FMs.Decode(latent)
    print("FMs Decode", spatial_map.shape)
    
if __name__ == '__main__':
    main()
    
'''
Expected Output:

CE Forward torch.Size([8, 1, 512, 512])
CE Encode torch.Size([8, 512])
FM Decode torch.Size([8, 32, 512, 512])
IM Generate torch.Size([8, 3, 512, 512])
IM Patch D1 torch.Size([8, 512, 32, 32])
IM Patch D2 torch.Size([8, 512, 16, 16])
IM Patch D3 torch.Size([8, 512, 8, 8])

CE and FM with dim=168
CEs Forward torch.Size([8, 1, 168, 168])
CEs Encode torch.Size([8, 512])
FMs Decode torch.Size([8, 32, 168, 168])

'''