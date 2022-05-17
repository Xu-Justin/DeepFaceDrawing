import torch
import torch.nn as nn
import torch.nn.functional as F

import os

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
            self.output_dimension = self.output_dimension // 2
            
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
            self.output_dimension = self.output_dimension * 2
            
        self.latent = nn.Linear(self.latent_dimension, self.channels[0] * self.input_dimension * self.input_dimension)
    
    def forward(self, x):
        x = self.latent(x)
        x = torch.reshape(x, (x.shape[0], self.channels[0], self.input_dimension, self.input_dimension))
        for i in range(len(self.decoder)):
            x = self.decoder[i](x)
        return x

def ReflectionPad2d(source_dimension, target_dimension):
    dif = target_dimension - source_dimension
    padding_left = padding_right = padding_top = padding_bottom = dif // 2
    if dif%2: padding_right = padding_bottom = (dif // 2) + 1
    return nn.ReflectionPad2d((padding_left, padding_right, padding_top, padding_bottom))
    
class ComponentEmbedding(nn.Module):
    
    def __init__(self, dimension, encoder=True, decoder=True, latent_dimension=512, verify=False):
        super().__init__()
        self.encoder = None
        self.decoder = None
        self.output = None
        
        self.dimension = dimension
        self.latent_dimension = latent_dimension
        
        if encoder:
            self.init_encoder()
            if verify: self.verify_encoder()
        
        if decoder:
            self.init_decoder()
            if verify: self.verify_decoder()
    
    def init_encoder(self):
        self.encoder_channels = [1, 32, 64, 128, 256, 512]
        self.encoder = Encoder(self.encoder_channels, self.dimension, self.latent_dimension)
        
    def init_decoder(self):
        self.decoder_channels = [512, 256, 128, 64, 32, 32]
        self.decoder = Decoder(self.decoder_channels, self.encoder.output_dimension, self.latent_dimension)
        self.output = nn.Sequential(
            Resnet_Block(self.decoder_channels[-1]),
            ReflectionPad2d(self.decoder.output_dimension, self.dimension + 4),
            nn.Conv2d(self.decoder_channels[-1], 1, kernel_size=5),
        )
        
    def delete_encoder(self):
        del self.encoder_channels
        del self.encoder
    
    def delete_decoder(self):
        del self.decoder_channels
        del self.decoder
        del self.output
    
    def verify_encoder(self):
        x = torch.rand(1, 1, self.dimension, self.dimension)
        x = self.Encode(x)
        assert x.shape == (1, self.latent_dimension), f"{x.shape} != {(1, self.latent_dimension)}"
        
    def verify_decoder(self):
        x = torch.rand(1, self.latent_dimension)
        x = self.Decode(x)
        assert x.shape == (1, 1, self.dimension, self.dimension), f"{x.shape} != {(1, 1, self.dimension, self.dimension)}"
        
    def forward(self, x):
        x = self.Encode(x)
        x = self.Decode(x)
        return x
    
    def Encode(self, sketch):
        assert sketch.shape[1:] == (1, self.dimension, self.dimension), f'{sketch.shape[1:]} != {(1, self.dimension, self.dimension)}'
        latent = self.encoder(sketch)
        return latent
    
    def Decode(self, latent):
        assert latent.shape[1:] == (self.latent_dimension,), f'{latent.shape[1:]} != {(self.latent_dimension,)}'
        sketch = self.decoder(latent)
        sketch = self.output(sketch)
        return sketch
    
    path_dict = {
        'encoder' : 'encoder.pth',
        'decoder' : 'decoder.pth',
        'output' : 'output.pth'
    }
    
    def get_path(self, path, key):
        return os.path.join(path, self.path_dict[key])
    
    def save_encoder(self, path):
        torch.save(self.encoder.state_dict(), path)
        print(f'Saved Component Embedding : encoder to {path}')
    
    def save_decoder(self, path):
        torch.save(self.decoder.state_dict(), path)
        print(f'Saved Component Embedding : decoder to {path}')
    
    def save_output(self, path):
        torch.save(self.output.state_dict(), path)
        print(f'Saved Component Embedding : output to {path}')
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        if self.encoder: self.save_encoder(self.get_path(path, 'encoder'))
        if self.decoder: self.save_decoder(self.get_path(path, 'decoder'))
        if self.output: self.save_output(self.get_path(path, 'output'))
    
    def load_encoder(self, path):
        self.encoder.load_state_dict(torch.load(path))
        print(f'Loaded Component Embedding : encoder from {path}')
    
    def load_decoder(self, path):
        self.decoder.load_state_dict(torch.load(path))
        print(f'Loaded Component Embedding : decoder from {path}')
    
    def load_output(self, path):
        self.output.load_state_dict(torch.load(path))
        print(f'Loaded Component Embedding : output from {path}')
    
    def load(self, path):
        if self.encoder: self.load_encoder(self.get_path(path, 'encoder'))
        if self.decoder: self.load_decoder(self.get_path(path, 'decoder'))
        if self.output: self.load_output(self.get_path(path, 'output'))

class ComponentEmbedding_Master(ComponentEmbedding):
    
    def __init__(self, part, prefix, encoder=True, decoder=True):
        self.master_dimension = 512
        self.part = part
        self.prefix = prefix
        self.crop_dimension = (part[0], part[1], part[0] + part[2], part[1] + part[2]) # xmin, ymin, xmax, ymax
        super().__init__(part[2], encoder, decoder)
    
    def crop(self, sketch):
        assert sketch.shape[1:] == (1, self.master_dimension, self.master_dimension), f'{sketch.shape[1:]} != {(1, self.master_dimension, self.master_dimension)}'
        return sketch[:, :, self.crop_dimension[1]:self.crop_dimension[3], self.crop_dimension[0]:self.crop_dimension[2]].clone()
    
    def save(self, path):
        super().save(os.path.join(path, self.prefix))
    
    def load(self, path):
        super().load(os.path.join(path, self.prefix))
    
class ComponentEmbedding_LeftEye(ComponentEmbedding_Master):
    def __init__(self, encoder=True, decoder=True):
        self.part = (108, 126, 128)
        self.prefix = 'left_eye'
        super().__init__(self.part, self.prefix, encoder, decoder)
    
class ComponentEmbedding_RightEye(ComponentEmbedding_Master):
    def __init__(self, encoder=True, decoder=True):
        self.part = (255, 126, 128)
        self.prefix = 'right_eye'
        super().__init__(self.part, self.prefix, encoder, decoder)

class ComponentEmbedding_Nose(ComponentEmbedding_Master):
    def __init__(self, encoder=True, decoder=True):
        self.part = (182, 232, 160)
        self.prefix = 'nose'
        super().__init__(self.part, self.prefix, encoder, decoder)
    
class ComponentEmbedding_Mouth(ComponentEmbedding_Master):
    def __init__(self, encoder=True, decoder=True):
        self.part = (169, 301, 192)
        self.prefix = 'mouth'
        super().__init__(self.part, self.prefix, encoder, decoder)
        
class ComponentEmbedding_Background(ComponentEmbedding_Master):
    def __init__(self, encoder=True, decoder=True):
        self.part = (0, 0, 512)
        self.prefix = 'background'
        super().__init__(self.part, self.prefix, encoder, decoder)
        
class FeatureMapping(nn.Module):
    
    def __init__(self, dimension, decoder=True, latent_dimension=512, spatial_channel=32, verify=False):
        super().__init__()
        self.decoder = None
        self.output = None
        
        self.dimension = dimension
        self.latent_dimension = latent_dimension
        self.spatial_channel = spatial_channel
        
        if decoder:
            self.init_encoder()
            self.init_decoder()
            self.delete_encoder()
            if verify: self.verify_decoder()
        
    def init_encoder(self):
        self.encoder_channels = [1, 32, 64, 128, 256, 512]
        self.encoder = Encoder(self.encoder_channels, self.dimension, self.latent_dimension)
        
    def init_decoder(self):
        self.decoder_channels = [512, 256, 256, 128, 64, 64]
        self.decoder = Decoder(self.decoder_channels, self.encoder.output_dimension, self.latent_dimension)
        self.output = nn.Sequential(
            Resnet_Block(self.decoder_channels[-1]),
            ReflectionPad2d(self.decoder.output_dimension, self.dimension + 4),
            nn.Conv2d(self.decoder_channels[-1], self.spatial_channel, kernel_size=5),
        )
    
    def delete_encoder(self):
        del self.encoder_channels
        del self.encoder
    
    def delete_decoder(self):
        del self.decoder_channels
        del self.decoder
        del self.output
        
    def verify_decoder(self):
        x = torch.rand(1, self.latent_dimension)
        x = self.Decode(x)
        assert x.shape == (1, self.spatial_channel, self.dimension, self.dimension), f"{x.shape} == {(1, self.spatial_channel, self.dimension, self.dimension)}"
    
    def forward(self, x):
        x = self.Decode(x)
        return x
    
    def Decode(self, latent):
        assert latent.shape[1:] == (self.latent_dimension,)
        spatial_map = self.decoder(latent)
        spatial_map = self.output(spatial_map)
        return spatial_map
    
    path_dict = {
        'decoder' : 'decoder.pth',
        'output' : 'output.pth'
    }
    
    def get_path(self, path, key):
        return os.path.join(path, self.path_dict[key])
    
    def save_decoder(self, path):
        torch.save(self.decoder.state_dict(), path)
        print(f'Saved Feature Mapping : decoder to {path}')
    
    def save_output(self, path):
        torch.save(self.output.state_dict(), path)
        print(f'Saved Feature Mapping : output to {path}')
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        if self.decoder: self.save_decoder(self.get_path(path, 'decoder'))
        if self.output: self.save_output(self.get_path(path, 'output'))
 
    def load_decoder(self, path):
        self.decoder.load_state_dict(torch.load(path))
        print(f'Loaded Feature Mapping : decoder from {path}')
    
    def load_output(self, path):
        self.output.load_state_dict(torch.load(path))
        print(f'Loaded Feature Mapping : output from {path}')
    
    def load(self, path):
        if self.decoder: self.load_decoder(self.get_path(path, 'decoder'))
        if self.output: self.load_output(self.get_path(path, 'output'))

class FeatureMapping_Master(FeatureMapping):
    def __init__(self, CE, decoder=True):
        self.master_dimension = CE.master_dimension
        self.part = CE.part
        self.prefix = CE.prefix
        self.crop_dimension = CE.crop_dimension
        super().__init__(CE.dimension, decoder)
    
    def merge(self, spatial_map, patch):
        assert spatial_map.shape[1:] == (self.spatial_channel, self.master_dimension, self.master_dimension), f'{spatial_map.shape[1:]} != {(self.spatial_channel, self.master_dimension, self.master_dimension)}'
        assert patch.shape[1:] == (self.spatial_channel, self.dimension, self.dimension), f'{patch.shape[1:]} != {(self.spatial_channel, self.dimension, self.dimension)}'
        spatial_map = spatial_map.clone()
        spatial_map[:, :, self.crop_dimension[1]:self.crop_dimension[3], self.crop_dimension[0]:self.crop_dimension[2]] = patch.clone()
        return spatial_map
    
    def save(self, path):
        super().save(os.path.join(path, self.prefix))
    
    def load(self, path):
        super().load(os.path.join(path, self.prefix))

class FeatureMapping_LeftEye(FeatureMapping_Master):
    def __init__(self, decoder=True):
        super().__init__(ComponentEmbedding_LeftEye(encoder=False, decoder=False), decoder)

class FeatureMapping_RightEye(FeatureMapping_Master):
    def __init__(self, decoder=True):
        super().__init__(ComponentEmbedding_RightEye(encoder=False, decoder=False), decoder)

class FeatureMapping_Nose(FeatureMapping_Master):
    def __init__(self, decoder=True):
        super().__init__(ComponentEmbedding_Nose(encoder=False, decoder=False), decoder)

class FeatureMapping_Mouth(FeatureMapping_Master):
    def __init__(self, decoder=True):
        super().__init__(ComponentEmbedding_Mouth(encoder=False, decoder=False), decoder)

class FeatureMapping_Background(FeatureMapping_Master):
    def __init__(self, decoder=True):
        super().__init__(ComponentEmbedding_Background(encoder=False, decoder=False), decoder)

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
        x = torch.tanh(x)
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
            self.output_dimension = self.output_dimension // 2
        
        self.dis_channels = [self.spatial_channel + 3, 64, 128, 256, 512]
        self.dis = nn.ModuleList()
        for i in range(1, len(self.dis_channels)):
            self.dis.append(Conv2D_Block(self.dis_channels[i-1], self.dis_channels[i]))
            self.output_dimension = self.output_dimension // 2
    
    def forward(self, x):
        for i in range(len(self.pool)):
            x = self.pool[i](x)
        for i in range(len(self.dis)):
            x = self.dis[i](x)
        x = torch.sigmoid(x)
        return x

class ImageSynthesis(nn.Module):
    
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
        return self.Generate(x)
    
    def Generate(self, spatial_map):
        assert spatial_map.shape[1:] == (self.spatial_channel, self.dimension, self.dimension)
        photo = self.G(spatial_map)
        return photo
    
    def Discriminate(self, spatial_map, photo):
        assert spatial_map.shape[0] == photo.shape[0]
        assert spatial_map.shape[1:] == (self.spatial_channel, self.dimension, self.dimension)
        assert photo.shape[1:] == (3, self.dimension, self.dimension)
        
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
        
    def load_G(self, path):
        self.G.load_state_dict(torch.load(path))
        print(f'Loaded Image Synthesis : G from {path}')
    
    def load_D1(self, path):
        self.D1.load_state_dict(torch.load(path))
        print(f'Loaded Image Synthesis : D1 from {path}')
    
    def load_D2(self, path):
        self.D2.load_state_dict(torch.load(path))
        print(f'Loaded Image Synthesis : D2 from {path}')
    
    def load_D3(self, path):
        self.D3.load_state_dict(torch.load(path))
        print(f'Loaded Image Synthesis : D3 from {path}')
    
    def load(self, path):
        if self.G: self.load_G(self.get_path(path, 'G'))
        if self.D1: self.load_D1(self.get_path(path, 'D1'))
        if self.D2: self.load_D2(self.get_path(path, 'D2'))
        if self.D3: self.load_D3(self.get_path(path, 'D3'))
        
class DeepFaceDrawing(nn.Module):
    
    def __init__(self, CE=True, FM=True, IS=True, manifold=False, CE_encoder=True, CE_decoder=False, FM_decoder=True, IS_generator=True, IS_discriminator=False):
        super().__init__()
        self.CE = None
        self.FM = None
        self.IS = None
        self.MN = None
        
        self.keys = ['left_eye', 'right_eye', 'nose', 'mouth', 'background']
            
        if CE:
            self.CE = nn.ModuleDict({
                'left_eye' : ComponentEmbedding_LeftEye(encoder=CE_encoder, decoder=CE_decoder),
                'right_eye' : ComponentEmbedding_RightEye(encoder=CE_encoder, decoder=CE_decoder),
                'nose' : ComponentEmbedding_Nose(encoder=CE_encoder, decoder=CE_decoder),
                'mouth' : ComponentEmbedding_Mouth(encoder=CE_encoder, decoder=CE_decoder),
                'background' : ComponentEmbedding_Background(encoder=CE_encoder, decoder=CE_decoder)
            })
        
        if FM:
            self.FM = nn.ModuleDict({
                'left_eye' : FeatureMapping_LeftEye(decoder=FM_decoder),
                'right_eye' : FeatureMapping_RightEye(decoder=FM_decoder),
                'nose' : FeatureMapping_Nose(decoder=FM_decoder),
                'mouth' : FeatureMapping_Mouth(decoder=FM_decoder),
                'background' : FeatureMapping_Background(decoder=FM_decoder)
            })
        
        if IS:
            self.IS = ImageSynthesis(generator=IS_generator, discriminator=IS_discriminator)
        
        if manifold:
            raise NotImplementedError
        
    def forward(self, x):
        x = self.CE_Encode(x)
        if self.MN: x = self.Manifold(x)
        x = self.FM_Decode(x)
        x = self.IS_Synthesis(x)
        return x
    
    def CE_Encode(self, sketch):
        patch_LeftEye = self.CE['left_eye'].crop(sketch)
        latent_LeftEye = self.CE['left_eye'].Encode(patch_LeftEye)
        
        patch_RightEye = self.CE['right_eye'].crop(sketch)
        latent_RightEye = self.CE['right_eye'].Encode(patch_RightEye)
        
        patch_Nose = self.CE['nose'].crop(sketch)
        latent_Nose = self.CE['nose'].Encode(patch_Nose)
        
        patch_Mouth = self.CE['mouth'].crop(sketch)
        latent_Mouth = self.CE['mouth'].Encode(patch_Mouth)
        
        patch_Background = self.CE['background'].crop(sketch)
        latent_Background = self.CE['background'].Encode(patch_Background)
        
        return {
            'left_eye' : latent_LeftEye,
            'right_eye' : latent_RightEye,
            'nose' : latent_Nose,
            'mouth' : latent_Nose,
            'background' : latent_Background
        }
    
    def CE_Decode(self, latent):
        patch_LeftEye = self.CE['left_eye'].Decode(latent['left_eye'])
        patch_RightEye = self.CE['right_eye'].Decode(latent['right_eye'])
        patch_Nose = self.CE['nose'].Decode(latent['nose'])
        patch_Mouth = self.CE['mouth'].Decode(latent['mouth'])
        patch_Background = self.CE['background'].Decode(latent['background'])
        
        return {
            'left_eye' : patch_LeftEye,
            'right_eye' : patch_RightEye,
            'nose' : patch_Nose,
            'mouth' : patch_Mouth,
            'background' : patch_Background
        }
        
    def FM_Decode(self, latent):
        patch_LeftEye = self.FM['left_eye'].Decode(latent['left_eye'])
        patch_RightEye = self.FM['right_eye'].Decode(latent['right_eye'])
        patch_Nose = self.FM['nose'].Decode(latent['nose'])
        patch_Mouth = self.FM['mouth'].Decode(latent['mouth'])
        patch_Background = self.FM['background'].Decode(latent['background'])
        
        spatial_map = patch_Background.clone()
        spatial_map = self.FM['left_eye'].merge(spatial_map, patch_LeftEye)
        spatial_map = self.FM['right_eye'].merge(spatial_map, patch_RightEye)
        spatial_map = self.FM['nose'].merge(spatial_map, patch_Nose)
        spatial_map = self.FM['mouth'].merge(spatial_map, patch_Mouth)
        
        return spatial_map
    
    def IS_Synthesis(self, spatial_map):
        return self.IS.Generate(spatial_map)
        
    def IS_Discriminate(self, spatial_map, photo):
        return self.IS.Discriminate(spatial_map, photo)
    
    def Manifold(self, latent):
        raise NotImplementedError
    
    path_dict = {
        'CE' : 'CE',
        'FM' : 'FM',
        'IS' : 'IS'
    }
    
    def get_path(self, path, key):
        return os.path.join(path, self.path_dict[key])
    
    def save_CE(self, path):
        for key, CEs in self.CE.items():
            CEs.save(path)
    
    def save_FM(self, path):
        for key, FMs in self.FM.items():
            FMs.save(path)
    
    def save_IS(self, path):
        self.IS.save(path)
        
    def save_MN(self, path):
        raise NotImplementedError
    
    def save(self, path):
        if self.CE: self.save_CE(self.get_path(path, 'CE'))
        if self.FM: self.save_FM(self.get_path(path, 'FM'))
        if self.IS: self.save_IS(self.get_path(path, 'IS'))
        if self.MN: raise NotImplementedError
    
    def load_CE(self, path):
        for key, CEs in self.CE.items():
            CEs.load(path)
    
    def load_FM(self, path):
        for key, FMs in self.FM.items():
            FMs.load(path)
    
    def load_IS(self, path):
        self.IS.load(path)
        
    def load_MN(self, path):
        raise NotImplementedError
    
    def load(self, path):
        if self.CE: self.load_CE(self.get_path(path, 'CE'))
        if self.FM: self.load_FM(self.get_path(path, 'FM'))
        if self.IS: self.load_IS(self.get_path(path, 'IS'))
        if self.MN: raise NotImplementedError
        
def main():
    model = DeepFaceDrawing(
        CE=True, CE_encoder=True, CE_decoder=True,
        FM=True, FM_decoder=True,
        IS=True, IS_generator=True, IS_discriminator=True,
        manifold=False
    )
    
if __name__ == '__main__':
    main()