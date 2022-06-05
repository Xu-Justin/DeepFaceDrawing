import os
import torch
import torch.nn as nn
from . import autoencoder, ComponentEmbedding

class Base(nn.Module):
    def __init__(self, dimension, latent_dimension, spatial_channel, decoder=True):
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

    def init_encoder(self):
        self.encoder_channels = [1, 32, 64, 128, 256, 512]
        self.encoder = autoencoder.Encoder(self.encoder_channels, self.dimension, self.latent_dimension)
        
    def init_decoder(self):
        self.decoder_channels = [512, 256, 256, 128, 64, 64, self.spatial_channel]
        self.decoder = autoencoder.Decoder(self.decoder_channels, self.encoder.conv_dimension, self.dimension, self.latent_dimension)

    def delete_encoder(self):
        self.encoder_channels = None
        self.encoder = None

    def delete_decoder(self):
        self.decoder_channels = None
        self.decoder = None

    def forward(self, x):
        x = self.decode(x)
        return x
        
    def decode(self, x):
        assert x.shape[1:] == (self.latent_dimension,), f'[Feature Mapping : decode] Expected input shape {(-1, self.latent_dimension)}, but received {x.shape}.'
        x = self.decoder(x)
        assert x.shape[1:] == (self.decoder_channels[-1], self.dimension, self.dimension), f'[Feature Mapping : decode] Expected output shape {(-1, self.decoder_channels[-1], self.dimension, self.dimension)}, but yield {x.shape}.'
        return x
    
    path_dict = {
        'decoder' : 'decoder.pth'
    }
    
    def get_path(self, path, key):
        return os.path.join(path, self.path_dict[key])
    
    def save_decoder(self, path):
        torch.save(self.decoder.state_dict(), path)
        print(f'Saved Feature Mapping : decoder to {path}')
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        if self.decoder: self.save_decoder(self.get_path(path, 'decoder'))
    
    def load_decoder(self, path, map_location=torch.device('cpu')):
        self.decoder.load_state_dict(torch.load(path, map_location=map_location))
        print(f'Loaded Feature Mapping : decoder from {path}')
    
    def load(self, path, map_location=torch.device('cpu')):
        if self.decoder: self.load_decoder(self.get_path(path, 'decoder'), map_location=map_location)

class Master(Base):
    def __init__(self, CE, decoder=True):
        self.master_dimension = CE.master_dimension
        self.latent_dimension = CE.latent_dimension
        self.spatial_channel = 32
        self.part = CE.part
        self.prefix = CE.prefix
        self.crop_dimension = CE.crop_dimension
        super().__init__(self.part[2], self.latent_dimension, self.spatial_channel, decoder)
    
    def merge(self, spatial_map, patch):
        assert spatial_map.shape[1:] == (self.spatial_channel, self.master_dimension, self.master_dimension), f'[Feature Mapping : merge] Expected input shape of spatial_map {(-1, self.spatial_channel, self.master_dimension, self.master_dimension)}, but received {spatial_map.shape}.'
        assert patch.shape[1:] == (self.spatial_channel, self.dimension, self.dimension), f'[Feature Mapping : merge] Expected input shape of patch {(-1, self.spatial_channel, self.dimension, self.dimension)}, but received {patch.shape}.'
        spatial_map[:, :, self.crop_dimension[1]:self.crop_dimension[3], self.crop_dimension[0]:self.crop_dimension[2]] *= patch
        return spatial_map
    
    def save(self, path):
        super().save(os.path.join(path, self.prefix))
    
    def load(self, path, map_location=torch.device('cpu')):
        super().load(os.path.join(path, self.prefix), map_location=map_location)

class LeftEye(Master):
    def __init__(self, decoder=True):
        super().__init__(ComponentEmbedding.LeftEye(encoder=False, decoder=False), decoder)

class RightEye(Master):
    def __init__(self, decoder=True):
        super().__init__(ComponentEmbedding.RightEye(encoder=False, decoder=False), decoder)

class Nose(Master):
    def __init__(self, decoder=True):
        super().__init__(ComponentEmbedding.Nose(encoder=False, decoder=False), decoder)

class Mouth(Master):
    def __init__(self, decoder=True):
        super().__init__(ComponentEmbedding.Mouth(encoder=False, decoder=False), decoder)

class Background(Master):
    def __init__(self, decoder=True):
        super().__init__(ComponentEmbedding.Background(encoder=False, decoder=False), decoder)

class Module(nn.Module):
    def __init__(self, decoder=True):
        super().__init__()
        self.components = nn.ModuleDict({
                'left_eye' : LeftEye(decoder=decoder),
                'right_eye' : RightEye(decoder=decoder),
                'nose' : Nose(decoder=decoder),
                'mouth' : Mouth(decoder=decoder),
                'background' : Background(decoder=decoder)
            })

    def forward(self, x):
        x = self.decode(x)
        x = self.merge(x)
        return x
    
    def decode(self, latent):
        return {
            'left_eye' : self.components['left_eye'].decode(latent['left_eye']),
            'right_eye' : self.components['right_eye'].decode(latent['right_eye']),
            'nose' : self.components['nose'].decode(latent['nose']),
            'mouth' : self.components['mouth'].decode(latent['mouth']),
            'background' : self.components['background'].decode(latent['background']),
        }

    def merge(self, patches):
        spatial_map = patches['background']
        spatial_map = self.components['left_eye'].merge(spatial_map, patches['left_eye'])
        spatial_map = self.components['right_eye'].merge(spatial_map, patches['right_eye'])
        spatial_map = self.components['nose'].merge(spatial_map, patches['nose'])
        spatial_map = self.components['mouth'].merge(spatial_map, patches['mouth'])
        return spatial_map

    def save(self, path):
        for key, component in self.components.items():
            component.save(path)

    def load(self, path, map_location=torch.device('cpu')):
        for key, component in self.components.items():
            component.load(path, map_location=map_location)