import os
import torch
import torch.nn as nn
from . import autoencoder

class Base(nn.Module):
    def __init__(self, dimension, encoder=True, decoder=True, latent_dimension=512):
        super().__init__()
        
        self.encoder = None
        self.decoder = None
        self.dimension = dimension
        self.latent_dimension = latent_dimension
        
        if encoder:
            self.init_encoder()
        
        if decoder:
            self.init_decoder()
    
    def init_encoder(self):
        self.encoder_channels = [1, 32, 64, 128, 256, 512]
        self.encoder = autoencoder.Encoder(self.encoder_channels, self.dimension, self.latent_dimension)

    def init_decoder(self):
        self.decoder_channels = [512, 256, 128, 64, 32, 32, 1]
        self.decoder = autoencoder.Decoder(self.decoder_channels, self.encoder.conv_dimension, self.dimension, self.latent_dimension)
    
    def delete_encoder(self):
        self.encoder_channels = None
        self.encoder = None

    def delete_decoder(self):
        self.decoder_channels = None
        self.decoder = None

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
    def encode(self, x):
        assert x.shape[1:] == (self.encoder_channels[0], self.dimension, self.dimension), f'[Component Embedding : encode] Expected input shape {(-1, self.encoder_channels[0], self.dimension, self.dimension)}, but received {x.shape}.'
        x = self.encoder(x)
        assert x.shape[1:] == (self.latent_dimension,), f'[Component Embedding : encode] Expected output shape {(-1, self.latent_dimension)}, but yield {x.shape}.'
        return x
    
    def decode(self, x):
        assert x.shape[1:] == (self.latent_dimension,), f'[Component Embedding : decode] Expected input shape {(-1, self.latent_dimension)}, but received {x.shape}.'
        x = self.decoder(x)
        assert x.shape[1:] == (self.decoder_channels[-1], self.dimension, self.dimension), f'[Component Embedding : decode] Expected output shape {(-1, self.decoder_channels[-1], self.dimension, self.dimension)}, but yield {x.shape}.'
        return x

    path_dict = {
        'encoder' : 'encoder.pth',
        'decoder' : 'decoder.pth'
    }
    
    def get_path(self, path, key):
        return os.path.join(path, self.path_dict[key])
    
    def save_encoder(self, path):
        torch.save(self.encoder.state_dict(), path)
        print(f'Saved Component Embedding : encoder to {path}')
    
    def save_decoder(self, path):
        torch.save(self.decoder.state_dict(), path)
        print(f'Saved Component Embedding : decoder to {path}')
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        if self.encoder: self.save_encoder(self.get_path(path, 'encoder'))
        if self.decoder: self.save_decoder(self.get_path(path, 'decoder'))
    
    def load_encoder(self, path, map_location=torch.device('cpu')):
        self.encoder.load_state_dict(torch.load(path, map_location=map_location))
        print(f'Loaded Component Embedding : encoder from {path}')
    
    def load_decoder(self, path, map_location=torch.device('cpu')):
        self.decoder.load_state_dict(torch.load(path, map_location=map_location))
        print(f'Loaded Component Embedding : decoder from {path}')
    
    def load(self, path, map_location=torch.device('cpu')):
        if self.encoder: self.load_encoder(self.get_path(path, 'encoder'), map_location=map_location)
        if self.decoder: self.load_decoder(self.get_path(path, 'decoder'), map_location=map_location)

class Master(Base):
    def __init__(self, part, prefix, encoder=True, decoder=True):
        self.master_dimension = 512
        self.latent_dimension = 512
        self.part = part
        self.prefix = prefix
        self.crop_dimension = (part[0], part[1], part[0] + part[2], part[1] + part[2]) # xmin, ymin, xmax, ymax
        super().__init__(self.part[2], encoder, decoder, self.latent_dimension)
    
    def crop(self, sketch):
        assert sketch.shape[1:] == (1, self.master_dimension, self.master_dimension), f'[Component Embedding : crop] Expected input shape {(-1, 1, self.master_dimension, self.master_dimension)}, but received {(sketch.shape)}.'
        return sketch[:, :, self.crop_dimension[1]:self.crop_dimension[3], self.crop_dimension[0]:self.crop_dimension[2]].clone()
    
    def save(self, path):
        super().save(os.path.join(path, self.prefix))
    
    def load(self, path, map_location=torch.device('cpu')):
        super().load(os.path.join(path, self.prefix), map_location=map_location)

class LeftEye(Master):
    def __init__(self, encoder=True, decoder=True):
        self.part = (108, 126, 128)
        self.prefix = 'left_eye'
        super().__init__(self.part, self.prefix, encoder, decoder)
    
class RightEye(Master):
    def __init__(self, encoder=True, decoder=True):
        self.part = (255, 126, 128)
        self.prefix = 'right_eye'
        super().__init__(self.part, self.prefix, encoder, decoder)

class Nose(Master):
    def __init__(self, encoder=True, decoder=True):
        self.part = (182, 232, 160)
        self.prefix = 'nose'
        super().__init__(self.part, self.prefix, encoder, decoder)
    
class Mouth(Master):
    def __init__(self, encoder=True, decoder=True):
        self.part = (169, 301, 192)
        self.prefix = 'mouth'
        super().__init__(self.part, self.prefix, encoder, decoder)
        
class Background(Master):
    def __init__(self, encoder=True, decoder=True):
        self.part = (0, 0, 512)
        self.prefix = 'background'
        super().__init__(self.part, self.prefix, encoder, decoder)

class Module(nn.Module):
    def __init__(self, encoder=True, decoder=True):
        super().__init__()
        self.CE = nn.ModuleDict({
            'left_eye' : LeftEye(encoder=encoder, decoder=decoder),
            'right_eye' : RightEye(encoder=encoder, decoder=decoder),
            'nose' : Nose(encoder=encoder, decoder=decoder),
            'mouth' : Mouth(encoder=encoder, decoder=decoder),
            'background' : Background(encoder=encoder, decoder=decoder)
        })

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, sketch):
        patch_LeftEye = self.CE['left_eye'].crop(sketch)
        latent_LeftEye = self.CE['left_eye'].encode(patch_LeftEye)
        
        patch_RightEye = self.CE['right_eye'].crop(sketch)
        latent_RightEye = self.CE['right_eye'].encode(patch_RightEye)
        
        patch_Nose = self.CE['nose'].crop(sketch)
        latent_Nose = self.CE['nose'].encode(patch_Nose)
        
        patch_Mouth = self.CE['mouth'].crop(sketch)
        latent_Mouth = self.CE['mouth'].encode(patch_Mouth)
        
        patch_Background = self.CE['background'].crop(sketch)
        latent_Background = self.CE['background'].encode(patch_Background)
        
        return {
            'left_eye' : latent_LeftEye,
            'right_eye' : latent_RightEye,
            'nose' : latent_Nose,
            'mouth' : latent_Mouth,
            'background' : latent_Background
        }

    def decode(self, latent):
        patch_LeftEye = self.CE['left_eye'].decode(latent['left_eye'])
        patch_RightEye = self.CE['right_eye'].decode(latent['right_eye'])
        patch_Nose = self.CE['nose'].decode(latent['nose'])
        patch_Mouth = self.CE['mouth'].decode(latent['mouth'])
        patch_Background = self.CE['background'].decode(latent['background'])
        
        return {
            'left_eye' : patch_LeftEye,
            'right_eye' : patch_RightEye,
            'nose' : patch_Nose,
            'mouth' : patch_Mouth,
            'background' : patch_Background
        }

    def save(self, path):
        for key, CEs in self.CE.items():
            CEs.save(path)

    def load(self, path, map_location=torch.device('cpu')):
        for key, CEs in self.CE.items():
            CEs.load(path, map_location=map_location)