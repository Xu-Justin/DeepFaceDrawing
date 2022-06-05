import os
import torch
import torch.nn as nn

__all__ = ['ComponentEmbedding', 'FeatureMapping', 'ImageSynthesis', 'block', 'autoencoder']
from . import *

class DeepFaceDrawing(nn.Module):
    
    def __init__(self, CE=True, FM=True, IS=True, manifold=False, CE_encoder=True, CE_decoder=False, FM_decoder=True, IS_generator=True, IS_discriminator=False):
        super().__init__()
        self.CE = None
        self.FM = None
        self.IS = None
        self.MN = None
        
        self.components = ['left_eye', 'right_eye', 'nose', 'mouth', 'background']
            
        if CE: self.CE = ComponentEmbedding.Module(encoder=CE_encoder, decoder=CE_decoder)
        if FM: self.FM = FeatureMapping.Module(decoder=FM_decoder)        
        if IS: self.IS = ImageSynthesis.Module(generator=IS_generator, discriminator=IS_discriminator)
        if manifold: raise NotImplementedError
        
    def forward(self, x):
        x = self.CE.crop(x)
        x = self.CE.encode(x)
        if self.MN: raise NotImplementedError
        x = self.FM.decode(x)
        x = self.FM.merge(x)
        x = self.IS.generate(x)
        return x
    
    path_dict = {
        'CE' : 'CE',
        'FM' : 'FM',
        'IS' : 'IS'
    }
    
    def get_path(self, path, key):
        return os.path.join(path, self.path_dict[key])
    
    def save(self, path):
        if self.CE: self.CE.save(self.get_path(path, 'CE'))
        if self.FM: self.FM.save(self.get_path(path, 'FM'))
        if self.IS: self.IS.save(self.get_path(path, 'IS'))
        if self.MN: raise NotImplementedError
    
    def load(self, path, map_location=torch.device('cpu')):
        if self.CE: self.CE.load(self.get_path(path, 'CE'), map_location=map_location)
        if self.FM: self.FM.load(self.get_path(path, 'FM'), map_location=map_location)
        if self.IS: self.IS.load(self.get_path(path, 'IS'), map_location=map_location)
        if self.MN: raise NotImplementedError
