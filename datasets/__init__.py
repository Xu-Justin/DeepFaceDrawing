__all__ = ['augmentation', 'dataloader', 'transform', 'sketch_simplification']
from . import *

def range_tanh(x):
    return (x * 2) - 1

def range_itanh(x):
    return (x + 1) / 2