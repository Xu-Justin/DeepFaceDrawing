__all__ = ['convert', 'stack', 'heatmap']
from . import *

def patches2PIL(patches):
    images = []
    for i in range(len(patches['left_eye'])):
        image = [patches['left_eye'][i], patches['right_eye'][i], patches['nose'][i], patches['mouth'][i], patches['background'][i]]
        image = [convert.tensor2PIL(img).resize((512, 512)) for img in image]
        image = stack.hstack(image)
        images.append(image)
    return images