__all__ = ['convert', 'scale', 'stack', 'heatmap']
from . import *

def patches2PIL(patches):
    images = []
    for i in range(len(patches['left_eye'])):
        image = [patches['left_eye'][i], patches['right_eye'][i], patches['nose'][i], patches['mouth'][i], patches['background'][i]]
        image = [convert.tensor2numpy(image.squeeze()) for image in image]
        image = [scale.rescale(image)*255 for image in image]
        image = [convert.numpy2PIL(image).resize((512, 512)) for image in image]
        image = stack.hstack(image)
        images.append(image)
    return images