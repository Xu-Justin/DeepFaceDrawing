import numpy as np
from PIL import Image

def vstack(images):
    max_width = 0
    for image in images:
        width, height = image.size
        max_width = max(max_width, width)
    else:
        for i, image in enumerate(images):
            image = image.convert('RGB')
            width, height = image.size
            new_image = Image.new(image.mode, (max_width, height), (255, 255, 255))
            new_image.paste(image, (0, 0))
            images[i] = new_image
        else:
            return Image.fromarray(np.vstack([np.asarray(image) for image in images]))

def hstack(images):
    max_height = 0
    for image in images:
        width, height = image.size
        max_height = max(max_height, height)
    else:
        for i, image in enumerate(images):
            image = image.convert('RGB')
            width, height = image.size
            new_image = Image.new(image.mode, (width, max_height), (255, 255, 255))
            new_image.paste(image, (0, 0))
            images[i] = new_image
        else:
            return Image.fromarray(np.hstack([np.asarray(image) for image in images]))