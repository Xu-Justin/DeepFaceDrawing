import numpy as np
import PIL
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def tensor2numpy(tensor_array):
    return tensor_array.permute(1, 2, 0).cpu().detach().numpy()

def rescale(numpy_array):
    scaler = MinMaxScaler()
    return scaler.fit_transform(numpy_array)

def numpy2PIL(numpy_array):
    return PIL.Image.fromarray(np.uint8(numpy_array * 255))

def resize(Image, dim):
    return Image.resize(dim)

def hstack(Image_1, Image_2):
    return np.hstack((np.asarray(Image_1), np.asarray(Image_2)))

def imshow(Image, cmap='viridis'):
    plt.imshow(Image, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def stack_patches(patches):
    placeholder = dict()
    for key, images in patches.items():
        for i, image in enumerate(images):
            image = resize(numpy2PIL(rescale(tensor2numpy(image).squeeze())).convert('RGB'), (512, 512))
            if i not in placeholder.keys(): placeholder[i] = image
            else: placeholder[i] = hstack(placeholder[i], image)
    
    images = []
    for key, image in placeholder.items():
        images.append(image)
    
    return images

def display_patches(patches):
    images = stack_patches(patches)
    for image in images:
        imshow(image)

def stack_preview(sketches, fakes):
    placeholder = dict()
    for i, sketch in enumerate(sketches):
        placeholder[i] = dict()
        placeholder[i]['sketch'] = resize(numpy2PIL(rescale(tensor2numpy(sketch).squeeze())).convert('RGB'), (512, 512))

    for i, fake in enumerate(fakes):
        placeholder[i]['fake'] = resize(numpy2PIL(tensor2numpy(fake)), (512, 512))
    
    images = []
    for i, image in placeholder.items():
        images.append(image['sketch'])
        images[i] = hstack(images[i], image['fake'])

    return images

def display_preview(sketches, fakes):
    images = stack_preview(sketches, fakes)
    for image in images:
        imshow(image)
