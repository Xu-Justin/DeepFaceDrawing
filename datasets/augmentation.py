import random
from torchvision import transforms

def random_affine(images, p=1):
    param_affine = transforms.RandomAffine.get_params(
        degrees=(-5, 5),
        translate=(0.05, 0.05),
        scale_ranges=(0.95, 1.05),
        shears=(-10, 10),
        img_size=(512, 512)   
    )

    def transform(image):
        if random.random() > p: return image
        image = transforms.functional.affine(image, param_affine[0], param_affine[1], param_affine[2], param_affine[3])
        return image

    images = [transform(image) for image in images]
    return images

def random_erase(image, p=1):
    def transform(image):
        image = transforms.ToTensor()(image)
        image = transforms.RandomErasing(p=p, value=1)(image)
        image = transforms.ToPILImage()(image)
        return image

    image = transform(image)
    return image