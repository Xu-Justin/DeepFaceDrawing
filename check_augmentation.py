from dataset import augmentation
from PIL import Image
from utils import hstack, imshow
import matplotlib.pyplot as plt

sketch = Image.open('resources/sample_sketch/10413.jpg')
photo = Image.open('resources/sample_photo/10413.jpg')

for _ in range(10):
    aug_sketch, aug_photo = augmentation(sketch, photo)
    result = hstack(aug_sketch, aug_photo)
    imshow(result)
