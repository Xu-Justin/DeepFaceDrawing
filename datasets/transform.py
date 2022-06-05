from torchvision import transforms

transform_sketch = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(512),
    transforms.ToTensor()
])

transform_photo = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor()
])