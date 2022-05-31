from torchvision import transforms

transform_sketch = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomEqualize(p=1),
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0], std=[0.5])
])

transform_photo = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor()
])