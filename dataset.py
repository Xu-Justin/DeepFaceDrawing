from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
from PIL import Image

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

def augmentation(sketch, photo=None):
    raise NotImplementedError

def tanh(x):
    return (x * 2) - 1

def itanh(x):
    return (x + 1) / 2

class dataset(Dataset):
    
    def __init__(self, path, transform_sketch=transform_sketch, transform_photo=transform_photo, load_photo=True, augmentation=False):
        self.load_photo = load_photo
        self.augmentation = augmentation
        
        self.transfrom_sketch = transform_sketch
        self.transfrom_photo = transform_photo
        
        self.folder_sketch = os.path.join(path, 'sketch')
        self.folder_photo = os.path.join(path, 'photo')
        
        self.path_sketches = []
        self.path_photos = []
        
        for file_name in os.listdir(self.folder_sketch):
            self.path_sketches.append(os.path.join(self.folder_sketch, file_name))
            
            if self.load_photo:
                assert os.path.exists(os.path.join(self.folder_photo, file_name)), f'{os.path.join(self.folder_photo, file_name)} doesn\'t exists'
                self.path_photos.append(os.path.join(self.folder_photo, file_name))
    
    def __len__(self):
        return len(self.path_sketches)
    
    def __getitem__(self, idx):
        sketch = Image.open(self.path_sketches[idx])
        sketch = self.transfrom_sketch(sketch)
        
        if self.load_photo:
            photo = Image.open(self.path_photos[idx])
            photo = self.transfrom_photo(photo)
        
        if self.augmentation:
            if self.load_photo: sketch, photo = augmentation(sketch, photo)
            else: sketch = augmentation(sketch)
            
        if self.load_photo: return sketch, photo
        else: return sketch

def dataloader(path, batch_size, load_photo=True, shuffle=True, num_workers=4):
    custom_dataset = dataset(path, load_photo=load_photo)
    custom_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return custom_dataloader

def load_one_sketch(path):
    sketch = Image.open(path)
    sketch = transform_sketch(sketch)
    return sketch.unsqueeze(0)

def load_one_photo(path):
    photo = Image.open(path)
    photo = transform_photo(photo)
    return photo.unsqueeze(0)
