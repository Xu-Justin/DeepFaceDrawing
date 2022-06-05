import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from . import augmentation as Aug
from . import transform as T
from . import sketch_simplification

class dataset(Dataset):
    
    def __init__(self, path, transform_sketch=T.transform_sketch, transform_photo=T.transform_photo, load_photo=True, augmentation=False):
        self.load_photo = load_photo
        self.augmentation = augmentation
        
        self.transfrom_sketch = transform_sketch
        self.transfrom_photo = transform_photo
        
        self.folder_sketch = os.path.join(path, 'sketches')
        self.folder_photo = os.path.join(path, 'photos')
        
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
        if self.load_photo:
            return load_one_sketch_photo(self.path_sketches[idx], self.path_photos[idx], augmentation=self.augmentation)
        else:
            return load_one_sketch(self.path_sketches[idx], augmentation=self.augmentation)

def dataloader(path, batch_size, load_photo=True, shuffle=True, num_workers=4, augmentation=False):
    custom_dataset = dataset(path, load_photo=load_photo, augmentation=augmentation)
    custom_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return custom_dataloader

simplificator = None

def load_one_sketch(path, augmentation=False, simplify=False, device='cpu'):
    sketch = Image.open(path)
    if simplify:
        global simplificator
        if not simplificator: simplificator = sketch_simplification.sketch_simplification(device=device)
        sketch = simplificator.simplify(sketch)
    if augmentation:
        sketch = Aug.random_erase(sketch)
        sketch = Aug.random_affine([sketch])[0]
    sketch = T.transform_sketch(sketch)
    return sketch

def load_one_photo(path, augmentation=False, simplify=False, device='cpu'):
    photo = Image.open(path)
    if simplify:
        global simplificator
        if not simplificator: simplificator = sketch_simplification.sketch_simplification(device=device)
        photo = simplificator.simplify(photo)
    if augmentation:
        photo = Aug.random_affine([photo])[0]
    photo = T.transform_photo(photo)
    return photo

def load_one_sketch_photo(path_sketch, path_photo, augmentation=False, simplify=False, device='cpu'):
    sketch = Image.open(path_sketch)
    photo = Image.open(path_photo)
    if simplify:
        global simplificator
        if not simplificator: simplificator = sketch_simplification.sketch_simplification(device=device)
        sketch = simplificator.simplify(sketch)
        photo = simplificator.simplify(photo)
    if augmentation:
        sketch = Aug.random_erase(sketch)
        sketch, photo = Aug.random_affine([sketch, photo])
    sketch = T.transform_sketch(sketch)
    photo  = T.transform_photo(photo)
    return sketch, photo