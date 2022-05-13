import os
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms

from model import DeepFaceDrawing

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Deep Face Drawing: Train Stage 1')
    parser.add_argument('--dataset', type=str, required=True, help='Path to training dataset.')
    parser.add_argument('--dataset_validation', type=str, default=None, help='Path to validation dataset.')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--resume', type=str, default=None, help='Path to load model weights.')
    parser.add_argument('--output', type=str, required=True, help='Path to save weights.')
    args = parser.parse_args()
    return args

class Stage1Dataset(Dataset):
    
    def __init__(self, path, transform, augmentation_transform=None):
        self.transform = transform
        self.augmentation_transform = augmentation_transform
        
        folder_sketch = os.path.join(path, 'sketch')
        
        self.path_sketches = []
        for file_name_sketch in os.listdir(folder_sketch):
            self.path_sketches.append(os.path.join(folder_sketch, file_name_sketch))
    
    def __len__(self, ):
        return len(self.path_sketches)
    
    def __getitem__(self, idx):
        sketch = Image.open(self.path_sketches[idx])
        if self.augmentation_transform:
            sketch = self.augmentation_transform(sketch)
        sketch = self.transform(sketch)
        return sketch
        
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    
    model = DeepFaceDrawing(
        CE=True, CE_encoder=True, CE_decoder=True,
        FM=False, IS=False, manifold=False
    )
    
    if args.resume:
        model.load(args.resume)
        
    model.to(device)
        
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    augmentation_transform = None
    train_dataset = Stage1Dataset(args.dataset, transform, augmentation_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)
    
    if args.dataset_validation:
        validation_dataset = Stage1Dataset(args.dataset_validation, transform)
        validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = torch.nn.MSELoss()
        
    for epoch in range(args.epochs):
        
        running_loss = {
            'left_eye' : 0,
            'right_eye' : 0,
            'nose' : 0,
            'mouth' : 0,
            'background' : 0
        }
        
        model.train()
        for sketches in tqdm(train_dataloader, desc=f'Epoch - {epoch+1} / {args.epochs}'):
            optimizer.zero_grad()
            for key in model.keys:
                X = model.CE[key].crop(sketches).to(device)
                y = model.CE[key](X)
                loss = criterion(y, X)
                loss.backward()
                running_loss[key] += loss.item() * len(sketches) / len(train_dataloader.dataset)
            optimizer.step()
        
        if args.dataset_validation:
            validation_running_loss = {
                'left_eye' : 0,
                'right_eye' : 0,
                'nose' : 0,
                'mouth' : 0,
                'background' : 0
            }
            
            model.eval()
            with torch.no_grad():
                for sketches in tqdm(validation_dataloader, desc=f'Epoch - {epoch+1} / {args.epochs}'):
                    for key in model.keys:
                        X = model.CE[key].crop(sketches).to(device)
                        y = model.CE[key](X)
                        loss = criterion(y, X)
                        validation_running_loss[key] += loss.item() * len(sketches) / len(validation_dataloader.dataset)
        
        print()    
        print(f'Epoch - {epoch+1} / {args.epochs}')
        for key, loss in running_loss.items():
            print(f'Loss {key:10} : {loss:.6f}')
        if args.dataset_validation:
            for key, loss in validation_running_loss.items():
                print(f'Loss {key:10} : {loss:.6f}')
        print()
        
        model.save(args.output)
        
if __name__ == '__main__':
    args = get_args_parser()
    print(args)
    main(args)