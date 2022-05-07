import os
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms

from model import ComponentEmbedding_LeftEye
from model import ComponentEmbedding_RightEye
from model import ComponentEmbedding_Nose
from model import ComponentEmbedding_Mouth
from model import ComponentEmbedding_Background

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Deep Face Drawing: Train Stage 1')
    parser.add_argument('--dataset', type=str, required=True, help='Path to training dataset.')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--resume', type=str, default=None, help='Path to load weights.')
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
    
    CE_LeftEye = ComponentEmbedding_LeftEye()
    CE_RightEye = ComponentEmbedding_RightEye()
    CE_Nose = ComponentEmbedding_Nose()
    CE_Mouth = ComponentEmbedding_Mouth()
    CE_Background = ComponentEmbedding_Background()
    
    CE = [CE_LeftEye, CE_RightEye, CE_Nose, CE_Mouth, CE_Background]
    
    if args.resume:
        for CEs in CE: CEs.load(args.resume)
            
    for CEs in CE: CEs.to(device)
    
    transform = CE_Background.transform
    augmentation_transform = None
    dataset = Stage1Dataset(args.dataset, transform, augmentation_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)
    
    optimizers = [torch.optim.Adam(CEs.parameters(), lr=0.0002, betas=(0.5, 0.999)) for CEs in CE]
    criterion = torch.nn.MSELoss()
        
    for epoch in range(args.epochs):
        for CEs in CE: CEs.train()
        
        running_loss = [0 for _ in range(len(CE))]
        
        for sketch in tqdm(dataloader, desc=f'Epoch - {epoch+1} / {args.epochs}'):
            for i in range(len(CE)):
                cropped_sketch = CE[i].crop(sketch).to(device)
                optimizers[i].zero_grad()
                
                prediction = CE[i](cropped_sketch)
                loss = criterion(prediction, cropped_sketch)
                loss.backward()
                optimizers[i].step()
                
                running_loss[i] += loss.item() * len(sketch) * len(dataloader.sampler)
        
        print()    
        print(f'Epoch - {epoch+1} / {args.epochs}')
        print(f'Loss    : {running_loss:.6f}')
        print()
        
        for CEs in CE: CEs.save(args.output)
    
if __name__ == '__main__':
    args = get_args_parser()
    print(args)
    main(args)