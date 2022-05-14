import os
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
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
    parser.add_argument('--resume_CE', type=str, default=None, help='Path to load Component Embedding model weights. Required if --resume is not given. Skipped if --resume is given.')
    parser.add_argument('--output', type=str, default=None, help='Path to save weights.')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    return args

def validation_parser(args):
    if args.resume:
        if args.resume_CE: print('args.resume_CE will be skipped.')
    else:
        assert args.resume_CE, "Both args.resume and args.resume_CE can't be None." 

class Stage2Dataset(Dataset):
    
    def __init__(self, path, transform_sketch, transform_photo):
        self.transform_sketch = transform_sketch
        self.transform_photo = transform_photo
        
        folder_sketch = os.path.join(path, 'sketch')
        folder_photo = os.path.join(path, 'photo')
        
        self.path_sketches = []
        self.path_photos = []
        for file_name_sketch in os.listdir(folder_sketch):
            self.path_sketches.append(os.path.join(folder_sketch, file_name_sketch))
            assert os.path.exists(os.path.join(folder_photo, file_name_sketch)), f"{os.path.join(folder_photo, file_name_sketch)} doesn't exists"
            self.path_photos.append(os.path.join(folder_photo, file_name_sketch))
    
    def __len__(self, ):
        return len(self.path_sketches)
    
    def __getitem__(self, idx):
        sketch = Image.open(self.path_sketches[idx])
        photo = Image.open(self.path_photos[idx])
        sketch = self.transform_sketch(sketch)
        photo = self.transform_photo(photo)
        return sketch, photo
        
def main(args):
    device = torch.device(args.device)
    print(f'Device : {device}')
    
    model = DeepFaceDrawing(
        CE=True, CE_encoder=True, CE_decoder=False,
        FM=True, FM_decoder=True,
        IS=True, IS_generator=True, IS_discriminator=True,
        manifold=False
    )
    
    if args.resume:
        model.load(args.resume)
    else:
        model.load_CE(args.resume_CE)
    
    model.to(device)
        
    transform_sketch = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    transform_photo = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    train_dataset = Stage2Dataset(args.dataset, transform_sketch, transform_photo)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)
    
    if args.dataset_validation:
        validation_dataset = Stage2Dataset(args.dataset_validation, transform_sketch, transform_photo)
        validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)
    
    for key, CEs in model.CE.items():
        for param in CEs.parameters():
            param.requires_grad = False
    
    optimizer_generator = torch.optim.Adam( list(model.FM.parameters()) + list(model.IS.G.parameters()) , lr=0.0002, betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam( list(model.IS.D1.parameters()) + list(model.IS.D2.parameters()) + list(model.IS.D3.parameters()) , lr=0.0002, betas=(0.5, 0.999))
    
    criterion_generator = torch.nn.L1Loss()
    criterion_discriminator = torch.nn.BCELoss()
    
    label_real = model.IS.label_real
    label_fake = model.IS.label_fake
    
    for epoch in range(args.epochs):
        
        running_loss = {
            'loss_G' : 0,
            'loss_D' : 0
        }
        
        model.train()
        for sketches, photos in tqdm(train_dataloader, desc=f'Epoch - {epoch+1} / {args.epochs}'):
            
            sketches = sketches.to(device)
            photos = photos.to(device)
            
            latents = model.CE_Encode(sketches)
            spatial_map = model.FM_Decode(latents)
            fake_photos = model.IS_Synthesis(spatial_map)
            
            optimizer_generator.zero_grad()
            loss_G_L1 = criterion_generator(fake_photos, photos)
            patches = model.IS.Discriminate(spatial_map, fake_photos)
            loss_G_BCE = torch.tensor([criterion_discriminator(patch, torch.full(patch.shape, label_real, dtype=torch.float, requires_grad=True).to(device)) for patch in patches], dtype=torch.float, requires_grad=True).sum()
            loss_G = 100 * loss_G_L1 + loss_G_BCE
            loss_G.backward()
            optimizer_generator.step()
            
            optimizer_discriminator.zero_grad()
            patches = model.IS.Discriminate(spatial_map.detach(), fake_photos.detach())
            loss_D_fake = torch.tensor([criterion_discriminator(patch, torch.full(patch.shape, label_fake, dtype=torch.float, requires_grad=True).to(device)) for patch in patches], dtype=torch.float, requires_grad=True).sum()
            patches = model.IS.Discriminate(spatial_map.detach(), photos.detach())
            loss_D_real = torch.tensor([criterion_discriminator(patch, torch.full(patch.shape, label_real, dtype=torch.float, requires_grad=True).to(device)) for patch in patches], dtype=torch.float, requires_grad=True).sum()
            loss_D = loss_D_fake + loss_D_real
            loss_D.backward()
            optimizer_discriminator.step()
            
            running_loss['loss_G'] += loss_G.item() * len(sketches) / len(train_dataloader.dataset)
            running_loss['loss_D'] += loss_D.item() * len(sketches) / len(train_dataloader.dataset)
        
        if args.dataset_validation:
            validation_running_loss = {
                'val_loss_G' : 0,
                'val_loss_D' : 0
            }
            
            model.eval()
            with torch.no_grad():
                for sketches, photos in tqdm(validation_dataloader, desc=f'Validation Epoch - {epoch+1} / {args.epochs}'):
            
                    sketches = sketches.to(device)
                    photos = photos.to(device)
                    
                    latents = model.CE_Encode(sketches)
                    spatial_map = model.FM_Decode(latents)
                    fake_photos = model.IS_Synthesis(spatial_map)
                    
                    loss_G_L1 = criterion_generator(fake_photos, photos)
                    patches = model.IS.Discriminate(spatial_map, fake_photos)
                    loss_G_BCE = torch.tensor([criterion_discriminator(patch, torch.full(patch.shape, label_real, dtype=torch.float, requires_grad=True).to(device)) for patch in patches], dtype=torch.float, requires_grad=True).sum()
                    loss_G = 100 * loss_G_L1 + loss_G_BCE
                    
                    patches = model.IS.Discriminate(spatial_map.detach(), fake_photos.detach())
                    loss_D_fake = torch.tensor([criterion_discriminator(patch, torch.full(patch.shape, label_fake, dtype=torch.float, requires_grad=True).to(device)) for patch in patches], dtype=torch.float, requires_grad=True).sum()
                    patches = model.IS.Discriminate(spatial_map.detach(), photos.detach())
                    loss_D_real = torch.tensor([criterion_discriminator(patch, torch.full(patch.shape, label_real, dtype=torch.float, requires_grad=True).to(device)) for patch in patches], dtype=torch.float, requires_grad=True).sum()
                    loss_D = loss_D_fake + loss_D_real
                    
                    validation_running_loss['val_loss_G'] += loss_G.item() * len(sketches) / len(validation_dataloader.dataset)
                    validation_running_loss['val_loss_D'] += loss_D.item() * len(sketches) / len(validation_dataloader.dataset)
            
        def print_dict_loss(dict_loss):
            for key, loss in dict_loss.items():
                print(f'Loss {ley:12} : {loss:.6f}')
                
        print()    
        print(f'Epoch - {epoch+1} / {args.epochs}')
        print_dict_loss(running_loss)
        if args.dataset_validation: print_dict_loss(validation_running_loss)
        print()
        
        if args.output:
            model.save(args.output)
        
if __name__ == '__main__':
    args = get_args_parser()
    validation_parser(args)
    print(args)
    main(args)