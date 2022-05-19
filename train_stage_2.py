from tqdm import tqdm
from PIL import Image

import torch
from dataset import dataloader, tanh, itanh
from model import DeepFaceDrawing

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Deep Face Drawing: Train Stage 2')
    parser.add_argument('--dataset', type=str, required=True, help='Path to training dataset.')
    parser.add_argument('--dataset_validation', type=str, default=None, help='Path to validation dataset.')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--resume', type=str, default=None, help='Path to load model weights.')
    parser.add_argument('--resume_CE', type=str, default=None, help='Path to load Component Embedding model weights. Required if --resume is not given. Skipped if --resume is given.')
    parser.add_argument('--output', type=str, default=None, help='Path to save weights.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--comet', type=str, default=None, help='comet.ml API')
    parser.add_argument('--comet_log_image', type=str, default=None, help='Path to model input image to be inference and log the result to comet.ml. Skipped if --comet is not given.')
    args = parser.parse_args()
    return args

def validation_parser(args):
    if args.resume:
        if args.resume_CE: print('args.resume_CE will be skipped.')
    else:
        assert args.resume_CE, "Both args.resume and args.resume_CE can't be None." 
    if not args.comet:
        if args.comet_log_image: print('args.comet_log_image will be skipped.')
    
def main(args):
    
    device = torch.device(args.device)
    print(f'Device : {device}')
    
    if args.comet:
        from comet_ml import Experiment
        experiment = Experiment(
            api_key=args.comet,
            project_name="Deep Face Drawing: Training Stage 2",
            workspace="xu-justin",
            log_code=True
        )
        
        if args.comet_log_image:
            from dataset import load_one_sketch
            from utils import stack_preview
            log_image_sketch = load_one_sketch(args.comet_log_image).to(device)
    
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
    
    train_dataloader = dataloader(args.dataset, batch_size=args.batch_size, load_photo=True, augmentation=True)
    
    if args.dataset_validation:
        validation_dataloader = dataloader(args.dataset_validation, batch_size=args.batch_size, load_photo=True)
    
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
            photos = tanh(photos).to(device)
            
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
            
            iteration_loss = {
                'loss_G_it' : loss_G.item(),
                'loss_D_it' : loss_D.item()
            }
            
            for key, loss in iteration_loss.items():
                running_loss[key[:-3]] = loss * len(sketches) / len(train_dataloader.dataset)
            
            if args.comet:
                experiment.log_metrics(iteration_loss)
        
        if args.dataset_validation:
            validation_running_loss = {
                'val_loss_G' : 0,
                'val_loss_D' : 0
            }
            
            model.eval()
            with torch.no_grad():
                for sketches, photos in tqdm(validation_dataloader, desc=f'Validation Epoch - {epoch+1} / {args.epochs}'):
            
                    sketches = sketches.to(device)
                    photos = tanh(photos).to(device)
                    
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
                    
                    validation_iteration_loss = {
                        'val_loss_G_it' : loss_G.item(),
                        'val_loss_D_it' : loss_D.item()
                    }
                    
                    for key, loss in iteration_loss.items():
                        validation_running_loss[key[:-3]] = loss * len(sketches) / len(validation_dataloader.dataset)
                    
                    if args.comet:
                        experiment.log_metrics(validation_iteration_loss)
                        
        def print_dict_loss(dict_loss):
            for key, loss in dict_loss.items():
                print(f'Loss {key:12} : {loss:.6f}')
                
        print()    
        print(f'Epoch - {epoch+1} / {args.epochs}')
        print_dict_loss(running_loss)
        if args.dataset_validation: print_dict_loss(validation_running_loss)
        print()
        
        if args.comet:
            experiment.log_metrics(running_loss, step=epoch+1)
            if args.dataset_validation: experiment.log_metrics(validation_running_loss, step=epoch+1)
            if args.comet_log_image:
                log_image_fake = itanh(model(log_image_sketch))
                log_image_fake = stack_preview(log_image_sketch, log_image_fake)[0]
                experiment.log_image(log_image_fake, step=epoch+1)
                
        if args.output:
            model.save(args.output)
    
    if args.comet:
        experiment.end()
        
if __name__ == '__main__':
    args = get_args_parser()
    print(args)
    validation_parser(args)
    main(args)