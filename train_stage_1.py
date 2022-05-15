import os
from tqdm import tqdm
from PIL import Image

import torch
from dataset import dataloader
from model import DeepFaceDrawing

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Deep Face Drawing: Train Stage 1')
    parser.add_argument('--dataset', type=str, required=True, help='Path to training dataset.')
    parser.add_argument('--dataset_validation', type=str, default=None, help='Path to validation dataset.')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--resume', type=str, default=None, help='Path to load model weights.')
    parser.add_argument('--output', type=str, default=None, help='Path to save weights.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--comet', type=str, default=None, help='comet.ml API')
    args = parser.parse_args()
    return args

def main(args):
    
    if args.comet:
        from comet_ml import Experiment
        experiment = Experiment(
            api_key=args.comet,
            project_name="Deep Face Drawing: Training Stage 1",
            workspace="xu-justin",
            log_code=True
        )
    
    device = torch.device(args.device)
    print(f'Device : {device}')
    
    model = DeepFaceDrawing(
        CE=True, CE_encoder=True, CE_decoder=True,
        FM=False, IS=False, manifold=False
    )
    
    if args.resume:
        model.load(args.resume)
        
    model.to(device)

    train_dataloader = dataloader(args.dataset, batch_size=args.batch_size, load_photo=False)
    
    if args.dataset_validation:
        validation_dataloader = dataloader(args.dataset_validation, batch_size=args.batch_size, load_photo=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = torch.nn.MSELoss()
        
    for epoch in range(args.epochs):
        
        running_loss = {
            'loss_left_eye' : 0,
            'loss_right_eye' : 0,
            'loss_nose' : 0,
            'loss_mouth' : 0,
            'loss_background' : 0
        }
        
        model.train()
        for sketches in tqdm(train_dataloader, desc=f'Epoch - {epoch+1} / {args.epochs}'):
            optimizer.zero_grad()
            for key, CEs in model.CE.items():
                X = CEs.crop(sketches).to(device)
                y = CEs(X)
                loss = criterion(y, X)
                loss.backward()
                running_loss[f'loss_{key}'] += loss.item() * len(sketches) / len(train_dataloader.dataset)
            optimizer.step()
        
        if args.dataset_validation:
            validation_running_loss = {
                'val_loss_left_eye' : 0,
                'val_loss_right_eye' : 0,
                'val_loss_nose' : 0,
                'val_loss_mouth' : 0,
                'val_loss_background' : 0
            }
            
            model.eval()
            with torch.no_grad():
                for sketches in tqdm(validation_dataloader, desc=f'Validation Epoch - {epoch+1} / {args.epochs}'):
                    for key, CEs in model.CE.items():
                        X = CEs.crop(sketches).to(device)
                        y = CEs(X)
                        loss = criterion(y, X)
                        validation_running_loss[f'val_loss_{key}'] += loss.item() * len(sketches) / len(validation_dataloader.dataset)
        
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
        
        if args.output:
            model.save(args.output)
            
    if args.comet:
        experiment.end()
        
if __name__ == '__main__':
    args = get_args_parser()
    print(args)
    main(args)