import torch
import os
import datasets, models, utils

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Deep Face Drawing: Inference')
    parser.add_argument('--weight', type=str, required=True, help='Path to load model weights.')
    parser.add_argument('--image', type=str, default=None, help='Path to read image and be inferenced.')
    parser.add_argument('--folder', type=str, default=None, help='Path to folder to be inference.')
    parser.add_argument('--output', type=str, required=True, help='Path to save result image.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--manifold', action='store_true', help='Use manifold projection in the model.')
    args = parser.parse_args()
    return args

def inference(model, path_image, path_output, device, args):
    image = datasets.dataloader.load_one_sketch(path_image, simplify=True, device=args.device).unsqueeze(0).to(device)
    print(f'Loaded image from {path_image}')

    with torch.no_grad():
        result = model(image)
    result = utils.convert.tensor2PIL(result[0])
    result.save(path_output)
    print(f'Saved result to {path_output}')

def main(args):
    
    device = torch.device(args.device)
    print(f'Device : {device}')
    
    model = models.DeepFaceDrawing(
        CE=True, CE_encoder=True, CE_decoder=False,
        FM=True, FM_decoder=True,
        IS=True, IS_generator=True, IS_discriminator=False,
        manifold=args.manifold
    )
    model.load(args.weight)
    model.to(device)
    model.eval()
    
    if args.image:
        inference(model, args.image, args.output, device, args)
    
    if args.folder:
        os.makedirs(args.output, exist_ok=True)
        for file_name in os.listdir(args.folder):
            path_image = os.path.join(args.folder, file_name)
            path_output = os.path.join(args.output, file_name)
            inference(model, path_image, path_output, device, args)
    
if __name__ == '__main__':
    args = get_args_parser()
    print(args)
    main(args)