import torch
import datasets, models, utils

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Deep Face Drawing: Inference')
    parser.add_argument('--weight', type=str, required=True, help='Path to load model weights.')
    parser.add_argument('--image', type=str, required=True, help='Path to read image and be inferenced.')
    parser.add_argument('--output', type=str, required=True, help='Path to save result image.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--manifold', action='store_true', help='Use manifold projection in the model.')
    args = parser.parse_args()
    return args
    
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
    
    image = datasets.dataloader.load_one_sketch(args.image, simplify=True, device=args.device).unsqueeze(0).to(device)
    print(f'Loaded image from {args.image}')

    with torch.no_grad():
        result = model(image)
    result = utils.convert.tensor2PIL(result[0])
    result.save(args.output)
    print(f'Saved result to {args.output}')
    
if __name__ == '__main__':
    args = get_args_parser()
    print(args)
    main(args)