import models
import torch

def main():
    model = models.DeepFaceDrawing(
        CE=True, CE_encoder=True, CE_decoder=True,
        FM=True, FM_decoder=True,
        IS=True, IS_generator=True, IS_discriminator=True,
        manifold=False
    )

    sketch = torch.rand(2, 1, 512, 512)
    latents = model.CE.encode(sketch)

    for component in model.components:
        print(f'latents {component}: {latents[component].shape}')
    
    resketches = model.CE.decode(latents)
    for component in model.components:
        print(f'resketches {component}: {resketches[component].shape}')

    spatial_map = model.FM.decode(latents)
    print(f'spatial_map: {spatial_map.shape}')
    
    fake = model.IS.generate(spatial_map)
    print(f'fake: {fake.shape}')

    patch_D1, patch_D2, patch_D3 = model.IS.discriminate(spatial_map, fake)
    print(f'patch_D1: {patch_D1.shape}')
    print(f'patch_D2: {patch_D2.shape}')
    print(f'patch_D3: {patch_D3.shape}')

    path = 'temp/DeepFaceDrawing'
    print(f'Saving model to {path}')
    model.save(path)
    print(f'Loading model from {path}')
    model.load(path)
    
if __name__ == '__main__':
    main()
    
