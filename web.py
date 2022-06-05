import os, shutil
from flask import Flask, render_template, request, redirect, url_for

import torch
from torchvision import transforms
import models, datasets, utils

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Deep Face Drawing: Inference')
    parser.add_argument('--weight', type=str, required=True, help='Path to load model weights.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--manifold', action='store_true', help='Use manifold projection in the model.')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    return args

class Storage:
    
    def generate_folder(self):
        self.folder_name = str(hash(self))
        while os.path.exists(self.folder_name):
            self.folder_name = str(hash(self.folder_name))
        os.makedirs(self.folder_name)
        os.makedirs(os.path.join(self.folder_name, 'sketch'))
        os.makedirs(os.path.join(self.folder_name, 'photo'))
    
    def delete_folder(self):
        shutil.rmtree(self.folder_name)
    
    def get_folder_path(self):
        return os.path.abspath(self.folder_name)
    
    def __enter__(self):
        self.generate_folder()
        return self
    
    def __exit__(self, *args, **kwargs):
        self.delete_folder()

def main(args, storage):
    
    device = torch.device(args.device)
    print(f'Device : {device}')
    
    model = models.DeepFaceDrawing(
        CE=True, CE_encoder=True, CE_decoder=False,
        FM=True, FM_decoder=True,
        IS=True, IS_generator=True, IS_discriminator=False,
        manifold=args.manifold
    )
    model.load(args.weight, map_location=device)
    model.to(args.device)
    model.eval()
    
    template_folder = os.path.abspath('resources/templates/')
    app = Flask(__name__, template_folder=template_folder, static_folder=storage.get_folder_path())
    
    @app.route('/', methods=['GET', 'POST'])
    def index():
        if request.method == 'GET':
            return render_template('index.html')
        if request.method == 'POST':
            sketch = request.files['image']
            file_name = str(hash(str(sketch))) + '.jpg'
            sketch.save(os.path.join(storage.get_folder_path(), 'sketch', file_name))
            return redirect(url_for('forward', file_name=file_name))
    
    @app.route('/forward/<file_name>', methods=['GET'])
    def forward(file_name):
        x = datasets.dataloader.load_one_sketch(os.path.join(storage.get_folder_path(), 'sketch', file_name), simplify=True, device=args.device).unsqueeze(0).to(device)
        x = model(x)
        x = utils.convert.tensor2PIL(x[0])
        x.save(os.path.join(storage.get_folder_path(), 'photo', file_name))
        return redirect(url_for('display', file_name=file_name))
    
    @app.route('/display/<file_name>', methods=['GET'])
    def display(file_name):
        return render_template('display.html', file_name=file_name)
        
    host = args.host
    port = args.port
    app.run(host, port)
    
if __name__ == '__main__':
    args = get_args_parser()
    print(args)
    with Storage() as storage:
        main(args, storage)