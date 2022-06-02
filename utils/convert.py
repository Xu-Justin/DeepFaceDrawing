import io
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

plt.rcParams['savefig.dpi'] = 72

def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf)

def tensor2PIL(tensor):
    return transforms.ToPILImage()(tensor)

def tensor2numpy(tensor):
    return tensor.detach().cpu().clone().numpy()

def numpy2PIL(numpy):
    return Image.fromarray(np.uint8(numpy))