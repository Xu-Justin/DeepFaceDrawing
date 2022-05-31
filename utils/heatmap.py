import matplotlib.pyplot as plt
import seaborn as sns
from . import convert

def heatmap(tensor, title=''):
    if tensor.dim() == 2:
        return heatmap2D(tensor, title)
    if tensor.dim() == 3:
        return heatmap3D(tensor, title)
    if tensor.dim() == 4:
        return heatmap4D(tensor, title)
    assert False, f"[heatmap] Unexpected input tensor dim {tensor.dim()}"

def heatmap2D(tensor, title=''):
    assert tensor.dim() == 2, f"[heatmap2D] Expected input tensor dim {2}, but received {tensor.dim()}."
    return heatmap3D(tensor.unsqueeze(0), title)

def heatmap3D(tensor, title=''):
    assert tensor.dim() == 3, f"[heatmap3D] Expected input tensor dim {3}, but received {tensor.dim()}."
    return heatmap4D(tensor.unsqueeze(0), title)

def heatmap4D(tensor, title=''):
    assert tensor.dim() == 4, f"[heatmap4D] Expected input tensor dim {4}, but received {tensor.dim()}."
    data = tensor.detach().cpu().clone().numpy()
    vmin = data.min()
    vmax = data.max()
    nrow = data.shape[0]
    ncol = data.shape[1]
    fig, axs = plt.subplots(nrow, ncol, figsize=(7 * ncol, 6 * nrow), squeeze=False, constrained_layout=True)
    for row in range(nrow):
        for col in range(ncol):
            sns.heatmap((data[row][col]), vmin=vmin, vmax=vmax, square=True, ax=(axs[(row, col)]), cbar=False)
        else:
            fig.colorbar((axs[(0, 0)].collections[0]), ax=(axs[:, :]), shrink=0.8)
            if title:
                fig.suptitle(title, fontsize=16)
    return convert.fig2img(fig)