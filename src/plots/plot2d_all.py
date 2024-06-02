import os

import matplotlib.pyplot as plt

from data.postprocessing import postprocessing


def plot2d(data, axis, slice_idx, ax, title):
    if axis == 0:
        ax.imshow(data[slice_idx, :, :], cmap='gray')
    elif axis == 1:
        ax.imshow(data[:, slice_idx, :], cmap='gray')
    else:
        ax.imshow(data[:, :, slice_idx], cmap='gray')
    ax.set_title(title)


def plot2d_all(config, epoch, iter, data, num_slices, outdir):
    data = postprocessing(data)[0]
    fig, axs = plt.subplots(num_slices, 3, figsize=(15, 5 * num_slices))
    
    for i in range(num_slices):
        plot2d(data, 0, i, axs[i, 0], f"xy plane at z={i}")
        plot2d(data, 1, i, axs[i, 1], f"zx plane at y={i}")
        plot2d(data, 2, i, axs[i, 2], f"yz plane at x={i}")
    
    plt.tight_layout()
    os.makedirs(outdir + 'plot2d/', exist_ok=True)
    plt.savefig(outdir + f'plot2d/{config["run_name"]}_plot2d_epoch{epoch}_iter{iter}.png')
    plt.close()