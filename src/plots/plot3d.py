import numpy as np
import matplotlib.pyplot as plt


def plot3d(data, downsample, vmin, vmax, aspect=None, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    downsampled_data = data[::downsample, ::downsample, ::downsample]
    
    z, y, x = np.indices(downsampled_data.shape)
    x = x.flatten() * downsample
    y = y.flatten() * downsample
    z = z.flatten() * downsample
    
    color_values = downsampled_data.flatten()
    scatter = ax.scatter(x, y, z, c=color_values, marker='o', cmap='gray', vmin=vmin, vmax=vmax)
    
    cbar = plt.colorbar(scatter, pad=0.15, shrink=0.7)
    cbar.set_label("pixel value")
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    y_max = downsampled_data.shape[1] * downsample
    ax.set_yticks(np.arange(0, y_max, 50))
    z_max = downsampled_data.shape[0] * downsample
    ax.set_zticks(np.arange(0, z_max, 50))
    
    ax.set_box_aspect(aspect)
    
    plt.title(title)
    plt.tight_layout()
    plt.show()