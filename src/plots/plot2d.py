import numpy as np
import matplotlib.pyplot as plt


def plot2d_xy_plane(data, n_slices, aspect=None):
    # 両端のインデックスを含むため、中間のインデックスを計算
    middle_slice_indices = list(np.linspace(0, data.shape[0] - 1, n_slices, dtype=int))
    slice_indices = np.array([0] + middle_slice_indices[1:-1] + [data.shape[0] - 1])
    
    # 4行3列のサブプロットを作成
    fig, axes = plt.subplots(3, 4, figsize=(15, 9))
    
    for i in range(3):
        for j in range(4):
            idx = slice_indices[i * 4 + j]
            axes[i, j].imshow(data[idx, :, :].squeeze(), aspect=aspect[0]/aspect[1], cmap='gray')
            axes[i, j].set_title(f'xy plane at z={idx}')
            axes[i, j].set_xlabel('x')
            axes[i, j].set_ylabel('y')
            # メモリを消す
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    
    plt.tight_layout()
    plt.show()


def plot2d_zx_plane(data, n_slices, aspect=None):
    # 両端のインデックスを含むため、中間のインデックスを計算
    middle_slice_indices = list(np.linspace(0, data.shape[1] - 1, n_slices, dtype=int))
    slice_indices = np.array([0] + middle_slice_indices[1:-1] + [data.shape[1] - 1])
    
    # 4行3列のサブプロットを作成
    fig, axes = plt.subplots(4, 3, figsize=(15, 8))
    
    for i in range(4):
        for j in range(3):
            idx = slice_indices[i + 4 * j]
            axes[i, j].imshow(data[:, idx, :].squeeze(), aspect=aspect[2]/aspect[0], cmap='gray')
            axes[i, j].set_title(f'zx plane at y={idx}')
            axes[i, j].set_xlabel('x')
            axes[i, j].set_ylabel('z')
            # メモリを消す
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    
    plt.tight_layout()
    plt.show()


def plot2d_yz_plane(data, n_slices, aspect=None):
    # 両端のインデックスを含むため、中間のインデックスを計算
    middle_slice_indices = list(np.linspace(0, data.shape[2] - 1, n_slices, dtype=int))
    slice_indices = np.array([0] + middle_slice_indices[1:-1] + [data.shape[2] - 1])
    
    # 4行3列のサブプロットを作成
    fig, axes = plt.subplots(4, 3, figsize=(15, 10))
    
    for i in range(4):
        for j in range(3):
            idx = slice_indices[i + 4 * j]
            axes[i, j].imshow(data[:, :, idx].squeeze(), aspect=aspect[2]/aspect[1], cmap='gray')
            axes[i, j].set_title(f'yz plane at x={idx}')
            axes[i, j].set_xlabel('y')
            axes[i, j].set_ylabel('z')
            # メモリを消す
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    
    plt.tight_layout()
    plt.show()