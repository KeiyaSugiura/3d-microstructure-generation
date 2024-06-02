import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset


def preprocessing(config):
    xy_zx_yz_dataset = []
    for input_path in config["input_paths"]:
        img = plt.imread(input_path)
        
        if len(img.shape) > 2:
            img = img[:, :, 0]
        
        x_max, y_max= img.shape
        num_phases = np.unique(img)
        if type(config["num_crops"]) == str:
            num_crops = eval(config["num_crops"])
        else:
            num_crops = config["num_crops"]
        
        data = np.empty([num_crops, len(num_phases), config["crop_size"], config["crop_size"]])
        for i in range(num_crops):
            x = np.random.randint(1, x_max - config["crop_size"] - 1)
            y = np.random.randint(1, y_max - config["crop_size"] - 1)
            for phase_idx, phase in enumerate(num_phases):
                tmp = np.zeros([config["crop_size"], config["crop_size"]])
                tmp[img[x: x + config["crop_size"], y: y + config["crop_size"]] == phase] = 1
                data[i, phase_idx, :, :] = tmp
        data = torch.FloatTensor(data)
        
        dataset = TensorDataset(data)
        xy_zx_yz_dataset.append(dataset)
    return xy_zx_yz_dataset