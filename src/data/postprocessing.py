import numpy as np
import torch


def postprocessing(fake_3d):
    try:
        fake_3d = fake_3d.detach().cpu()
    except:
        pass
    nphase = fake_3d.shape[1]
    return 255 * torch.argmax(fake_3d, 1) / (nphase - 1)