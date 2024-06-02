import torch
import torch.nn as nn
import torch.nn.functional as F

from .self_attention import Self_Attention_3D


class Generator_SRGAN(nn.Module):
    def __init__(self, gf, gk, gs, gp):
        super(Generator_SRGAN, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for lay, (k, s, p) in enumerate(zip(gk, gs, gp)):
            self.convs.append(nn.utils.spectral_norm(nn.ConvTranspose3d(gf[lay], gf[lay+1], k, s, p, bias=False)))
            self.bns.append(nn.BatchNorm3d(gf[lay+1]))  
        self.attention = Self_Attention_3D(gf[-2])

    
    def forward(self, x):
        for conv, bn in zip(self.convs[:-1], self.bns[:-1]):
            x = F.relu_(bn(conv(x)))
        x, _ = self.attention(x)
        return torch.softmax(self.convs[-1](x), 1)


class Critic_SRGAN(nn.Module):
    def __init__(self, cf, ck, cs, cp):
        super(Critic_SRGAN, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for lay, (k, s, p) in enumerate(zip(ck, cs, cp)):
            self.convs.append(nn.utils.spectral_norm(nn.Conv2d(cf[lay], cf[lay+1], k, s, p, bias=False)))
        self.attention = Self_Attention_3D(cf[-2])
    
    def forward(self, x):
        for conv in self.convs[:-1]:
            x = F.relu_(conv(x))
        x, _ = self.attention(x)
        return self.convs[-1](x)


if __name__ == '__main__':
    from torchsummary import summary

    netG = Generator_SRGAN([64, 512, 256, 128, 64, 32, 2], [4, 4, 4, 4, 4, 4], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 3])
    netC = Critic_SRGAN([2, 32, 64, 128, 256, 512, 1], [4, 4, 4, 4, 4, 4], [2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 0])

    # Check if CUDA is available and move the model to GPU
    if torch.cuda.is_available():
        netG = netG.cuda()
        netC = netC.cuda()
    
    # Specify the input shape and move the tensor to the same device as the model
    input_shape = (64, 4, 4, 4)  # Batch size is implicit here
    x = torch.randn((1,) + input_shape)  # Adding batch dimension
    if torch.cuda.is_available():
        x = x.cuda()
    summary(netG, input_shape)
    
    input_shape = (2, 128, 128)
    x = torch.randn((1,) + input_shape)  # Adding batch dimension
    if torch.cuda.is_available():
        x = x.cuda()
    summary(netC, input_shape)