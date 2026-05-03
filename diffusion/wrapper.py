import torch
import torch.nn as nn
from torch.func import functional_call

class Step(nn.Module):
    def __init__(self, unet, hypernet):
        super().__init__()
        self.unet = unet
        self.hypernet = hypernet

    def make_params(self, x):
        return self.hypernet.forward(x)
    
    def forward(self, x, t, params):
        return functional_call(self.unet, params, (t, x), strict=False)