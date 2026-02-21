import torch
import torch.nn as nn
from torch.func import functional_call

# wrapper for pairing hypernet with unet
# use torch.compile on step.make_params for speedup
class Step(nn.Module):
    def __init__(self, unet, hypernet, preprocess):
        super().__init__()
        self.unet = unet
        self.hypernet = hypernet
        self.preprocess = preprocess

    def make_params(self, x):
        z = self.preprocess(x)
        params_batched = self.hypernet.forward_multiple_mlp(z)
        self.params = {k: v.squeeze(0) for k, v in params_batched.items()}
    
    def forward(self, x, t):
        return functional_call(self.unet, self.params, (t, x), strict=False)
