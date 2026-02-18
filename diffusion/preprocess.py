# preprocess.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PoolPyramidConditioner(nn.Module):
    def __init__(self, C: int, sizes=(32, 16, 8)):
        super().__init__()
        self.C = C
        self.sizes = sizes
        out_dim = sum(C * s * s for s in sizes)

        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # accept x as (C,H,W) or (B,C,H,W)
        added_batch = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            added_batch = True

        B, C, H, W = x.shape
        feats = []
        for s in self.sizes:
            pooled = F.adaptive_avg_pool2d(x, (s, s))
            feats.append(pooled.reshape(B, -1))

        z = torch.cat(feats, dim=1)

        if added_batch:
            z = z.squeeze(0)
        return z