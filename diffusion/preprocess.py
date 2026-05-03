# preprocess.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PoolPyramidConditioner(nn.Module):
    def __init__(self, C = 1, sizes=(32, 16, 8)):
        super().__init__()
        self.C = C
        self.sizes = sizes
        out_dim = sum(C * s * s for s in sizes)

        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # accept x as (C,H,W) or (B,C,H,W)
        #added_batch = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            added_batch = True
        B, C, H, W = x.shape
        feats = []
        for s in self.sizes:
            pooled = F.adaptive_avg_pool2d(x, (s, s))
            feats.append(pooled.reshape(B, -1))

        z = torch.cat(feats, dim=1)
        
        #condition_noise = torch.randn_like(z) * 0.01
        #z = z + condition_noise

        return z
    
class CNNConditioner(nn.Module):
    def __init__(self, C = 1, hidden_dims=(32, 64, 128, 256), out_dim=512, use_groupnorm=True,):
        super().__init__()
        self.C = C
        self.out_dim = out_dim

        dims = [C, *hidden_dims]
        blocks = []

        for i in range(len(dims) - 1):
            in_ch = dims[i]
            out_ch = dims[i + 1]
    
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                    nn.SiLU(),
                )
            )

        self.encoder = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (B, F, 1, 1)
            nn.Flatten(),              # (B, F)
            nn.LayerNorm(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)

        h = self.encoder(x)
        z = self.head(h)
        return z

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x)

        W_rfft = W // 2 + 1

        low = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1
        )  # (B, out_ch, modes1, modes2)

        high = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2],
            self.weights2
        )  # (B, out_ch, modes1, modes2)

        mid_h = H - 2 * self.modes1
        zeros_mid_h = torch.zeros(
            batchsize, self.out_channels, mid_h, self.modes2,
            dtype=torch.cfloat, device=x.device
        )

        left_block = torch.cat([low, zeros_mid_h, high], dim=2)  # (B, out_ch, H, modes2)

        zeros_right = torch.zeros(
            batchsize, self.out_channels, H, W_rfft - self.modes2,
            dtype=torch.cfloat, device=x.device
        )

        out_ft = torch.cat([left_block, zeros_right], dim=3)  # (B, out_ch, H, W_rfft)

        x = torch.fft.irfft2(out_ft, s=(H, W))
        return x


class PointwiseMLP2d(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, 1),
        )

    def forward(self, x):
        return self.net(x)


class FNOConditioner(nn.Module):
    """
    Input:  (B, C, H, W) or (C, H, W)
    Output: (B, out_dim) or (out_dim,)
    """
    def __init__(
        self,
        in_channels=1,
        modes1=16,
        modes2=16,
        width=64,
        out_dim=512,
        num_layers=4,
        use_layernorm=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.width = width
        self.out_dim = out_dim
        self.num_layers = num_layers

        self.p = nn.Linear(in_channels + 2, width)

        self.spec_convs = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2)
            for _ in range(num_layers)
        ])
        self.mlps = nn.ModuleList([
            PointwiseMLP2d(width, width, width)
            for _ in range(num_layers)
        ])
        self.ws = nn.ModuleList([
            nn.Conv2d(width, width, 1)
            for _ in range(num_layers)
        ])

        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.norms = nn.ModuleList([
                nn.LayerNorm(width) for _ in range(num_layers)
            ])
        else:
            self.norms = None

        self.compress = nn.Sequential(
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.GELU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 4 * 4, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        # Accept either (C, H, W) or (B, C, H, W)
        added_batch_dim = False
        if x.dim() == 3:
            x = x.unsqueeze(0)   # (1, C, H, W)
            added_batch_dim = True

        b, c, h, w = x.shape

        x = x.permute(0, 2, 3, 1).contiguous()   # (B, H, W, C)
        grid = self.get_grid(b, h, w, x.device)  # (B, H, W, 2)
        x = torch.cat([x, grid], dim=-1)         # (B, H, W, C+2)

        x = self.p(x)                            # (B, H, W, width)
        x = x.permute(0, 3, 1, 2).contiguous()   # (B, width, H, W)

        for k in range(self.num_layers):
            x1 = self.spec_convs[k](x)
            x1 = self.mlps[k](x1)
            x2 = self.ws[k](x)
            x = x1 + x2

            if self.use_layernorm:
                x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, width)
                x = self.norms[k](x)
                x = x.permute(0, 3, 1, 2).contiguous()  # (B, width, H, W)

            if k < self.num_layers - 1:
                x = F.gelu(x)

        #x = self.pool(x)   # (B, width, 1, 1)
        x = self.compress(x)
        x = self.pool(x)
        z = self.head(x)   # (B, out_dim)

        if added_batch_dim:
            z = z.squeeze(0)   # (out_dim,)

        return z

    def get_grid(self, batchsize, size_x, size_y, device):
        gridx = torch.linspace(0, 1, size_x, device=device).view(1, size_x, 1, 1)
        gridx = gridx.repeat(batchsize, 1, size_y, 1)

        gridy = torch.linspace(0, 1, size_y, device=device).view(1, 1, size_y, 1)
        gridy = gridy.repeat(batchsize, size_x, 1, 1)

        return torch.cat([gridx, gridy], dim=-1)
