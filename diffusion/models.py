import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.register_buffer("emb", torch.exp(torch.arange(half_dim) * -emb))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = x * self.emb
        emb = torch.concatenate((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb


class LinearTimeSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.group_norm = nn.GroupNorm(min(dim // 4, 32), dim)
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle (C,H,W) -> (1,C,H,W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
            remove_batch_dim = True
        else:
            remove_batch_dim = False

        n, c, h, w = x.shape
        x = self.group_norm(x)

        qkv = self.to_qkv(x)  # (N, 3*heads*dim_head, H, W)
        qkv = rearrange(
            qkv, "n (qkv heads c) h w -> qkv n heads c (h w)", heads=self.heads, qkv=3
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # (N, heads, c, HW)

        k = F.softmax(k, dim=-1)

        # context: sum over spatial -> (N, heads, c, c)
        context = torch.einsum("nhdc,nhec->nhde", k, v)

        # out: (N, heads, c, HW)
        out = torch.einsum("nhde,nhdc->nhec", context, q)

        out = rearrange(out, "n heads c (h w) -> n (heads c) h w", heads=self.heads, h=h, w=w)
        out = self.to_out(out)

        if remove_batch_dim:
            out = out.squeeze(0)
        return out


def upsample_2d(y: torch.Tensor, factor: int = 2) -> torch.Tensor:
    # Handle both (C,H,W) and (N,C,H,W)
    if y.dim() == 3:
        C, H, W = y.shape
        y = y.reshape([C, H, 1, W, 1])
        y = y.tile([1, 1, factor, 1, factor])
        return y.reshape([C, H * factor, W * factor])
    else:
        N, C, H, W = y.shape
        y = y.reshape([N, C, H, 1, W, 1])
        y = y.tile([1, 1, 1, factor, 1, factor])
        return y.reshape([N, C, H * factor, W * factor])


def downsample_2d(y: torch.Tensor, factor: int = 2) -> torch.Tensor:
    # Handle both (C,H,W) and (N,C,H,W)
    if y.dim() == 3:
        C, H, W = y.shape
        y = y.reshape([C, H // factor, factor, W // factor, factor])
        return y.mean(dim=[2, 4])
    else:
        N, C, H, W = y.shape
        y = y.reshape([N, C, H // factor, factor, W // factor, factor])
        return y.mean(dim=[3, 5])


def exact_zip(*args):
    _len = len(args[0])
    for arg in args:
        assert len(arg) == _len
    return zip(*args)


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.fn(x, *args, **kwargs) + x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        is_biggan: bool,
        up: bool,
        down: bool,
        time_emb_dim: int,
        dropout_rate: float,
        is_attn: bool,
        heads: int,
        dim_head: int,
    ):
        super().__init__()
        self.dim_out = dim_out
        self.is_biggan = is_biggan
        self.up = up
        self.down = down

        self.mlp_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out),
        )

        self.block1_groupnorm = nn.GroupNorm(min(dim_in // 4, 32), dim_in)
        self.block1_conv = nn.Conv2d(dim_in, dim_out, 3, padding=1)

        self.block2_layers = nn.Sequential(
            nn.GroupNorm(min(dim_out // 4, 32), dim_out),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(dim_out, dim_out, 3, padding=1),
        )

        assert not self.up or not self.down

        if is_biggan:
            if self.up:
                self.scaling = upsample_2d
            elif self.down:
                self.scaling = downsample_2d
            else:
                self.scaling = None
        else:
            if self.up:
                self.scaling = nn.ConvTranspose2d(dim_in, dim_in, kernel_size=4, stride=2, padding=1)
            elif self.down:
                self.scaling = nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=2, padding=1)
            else:
                self.scaling = None

        self.res_conv = nn.Conv2d(dim_in, dim_out, kernel_size=1)

        if is_attn:
            self.attn = Residual(
                LinearTimeSelfAttention(dim_out, heads=heads, dim_head=dim_head)
            )
        else:
            self.attn = None

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Handle (C,H,W) -> (1,C,H,W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
            remove_batch_dim = True
        else:
            remove_batch_dim = False

        _, C, _, _ = x.shape

        h = F.silu(self.block1_groupnorm(x))

        if self.up or self.down:
            h = self.scaling(h)  # type: ignore
            x = self.scaling(x)  # type: ignore

        h = self.block1_conv(h)

        t = self.mlp_layers(t)
        h = h + t[..., None, None]

        h = self.block2_layers(h)

        if C != self.dim_out or self.up or self.down:
            x = self.res_conv(x)

        out = (h + x) / math.sqrt(2)
        if self.attn is not None:
            out = self.attn(out)

        if remove_batch_dim:
            out = out.squeeze(0)
        return out


class UNet(nn.Module):
    def __init__(
        self,
        data_shape: tuple[int, int, int],
        is_biggan: bool,
        dim_mults: list[int],
        hidden_size: int,
        heads: int,
        dim_head: int,
        dropout_rate: float,
        num_res_blocks: int,
        attn_resolutions: list[int],
    ):
        super().__init__()

        data_channels, in_height, in_width = data_shape
        dims = [hidden_size] + [hidden_size * m for m in dim_mults]
        in_out = list(exact_zip(dims[:-1], dims[1:]))

        self.time_pos_emb = SinusoidalPosEmb(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.SiLU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

        self.first_conv = nn.Conv2d(data_channels, hidden_size, kernel_size=3, padding=1)

        h, w = in_height, in_width

        # Down path
        self.down_res_blocks = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_attn = (h in attn_resolutions and w in attn_resolutions)

            res_blocks = nn.ModuleList([
                ResnetBlock(
                    dim_in=dim_in,
                    dim_out=dim_out,
                    is_biggan=is_biggan,
                    up=False,
                    down=False,
                    time_emb_dim=hidden_size,
                    dropout_rate=dropout_rate,
                    is_attn=is_attn,
                    heads=heads,
                    dim_head=dim_head,
                )
            ])

            for _ in range(num_res_blocks - 2):
                res_blocks.append(
                    ResnetBlock(
                        dim_in=dim_out,
                        dim_out=dim_out,
                        is_biggan=is_biggan,
                        up=False,
                        down=False,
                        time_emb_dim=hidden_size,
                        dropout_rate=dropout_rate,
                        is_attn=is_attn,
                        heads=heads,
                        dim_head=dim_head,
                    )
                )

            if ind < (len(in_out) - 1):
                res_blocks.append(
                    ResnetBlock(
                        dim_in=dim_out,
                        dim_out=dim_out,
                        is_biggan=is_biggan,
                        up=False,
                        down=True,
                        time_emb_dim=hidden_size,
                        dropout_rate=dropout_rate,
                        is_attn=is_attn,
                        heads=heads,
                        dim_head=dim_head,
                    )
                )
                h, w = h // 2, w // 2

            self.down_res_blocks.append(res_blocks)

        # Middle
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(
            dim_in=mid_dim,
            dim_out=mid_dim,
            is_biggan=is_biggan,
            up=False,
            down=False,
            time_emb_dim=hidden_size,
            dropout_rate=dropout_rate,
            is_attn=True,
            heads=heads,
            dim_head=dim_head,
        )
        self.mid_block2 = ResnetBlock(
            dim_in=mid_dim,
            dim_out=mid_dim,
            is_biggan=is_biggan,
            up=False,
            down=False,
            time_emb_dim=hidden_size,
            dropout_rate=dropout_rate,
            is_attn=False,
            heads=heads,
            dim_head=dim_head,
        )

        # Up path
        self.ups_res_blocks = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_attn = (h in attn_resolutions and w in attn_resolutions)

            res_blocks = nn.ModuleList()
            for _ in range(num_res_blocks - 1):
                res_blocks.append(
                    ResnetBlock(
                        dim_in=dim_out * 2,
                        dim_out=dim_out,
                        is_biggan=is_biggan,
                        up=False,
                        down=False,
                        time_emb_dim=hidden_size,
                        dropout_rate=dropout_rate,
                        is_attn=is_attn,
                        heads=heads,
                        dim_head=dim_head,
                    )
                )

            res_blocks.append(
                ResnetBlock(
                    dim_in=dim_out + dim_in,
                    dim_out=dim_in,
                    is_biggan=is_biggan,
                    up=False,
                    down=False,
                    time_emb_dim=hidden_size,
                    dropout_rate=dropout_rate,
                    is_attn=is_attn,
                    heads=heads,
                    dim_head=dim_head,
                )
            )

            if ind < (len(in_out) - 1):
                res_blocks.append(
                    ResnetBlock(
                        dim_in=dim_in,
                        dim_out=dim_in,
                        is_biggan=is_biggan,
                        up=True,
                        down=False,
                        time_emb_dim=hidden_size,
                        dropout_rate=dropout_rate,
                        is_attn=is_attn,
                        heads=heads,
                        dim_head=dim_head,
                    )
                )
                h, w = h * 2, w * 2

            self.ups_res_blocks.append(res_blocks)

        self.final_conv_layers = nn.Sequential(
            nn.GroupNorm(min(hidden_size // 4, 32), hidden_size),
            nn.SiLU(),
            nn.Conv2d(hidden_size, hidden_size // 2, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_size // 2, 1, 1),
        )

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Add batch dimension if input is (C,H,W) -> (1,C,H,W)
        if y.dim() == 3:
            y = y.unsqueeze(0)
            add_batch_dim = True
        else:
            add_batch_dim = False

        t = self.time_pos_emb(t)
        t = self.mlp(t)

        h = self.first_conv(y)
        hs = [h]

        for res_blocks in self.down_res_blocks:
            for res_block in res_blocks:
                h = res_block(h, t)
                hs.append(h)

        h = self.mid_block1(h, t)
        h = self.mid_block2(h, t)

        for res_blocks in self.ups_res_blocks:
            for res_block in res_blocks:
                if res_block.up:
                    h = res_block(h, t)
                else:
                    skip = hs.pop()
                    concat_input = torch.concatenate((h, skip), dim=1)  # channel dim for (N,C,H,W)
                    h = res_block(concat_input, t)

        assert len(hs) == 0

        h = self.final_conv_layers(h)

        if add_batch_dim:
            h = h.squeeze(0)

        return h