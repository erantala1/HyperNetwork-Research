
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from preprocess import *
from fno import *
import math
from collections import defaultdict

class HyperNetwork(nn.Module):
    def __init__(self, hidden_width, num_mlp_layers, network, device, rank=1):
        super().__init__()
        self.device = device
        self.indices = {}
        self.num_mlp_layers = num_mlp_layers

        self.total_mlps = 0
        self.indices = {}
        self.jobs = []
        self.base_params = dict(network.named_parameters())
        
        mlp_names = []
        out_dims = []
        splits_list = []
        hyper_hidden_width = hidden_width
        self.preprocess_step = CNNConditioner()
        #self.preprocess_step = FNOConditioner(in_channels=1, modes1=16, modes2=16, width=16, out_dim=512, num_layers=2, use_layernorm=False)
        in_dim = self.preprocess_step.out_dim
        self.rank = rank

        for rank_idx in range(self.rank):
            for name, param in network.named_parameters():
                #if not self.should_adapt(name):
                #    continue

                dims = list(param.shape)
                out_dim = sum(dims)

                # if len(dims) == 4 and dims[2] != 1 and dims[3] != 1: # not giving updates to tensors
                #     continue
                # if len(dims) == 4: # hard coding this reshape logic for now 
                #     out_dim -= 2
                
                # #if out_dim < 256:
                # if out_dim < 32:
                #     continue
                
                print(f"MLP updating param: {name} with shape: {dims}, rank: {rank_idx}")
                mlp_idx = self.total_mlps

                mlp_names.append(f"{name}_rank{rank_idx}")
                out_dims.append(out_dim)
                splits_list.append(tuple(param.shape))
                #splits_list.append(tuple(param.shape[:2]))

                unsqueeze = (len(dims) == 4)
                base = self.base_params[name]
                #splits = tuple(base.shape[:2])
                splits = tuple(base.shape)

                self.jobs.append((name, mlp_idx, splits, base, unsqueeze))
                self.total_mlps += 1

        print(self.total_mlps)

        self.batched_mlp = GroupedBatchedMLP(
            out_dims=out_dims,
            in_dim=in_dim,
            hidden_dim=hyper_hidden_width,
            num_hidden_layers=self.num_mlp_layers,
            activation=F.gelu,
            device=device,
            dtype=torch.float32,
        ).to(device)

    def should_adapt(self, name):
        if name.startswith("time"): return False
        if "mlp" in name: return False
        #if "block2" in name: return False
        #if "conv" in name: return False
        #if "bias" in name: return False
        return True

    def make_vectors(self, mlp_out, splits):
        return torch.split(mlp_out, splits, dim=1)

    # use einsum instead of this
    def broadcasting(self, vecs, unsqueeze):
        n = len(vecs)
        if n == 1:
            return vecs[0]
        if n == 2:
            v0, v1 = vecs
            return v0.unsqueeze(-1) * v1.unsqueeze(-2)
        
        out = vecs[0]
        for i in range(1, n):
            v = vecs[i]
            out = out.unsqueeze(-1)
            shape = [v.shape[0]] + [1] * (out.dim() - 2) + [v.shape[1]]
            v_view = v.view(*shape)
            out = out * v_view
        return out

    # use this broadcasting function when ignoring tensors
    # def broadcasting(self, vecs, unsqueeze):
    #     if len(vecs) == 1:
    #         return vecs[0]
    #     v0, v1 = vecs
    #     out = v0.unsqueeze(2) * v1.unsqueeze(1)
    #     if unsqueeze:
    #         out = out.unsqueeze(-1).unsqueeze(-1)
    #     return out
    
    def make_update(self, mlp_out, splits, unsqueeze):
        v = self.make_vectors(mlp_out, splits)
        return self.broadcasting(v, unsqueeze)
    
    def forward(self, U):
        u = self.preprocess_step(U)
        Y = self.batched_mlp(u)
        new_params = {}

        for (name, mlp_idx, splits, base, unsqueeze) in self.jobs:
            mlp_out = Y[mlp_idx]
            update = self.make_update(mlp_out, splits, unsqueeze)
            if name in new_params:
                new_params[name] = new_params[name] + update
            else:
                new_params[name] = base.unsqueeze(0) + update

        return new_params


class GroupedBatchedMLP(nn.Module):
    """
    A collection of batched MLPs grouped by identical output dimension.

    Each group contains M_g independent MLPs with the same:
      - in_dim
      - hidden_dim
      - num_hidden_layers
      - out_dim   (shared within the group)

    This avoids padding all MLPs to one global max_out - this gave inflated model parameter count
    """

    def __init__(self, out_dims, in_dim, hidden_dim, num_hidden_layers, activation=F.silu, device=None, dtype=None,):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation

        # Build groups: out_dim -> list of global mlp indices
        groups = defaultdict(list)
        for global_idx, out_dim in enumerate(out_dims):
            groups[int(out_dim)].append(global_idx)

        self.group_out_dims = sorted(groups.keys())
        self.group_to_global = {str(out_dim): groups[out_dim] for out_dim in self.group_out_dims}
        self.global_to_group = {}

        # Metadata for mapping global MLP index -> (group_key, local_idx)
        for out_dim in self.group_out_dims:
            global_indices = groups[out_dim]
            for local_idx, global_idx in enumerate(global_indices):
                self.global_to_group[global_idx] = (str(out_dim), local_idx)

        # One batched sub-MLP per out_dim group
        self.groups = nn.ModuleDict()
        for out_dim in self.group_out_dims:
            M_g = len(groups[out_dim])
            self.groups[str(out_dim)] = _BatchedMLPExactOut(
                M=M_g,
                in_dim=in_dim,
                hidden_dim=min(int(out_dim*0.5) + 1,512), # hard coding this for now
                out_dim=out_dim,
                num_hidden_layers=num_hidden_layers,
                activation=activation,
                device=device,
                dtype=dtype,
            )

    def forward_grouped(self, x):
        return {
            key: module(x)
            for key, module in self.groups.items()
        }

    def forward(self, x):
        grouped = self.forward_grouped(x)
        out = {}
        for key, Y in grouped.items():
            global_indices = self.group_to_global[key]
            for local_idx, global_idx in enumerate(global_indices):
                out[global_idx] = Y[local_idx]
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class _BatchedMLPExactOut(nn.Module):
    """
    A stack of M independent MLPs with the same exact out_dim.
    """
    def __init__(self, M, in_dim, hidden_dim, out_dim, num_hidden_layers, activation=F.silu, device=None, dtype=None,):
        super().__init__()
        self.M = M
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation

        self.W_in = nn.Parameter(torch.empty(M, hidden_dim, in_dim, device=device, dtype=dtype))
        self.b_in = nn.Parameter(torch.empty(M, hidden_dim, device=device, dtype=dtype))

        L = num_hidden_layers
        self.W_h = nn.Parameter(torch.empty(L, M, hidden_dim, hidden_dim, device=device, dtype=dtype))
        self.b_h = nn.Parameter(torch.empty(L, M, hidden_dim, device=device, dtype=dtype))

        self.W_out = nn.Parameter(torch.empty(M, out_dim, hidden_dim, device=device, dtype=dtype))
        self.b_out = nn.Parameter(torch.empty(M, out_dim, device=device, dtype=dtype))

        self.reset_parameters()
        self.dropout = nn.Dropout(0.1)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.W_in)
        nn.init.zeros_(self.b_in)

        for k in range(self.num_hidden_layers):
            nn.init.xavier_normal_(self.W_h[k])
            nn.init.zeros_(self.b_h[k])

        nn.init.xavier_normal_(self.W_out)
        nn.init.zeros_(self.b_out)

    def bmm_linear(self, X_mb_i, W_m_o_i, b_m_o):
        # X: (M, B, in), W: (M, out, in), b: (M, out)
        return torch.bmm(X_mb_i, W_m_o_i.transpose(1, 2)) + b_m_o[:, None, :]

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B = x.shape[0]
        # (B, in_dim) -> (M, B, in_dim)
        x = x.unsqueeze(0).expand(self.M, B, self.in_dim)

        h = self.bmm_linear(x, self.W_in, self.b_in)
        h = self.activation(h)

        for k in range(self.num_hidden_layers):
            h = self.bmm_linear(h, self.W_h[k], self.b_h[k])
            h = self.activation(h)
            h = self.dropout(h)

        out = self.bmm_linear(h, self.W_out, self.b_out)
        return out