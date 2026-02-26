
import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperNetwork(nn.Module):
    def __init__(self, num_mlp_layers, in_dim, network, device):
        super().__init__()
        self.network = network.to(device)
        self.device = device
        self.indices = {}
        self.num_mlp_layers = num_mlp_layers

        self.total_mlps = 0
        self.indices = {}
        self.jobs = []
        self.base_params = dict(self.network.named_parameters()) # keep this for updates

        mlp_names = []
        out_dims = []
        splits_list = []

        for name, param in self.network.named_parameters():
            if not self.should_adapt(name):
                continue

            dims = list(param.shape)
            out_dim = sum(dims)
            
            # don't make mlp for smaller parameters
            if out_dim < 512:
                continue
            
            print(f"MLP updating param: {name} with shape: {dims}")
            mlp_idx = self.total_mlps
            self.indices[name] = mlp_idx
            mlp_names.append(name)
            out_dims.append(out_dim)
            splits_list.append(tuple(param.shape))
            self.total_mlps += 1

        # build one batched MLP instead of all individual MLP modules
        hyper_hidden_width = 1024
        self.batched_mlp = BatchedMLP(
            M=self.total_mlps,
            in_dim=in_dim,
            hidden_dim=hyper_hidden_width,
            out_dims=out_dims,
            num_hidden_layers=self.num_mlp_layers,
            activation=F.gelu,
            device=device,
            dtype=torch.float32,
        ).to(device)

        for name, mlp_idx in self.indices.items():
            base = self.base_params[name]
            splits = tuple(base.shape)
            out_dim = sum(splits)
            self.jobs.append((name, mlp_idx, splits, base, out_dim))

    def should_adapt(self, name):
        # parameters that don't depend on state shouldn't get updates
        # time embedding parameters
        if name.startswith("time"): return False
        if name.startswith("mlp"): return False
        return True

    def make_vectors(self, mlp_out, splits):
        return torch.split(mlp_out, splits, dim=1)
    

    
    def broadcasting(self, vecs):
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
 
    def make_update(self, mlp_out, splits):
        v = self.make_vectors(mlp_out, splits)
        return self.broadcasting(v)
    
    def forward(self, u):
        # get mlp outputs
        Y = self.batched_mlp(u)
        new_params = {}

        # build r1 updates
        for (name, mlp_idx, splits, base, out_dim) in self.jobs:
            mlp_out = Y[mlp_idx, :, :out_dim]
            update = self.make_update(mlp_out, splits)
            new_params[name] = base.unsqueeze(0) + update

        return new_params


class BatchedMLP(nn.Module):
    
    # a stack of M independent MLPs with identical architecture up to the output,
    # same in_dim, hidden_dim, num_hidden_layers for all MLPs
    # to account for different out dimensions, give all mlps the same maximum output width, then slice up until their corresponding desired output

    def __init__(self, M: int, in_dim: int, hidden_dim: int, out_dims: list[int], num_hidden_layers: int,
                 activation=F.gelu, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dims = torch.tensor(out_dims, device=device)
        self.max_out = int(max(out_dims))
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation

        self.W_in = nn.Parameter(torch.empty(M, hidden_dim, in_dim, device=device, dtype=dtype))
        self.b_in = nn.Parameter(torch.empty(M, hidden_dim, device=device, dtype=dtype))

        L = num_hidden_layers
        self.W_h = nn.Parameter(torch.empty(L, M, hidden_dim, hidden_dim, device=device, dtype=dtype))
        self.b_h = nn.Parameter(torch.empty(L, M, hidden_dim, device=device, dtype=dtype))

        self.W_out = nn.Parameter(torch.empty(M, self.max_out, hidden_dim, device=device, dtype=dtype))
        self.b_out = nn.Parameter(torch.empty(M, self.max_out, device=device, dtype=dtype))

        nn.init.xavier_normal_(self.W_in)
        nn.init.zeros_(self.b_in)
        for k in range(L):
            nn.init.xavier_normal_(self.W_h[k])
            nn.init.zeros_(self.b_h[k])
        nn.init.xavier_normal_(self.W_out)
        nn.init.zeros_(self.b_out)


    def bmm_linear(self, X_mb_i, W_m_o_i, b_m_o):
        # x is (M, B, in), W is (M, out, in), b is (M, out)
        return torch.bmm(X_mb_i, W_m_o_i.transpose(1, 2)) + b_m_o[:, None, :]

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B = x.shape[0]

        # expand input across M
        input = x.unsqueeze(0).expand(self.M, B, self.in_dim)

        hidden_layer = self.bmm_linear(input, self.W_in, self.b_in)
        hidden_layer = self.activation(hidden_layer)

        for k in range(self.num_hidden_layers):
            hidden_layer = self.bmm_linear(hidden_layer, self.W_h[k], self.b_h[k])
            hidden_layer = self.activation(hidden_layer)

        out = self.bmm_linear(hidden_layer, self.W_out, self.b_out)
        return out