import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
import string

#mlp class to be used by hypernet
class MLP_net_variable(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, activation=F.gelu, use_act = True, use_dropout=True):
        super().__init__()
        self.linear_in = nn.Linear(in_dim, hidden_dim)
        torch.nn.init.xavier_normal_(self.linear_in.weight)
        self.activation = activation
        self.layers_1 = nn.ModuleList()
        if use_dropout:
            self.drop = nn.ModuleList()
        self.use_drop = use_dropout
        self.num_layers = num_layers
        self.use_act = use_act
        for i in range(0,num_layers): 
            self.layers_1.append(nn.Linear(hidden_dim, hidden_dim))
            torch.nn.init.xavier_normal_(self.layers_1[i].weight)
            if use_dropout:
                self.drop.append(nn.Dropout(p=0.5))
        self.linear_out = nn.Linear(hidden_dim, out_dim)
        torch.nn.init.xavier_normal_(self.linear_out.weight)

    def forward(self, x):
        x = self.activation(self.linear_in(x))
        x_0 = x
        for i in range(0,self.num_layers):
            x = self.activation(self.layers_1[i](x)) 
            if self.use_drop:
                x = self.drop[i](x)
            # x = x + x_0
        x = self.linear_out(x)
        if self.use_act:
            x = self.activation(x)
        return x

class HyperNetwork(nn.Module):
    def __init__(self, num_mlp_layers, in_dim, hyper_hidden_scale, rank, network, device):
        super().__init__()
        self.network = network.to(device)
        self.device = device
        self.names = []
        self.shapes = []
        self.is_complex = []
        #self.hyper_layers = nn.ModuleList()
        self.indices = {}
        self.num_mlp_layers = num_mlp_layers
        self.hyper_hidden_scale = hyper_hidden_scale
        self.rank = rank

        # new version - one big mlp, split by param shapes and perform same rank 1 updates to each parameter
        self.param_splits = {}   # name : { rank_idx → list[(start,end), ...] }
        out_dim = 0

        for i in range(self.rank):
            for name, param in self.network.named_parameters():
                if not self.should_adapt(name):
                    continue
                #print(f"Param: {name}, param shape: {param.shape}")
                dims = list(param.shape)
                size = sum(dims)
                if param.is_complex():
                    size *= 2

                if name not in self.param_splits:
                    self.param_splits[name] = {}

                splits = []
                start = out_dim
                if param.is_complex():
                    cur = start
                    for d in dims:
                        splits.append((cur, cur + d))
                        cur += d
                    for d in dims:
                        splits.append((cur, cur + d))
                        cur += d
                else:
                    cur = start
                    for d in dims:
                        splits.append((cur, cur + d))
                        cur += d

                self.param_splits[name][i] = splits
                out_dim += size

        hyper_hidden_width = int(out_dim * hyper_hidden_scale)
        #hyper_hidden_width = min(hyper_hidden_width, 4096) # still need to test what the right mlp size is
        print(f"in_dim: {in_dim}, hidden_width: {hyper_hidden_width}, out_dim: {out_dim}")
        self.mlp = MLP_net_variable(in_dim, out_dim, hyper_hidden_width, self.num_mlp_layers, activation=F.gelu, use_act=False, use_dropout=False).to(device)

    def should_adapt(self, name):
        # Always skip huge attention weights
        if "attn.fn.to_qkv.weight" in name: return False
        if "attn.fn.to_out.weight" in name: return False

        # Start with time embedding projections + norm affine + small I/O
        if "mlp_layers.1." in name: return True
        if "groupnorm" in name.lower(): return True
        if "block2_layers.0." in name: return True   # groupnorm inside block2_layers
        if "attn.fn.group_norm" in name: return True
        if name.startswith("final_conv_layers.0."): return True
        if name.startswith("final_conv_layers.2.") or name.startswith("final_conv_layers.4."): return True
        if name.startswith("first_conv."): return True

        return False

    def make_vectors(self, mlp_out, splits):
        #splits = self.param_splits[name]  # list of (start, end)
        return [mlp_out[:, start:end] for (start, end) in splits]
    
    #complex case where there are twice as many vectors (real and im)
    def split_complex(self, mlp_out, param_shape):
        real, imag = mlp_out.chunk(2,dim=1)
        update_real = self.make_vectors(real, param_shape)
        update_imag = self.make_vectors(imag, param_shape)
        return update_real, update_imag

    def broadcasting(self, vecs):
        n = len(vecs)
        if n == 0:
            raise ValueError("broadcasting(): empty vecs")
        if n == 1:
            return vecs[0]

        letters = [ch for ch in string.ascii_lowercase if ch != "b"]
        if n > len(letters):
            raise ValueError(f"broadcasting(): too many dims ({n}) for einsum indexing")

        dim_syms = letters[:n]

        inputs = [f"b{sym}" for sym in dim_syms]
        out = "b" + "".join(dim_syms)
        eq = ",".join(inputs) + "->" + out

        return torch.einsum(eq, *vecs)

    def make_update(self, flat, splits, param):
        if param.is_complex():
            v = self.make_vectors(flat, splits)
            v_real, v_imag = v[:(len(v)//2)], v[(len(v)//2):]
            upd_r = self.broadcasting(v_real)
            upd_i = self.broadcasting(v_imag)
            return torch.complex(upd_r, upd_i)
        else:
            v = self.make_vectors(flat, splits)
            return self.broadcasting(v)

    #for each param make rank number of updates and add to each parameter
    #returns dictionary of updated parameters to be used in functional_call
    def forward(self, u_0):
        if u_0.dim() == 1:
            u_0 = u_0.unsqueeze(0)
        B = u_0.shape[0]
        new_params = OrderedDict()
        mlp_out = self.mlp(u_0)
        for name, param in self.network.named_parameters():              
            if name not in self.param_splits:
                new_params[name] = param.unsqueeze(0).expand(B, *param.shape)
                continue
            for i in range(self.rank):
                splits = self.param_splits[name][i]
                update = self.make_update(mlp_out, splits, param)
                if i == 0:
                    new_params[name] = param.unsqueeze(0) + update
                else:
                    new_params[name] += update

        return new_params