import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
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
                self.drop.append(nn.Dropout(p=0.9))
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

#main hypernetwork class
#creates mlp, each with specified number of layers and hidden dim, for each param of outer network
#returns dictionary with updates for main network's params
class HyperNetwork(nn.Module):
    def __init__(self, num_mlp_layers, in_dim, hyper_hidden_scale, which_params, rank, network, device):
        super().__init__()
        self.fno = network.to(device)
        self.device = device
        self.names = []
        self.shapes = []
        self.is_complex = []
        #self.hyper_layers = nn.ModuleList()
        self.indices = {}
        self.num_mlp_layers = num_mlp_layers
        self.hyper_hidden_scale = hyper_hidden_scale
        self.rank = rank
        self.which_params = which_params

        # new version - one big mlp, split by param shapes and perform same rank 1 updates to each parameter
        self.param_splits = {}   # name : { rank_idx → list[(start,end), ...] }
        out_dim = 0

        for i in range(self.rank):
            for name, param in self.fno.named_parameters():
                if name not in self.which_params:
                    continue

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
        hyper_hidden_width = max(hyper_hidden_width, 1)
        print(f"in_dim{in_dim}, hidden_width: {hyper_hidden_width}, out_dim: {out_dim}")
        self.mlp = MLP_net_variable(in_dim, out_dim, hyper_hidden_width, self.num_mlp_layers, activation=F.gelu, use_act=False, use_dropout=False).to(device)

        # old version - mlp for each param
        '''
        #each mlp's output dimension will be the sum of the outer network parameter's shape
        for i in range(self.rank):
            for name, param in self.fno.named_parameters():
                if name not in self.which_params:
                    continue

                self.names.append(name)
                self.shapes.append(param.shape)
                self.is_complex.append(param.is_complex())
                dims = list(param.shape)
                out_dim = sum(dims)
                # precompute split indices for this param to speed up updating process
                splits = []
                start = 0
                for d in dims:
                    end = start + d
                    splits.append((start, end))
                    start = end
                self.param_splits[name] = splits
                if param.is_complex():
                    out_dim *= 2
                hyper_hidden_width = int(out_dim * hyper_hidden_scale)
                if hyper_hidden_width == 0:
                    hyper_hidden_width = 1
                mlp = MLP_net_variable(in_dim, out_dim, hyper_hidden_width, self.num_mlp_layers, activation=F.gelu, use_act = False, use_dropout=False).to(device)
                self.indices[name+f"{i}"] = len(self.hyper_layers)
                self.hyper_layers.append(mlp)
                print(f"Name:{name}")
        '''
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
        # vecs = [v1] or [v1, v2] or [v1, v2, v3]
        if len(vecs) == 1:
            return vecs[0]

        elif len(vecs) == 2:
            v1, v2 = vecs
            return torch.einsum("bi, bj -> bij", v1, v2)

        elif len(vecs) == 3:
            v1, v2, v3 = vecs
            return torch.einsum("bi, bj, bk -> b i j k", v1, v2, v3)

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
        B = u_0.shape[0]
        vec_u = u_0.reshape(B, -1)
        new_params = OrderedDict()
        #all_flats = [mlp(vec_u) for mlp in self.hyper_layers] # pre-compute all mlp outputs successively for kernel latency
        mlp_out = self.mlp(vec_u)
        for name, param in self.fno.named_parameters():              

            if name not in self.which_params:
                new_params[name] = param.unsqueeze(0).expand(B, *param.shape)
            else:
                for i in range(self.rank):
                    #flat = all_flats[self.indices[name + f"{i}"]]
                    #update = self.make_update(flat, param, name)
                    splits = self.param_splits[name][i]
                    update = self.make_update(mlp_out, splits, param)
                    if i == 0:
                        new_params[name] = param.unsqueeze(0) + update
                    else:
                        new_params[name] += update

        return new_params
    