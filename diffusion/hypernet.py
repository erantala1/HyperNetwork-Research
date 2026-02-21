import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torch.func import vmap
import numpy as np
import string

#mlp class to be used by hypernet
class MLP(nn.Module):
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
    def __init__(self, num_mlp_layers, in_dim, hyper_hidden_scale, one_mlp, network, device):
        super().__init__()
        '''
        hyper_layers = [ MLP for param 1, MLP for param 2, ... ]
        indices dict = {main network parameter name : corresponding mlp index in hyper_layers}
        param_splits = [start index for vector 1, end index for vector 1/start index for vector 2....]
                       (slice vectors from mlp outputs to make r1 outerproduct update)
        '''
        self.network = network.to(device)
        self.device = device
        self.hyper_layers = nn.ModuleList()
        self.indices = {}
        self.num_mlp_layers = num_mlp_layers
        self.hyper_hidden_scale = hyper_hidden_scale
        self.one_mlp = one_mlp
        
        # one big mlp version, split by param shapes and perform same rank 1 updates to each parameter
        self.param_splits = {}
        out_dim = 0
        
        if self.one_mlp:
            for name, param in self.network.named_parameters():
                if not self.should_adapt(name):
                    continue
                print(f"Param: {name}, param shape: {param.shape}")
                dims = list(param.shape)
                size = sum(dims)
   
                if name not in self.param_splits:
                    self.param_splits[name] = {}

                splits = []
                start = out_dim
                cur = start
                for d in dims:
                    splits.append((cur, cur + d))
                    cur += d

                self.param_splits[name] = splits
                out_dim += size

            hyper_hidden_width = int(out_dim * hyper_hidden_scale)
            #hyper_hidden_width = min(hyper_hidden_width, 2048) # still need to test what the right mlp size is
            #hyper_hidden_width = 2 * out_dim
            hyper_hidden_width = 1024
            print(f"in_dim: {in_dim}, hidden_width: {hyper_hidden_width}, out_dim: {out_dim}")
            self.mlp = MLP(in_dim, out_dim, hyper_hidden_width, self.num_mlp_layers, activation=F.gelu, use_act=False, use_dropout=False).to(device)
        

        #each mlp's output dimension will be the sum of the outer network parameter's shape
        else:
            self.total_mlps = 0
            for name, param in self.network.named_parameters():
                if not self.should_adapt(name):
                    #print(f"Non applicable parameter: {name}")
                    continue
                dims = list(param.shape)
                out_dim = sum(dims)
                
                # don't make mlp for smaller parameters
                if out_dim < 512:
                    #print(f"Skipping {name}, out_dim : {out_dim}")
                    continue
                
                # precompute split indices for this param to speed up updating process
                splits = []
                start = 0
                for d in dims:
                    end = start + d
                    splits.append((start, end))
                    start = end
                self.param_splits[name] = splits
                
                temp = int(out_dim * hyper_hidden_scale)
                if temp % 8 != 0: # hidden width multiple of 8 for performance
                    temp = (temp // 8) * 8
                hyper_hidden_width = max(temp, 64)
                mlp = MLP(in_dim, out_dim, hyper_hidden_width, self.num_mlp_layers, activation=F.gelu, use_act = False, use_dropout=False).to(device)
                self.indices[name] = len(self.hyper_layers)
                self.hyper_layers.append(mlp)
                #print(f"Name:{name}, in_dim: {in_dim}, hidden_width: {hyper_hidden_width}, out_dim: {out_dim}")
                self.total_mlps += 1

            self.streams = [torch.cuda.Stream() for _ in range(len(self.hyper_layers))]
            self.jobs = []
            self.base_params = dict(self.network.named_parameters()) # using this to avoid building full dict of params for functional call
            # can use functional call with strict=False, faster runtime
            for name, mlp_idx in self.indices.items():
                self.jobs.append((name, mlp_idx, self.param_splits[name], self.base_params[name]))
            
    
    def should_adapt(self, name):
        # parameters that don't depend on state shouldn't get updates
        # time embedding parameters
        if name.startswith("time"): return False
        if name.startswith("mlp"): return False
        return True

    def make_vectors(self, mlp_out, splits):
        return [mlp_out[:, start:end] for (start, end) in splits]
    
    '''
    def broadcasting(self, vecs):
        n = len(vecs)
        if n == 1:
            return vecs[0]
        letters = [ch for ch in string.ascii_lowercase if ch != "b"]
        dim_syms = letters[:n]

        inputs = [f"b{sym}" for sym in dim_syms]
        out = "b" + "".join(dim_syms)
        eq = ",".join(inputs) + "->" + out

        return torch.einsum(eq, *vecs)
    '''

    # changing to this was faster than einsum (less kernel launches)
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


    def forward_multiple_mlp(self, u):
        if u.dim() == 1:
            u = u.unsqueeze(0)
        outs = [None] * len(self.hyper_layers)
        new_params = {}
        default_stream = torch.cuda.current_stream()
        
        # wait on default stream to finish prepping anything for mlps
        for s in self.streams:
            s.wait_stream(default_stream)

        # mlp forward pass on designated stream, give chance to parallelize?
        for (name, mlp_idx, splits, base) in self.jobs:
            s = self.streams[mlp_idx]
            with torch.cuda.stream(s):
                outs[mlp_idx] = self.hyper_layers[mlp_idx](u)

        # wait on mlps to be done before doing updates
        for s in self.streams:
            default_stream.wait_stream(s)

        new_params = {}
        for (name, mlp_idx, splits, base) in self.jobs:
            update = self.make_update(outs[mlp_idx], splits)
            new_params[name] = base.unsqueeze(0) + update
        
        return new_params
    

    def forward_one_mlp(self, u):
        if u.dim() == 1:
            u = u.unsqueeze(0)
        B = u.shape[0]
        new_params = OrderedDict()
        mlp_out = self.mlp(u)
        for name, param in self.network.named_parameters():            
            if name not in self.param_splits:
                new_params[name] = param.unsqueeze(0).expand(B, *param.shape)
            else:
                splits = self.param_splits[name]
                update = self.make_update(mlp_out, splits)

                new_params[name] = param.unsqueeze(0) + update

        return new_params
    

    def forward(self, u):
        if self.one_mlp:
            return self.forward_one_mlp(u)
        else:
            return self.forward_multiple_mlp(u)