import torch
import torch.nn as nn
from torch.func import functional_call
import timeit
from utils import createModel
from preprocess import PoolPyramidConditioner
from hypernet import HyperNetwork

# wrapper for pairing hypernet with unet
# use torch.compile on step.make_params
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



def one_forward_pass_step_call(step, condition_x_t):
    torch.cuda.nvtx.range_push("hypernet forward")
    step.make_params(condition_x_t)
    torch.cuda.nvtx.range_pop()
    return


def main():
    device = torch.device("cuda")
    torch.cuda.set_device(0)

    data = torch.randn(1, 256, 256, device=device)
    t = torch.tensor(0.5, device=device, dtype=torch.float32)
    hyperparameters = {
        "data_shape": (2, 256, 256),
        "dim_mults": [2, 4, 4, 8, 16],
        "hidden_size": 8,
        "heads": 16,
        "dim_head": 32,
        "dropout_rate": 0.1,
        "num_res_blocks": 2,
        "attn_resolutions": [32, 16, 8],
    }

    use_hypernet = True

    model = createModel(**hyperparameters).to(device).float().eval()
    
    if use_hypernet:
        preprocess_step = PoolPyramidConditioner(
            C=1, sizes=(32, 16, 8)
        ).to(device).eval()

        hypernet = HyperNetwork(
            num_mlp_layers=10,
            in_dim=preprocess_step.out_dim,
            hyper_hidden_scale=1.0,
            one_mlp=False,
            network=model,
            device=device,
        ).to(device).eval()
        
    else:
        hypernet, preprocess_step = None, None
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    step = Step(model, hypernet, preprocess_step).eval().to(device)
    #step.make_params = torch.compile(step.make_params)
    #step = torch.compile(step)

    # warmup
    for _ in range(20):
        one_forward_pass_step_call(step, data)


    # timeit measurement
    def timed_forward():
        one_forward_pass_step_call(step, data)

    total_time = timeit.timeit(timed_forward, number=200)

    avg_time = total_time / 200

    print(f"\nTotal time for 200 forward passes: {total_time:.6f} seconds")
    print(f"Average time per forward: {avg_time*1000:.3f} ms")


if __name__ == "__main__":
    main()