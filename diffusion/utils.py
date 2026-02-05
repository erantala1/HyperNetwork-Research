import json
import torch
import io
from models import UNet
from tformer import TransformerModel
from hypernet import HyperNetwork
from preprocess import PoolPyramidConditioner

def createModel(*,
               data_shape,
               dim_mults,
               hidden_size,
               heads,
               dim_head,
               dropout_rate,
               num_res_blocks,
               attn_resolutions):

    model = UNet(
        data_shape=data_shape,
        is_biggan=False,
        dim_mults=dim_mults,
        hidden_size=hidden_size,
        heads=heads,
        dim_head=dim_head,
        dropout_rate=dropout_rate,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
    )
    return model


def createTransformerModel(*,
                           num_blocks,
                           num_heads,
                           ff_widening_factor):

    model = TransformerModel(
        num_blocks=num_blocks,
        num_heads=num_heads,
        ff_widening_factor=ff_widening_factor,
    )
    return model

def saveModel(filename, hyperparams, model):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        torch.save(model.state_dict(), f)


def loadModel(filename, device="cpu"):
    with open(filename, "rb") as f:
        # Read hyperparameters
        first_line = f.readline()
        hyperparams = json.loads(first_line.decode())

        # Load remaining bytes as torch checkpoint
        remaining_data = f.read()
        buffer = io.BytesIO(remaining_data)

        model = createModel(**hyperparams)
        state_dict = torch.load(buffer, map_location=device)
        model.load_state_dict(state_dict)

        return model.to(device)

def saveTransformerModel(filename, hyperparams, model):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        torch.save(model.state_dict(), f)

def loadTransformerModel(filename):
    with open(filename, "rb") as f:
        # Read the first line (JSON hyperparameters)
        first_line = f.readline()
        hyperparams = json.loads(first_line.decode())
        
        # Read the rest of the file (torch.save data)
        import io
        remaining_data = f.read()
        buffer = io.BytesIO(remaining_data)
        
        generator = torch.Generator()
        generator.manual_seed(0)
        model = createTransformerModel(key=generator, **hyperparams)
        model.load_state_dict(torch.load(buffer, map_location='cpu', weights_only=False))
        return model


# Hypernet _________________________________________________

def createPoolPyramidConditioner(*, C, sizes=(32, 16, 8), add_moments=True, device="cpu"):
    return PoolPyramidConditioner(
        C=C,
        sizes=sizes,
        add_moments=add_moments,
    ).to(device)

def infer_mlp_sizes_from_state_dict(hnet_sd: dict):
    # Infer (in_dim, hidden_width, out_dim, n_linear_layers) from hypernet checkpoint.
    
    weight_keys = [k for k in hnet_sd.keys() if k.endswith(".weight")]
    bias_keys   = [k for k in hnet_sd.keys() if k.endswith(".bias")]

    # Keep only mlp.* keys (avoid other params)
    weight_keys = [k for k in weight_keys if k.startswith("mlp.")]
    bias_keys   = [k for k in bias_keys if k.startswith("mlp.")]

    if len(weight_keys) == 0:
        raise RuntimeError("Could not find mlp.*.weight keys in hypernet checkpoint; adjust inference logic.")

    # Sort keys by layer index if possible
    def layer_idx(k):
        # expected like "mlp.0.weight" or "mlp.layers.0.weight" etc.
        parts = k.split(".")
        for p in parts:
            if p.isdigit():
                return int(p)
        return 10**9

    weight_keys_sorted = sorted(weight_keys, key=layer_idx)

    first_w = hnet_sd[weight_keys_sorted[0]]
    last_w  = hnet_sd[weight_keys_sorted[-1]]

    hidden_width = int(first_w.shape[0])
    in_dim       = int(first_w.shape[1])
    out_dim      = int(last_w.shape[0])

    # Count linear layers as number of weight matrices
    n_linear = len(weight_keys_sorted)
    return in_dim, hidden_width, out_dim, n_linear


def load_hypernet(hnet_path: str, model, device: torch.device):
    # load HyperNetwork with hyper_hidden_scale inferred from checkpoint layer sizes.
    hnet_sd = torch.load(hnet_path, weights_only=True,map_location="cpu")
    in_dim_ckpt, hidden_width_ckpt, out_dim_ckpt, _ = infer_mlp_sizes_from_state_dict(hnet_sd)

    H, W, C_state = 256, 256, 1
    cond = PoolPyramidConditioner(C=C_state, sizes=(32, 16, 8), add_moments=True).to(device)
    if cond.out_dim != in_dim_ckpt:
        print(f"[warn] conditioner out_dim={cond.out_dim} != ckpt in_dim={in_dim_ckpt}. "
              f"If this is expected, adjust conditioner to match training.")
    in_dim = cond.out_dim
    hyper_hidden_scale = float(hidden_width_ckpt) / float(out_dim_ckpt)

    # hard coding this for now
    num_mlp_layers = 16
    rank = 1

    hnet = HyperNetwork(num_mlp_layers, in_dim, hyper_hidden_scale, rank, model, device).to(device)
    hnet.load_state_dict(torch.load(hnet_path, map_location=device))
    hnet.eval()
    return hnet, cond

