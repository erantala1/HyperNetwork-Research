import json
import torch
import io
import os
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
        state_dict = torch.load(buffer, map_location=device, weights_only=True)
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

def createPoolPyramidConditioner(*, C, sizes=(32, 16, 8), device="cpu"):
    return PoolPyramidConditioner(
        C=C,
        sizes=sizes,
    ).to(device)


def load_hypernet(hnet_path, model, device):
    # multiple mlp version
    num_mlp_layers = 10
    one_mlp = False
    hyper_hidden_scale = 1.0
    H, W, C_state = 256, 256, 1
    cond = PoolPyramidConditioner(C=C_state, sizes=(32, 16, 8)).to(device)
    in_dim =  cond.out_dim

    hnet = HyperNetwork(num_mlp_layers, in_dim, hyper_hidden_scale, one_mlp, model, device).to(device)
    hnet.load_state_dict(torch.load(hnet_path, map_location=device, weights_only=True))
    hnet.eval()
    return hnet, cond

def resume_training(output_dir, epoch_to_resume, model, optimizer, device, use_hypernet, hypernet=None, optimizer_hyper=None):
    # model ckpt
    opt_ckpt_path = os.path.join(
        output_dir,
        f"optimizer_lr_epoch_gs_{epoch_to_resume}.pt"
    )
    model_ckpt = os.path.join(output_dir, f"model_epoch_{epoch_to_resume}.pt")
    model = loadModel(model_ckpt, device)
    model.train()

    print(f"Loading optimizer state from {opt_ckpt_path}")
    opt_ckpt = torch.load(opt_ckpt_path, map_location=device)

    optimizer.load_state_dict(opt_ckpt["optimizer"])
    start_epoch = opt_ckpt["epoch"] + 1
    global_step = opt_ckpt["global_step"]

    for g in optimizer.param_groups:
        g["lr"] = opt_ckpt["lr"]


    if use_hypernet:
        hyper_ckpt = os.path.join(
            output_dir,
            f"hypernet_epoch_{epoch_to_resume}.pt"
        )

        print(f"Loading hypernet weights from {hyper_ckpt}")
        hypernet.load_state_dict(torch.load(hyper_ckpt, map_location=device))
        hypernet.to(device)
        hypernet.train()

        optimizer_hyper.load_state_dict(opt_ckpt["optimizer_hyper"])

        for g in optimizer_hyper.param_groups:
            g["lr"] = opt_ckpt["lr_hyper"]
            
        return model, start_epoch, global_step, optimizer, hypernet, optimizer_hyper
    else:
        return model, start_epoch, global_step, optimizer