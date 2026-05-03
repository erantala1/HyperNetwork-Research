import json
import torch
import io
import os
from models import *
from tformer import TransformerModel
from hypernet import HyperNetwork

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

def load_hypernet(hnet_path, model, device):
    num_mlp_layers = 2
    hidden_dim = 512
    hnet = HyperNetwork(hidden_dim, num_mlp_layers, model, device, rank=2).to(device)
    hnet.load_state_dict(torch.load(hnet_path, map_location=device, weights_only=True))
    hnet.eval()
    return hnet

def resume_training(output_dir, epoch_to_resume, device, use_hypernet,hypernet=None):
    opt_ckpt_path = os.path.join(output_dir, f"optimizer_lr_epoch_gs_{epoch_to_resume}.pt")
    model_ckpt = os.path.join(output_dir, f"model_epoch_{epoch_to_resume}.pt")

    print(f"Loading model weights from {model_ckpt}")
    model = loadModel(model_ckpt, device)
    model.train()

    print(f"Loading optimizer state from {opt_ckpt_path}")
    opt_ckpt = torch.load(opt_ckpt_path, map_location=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_ckpt["lr"])
    optimizer.load_state_dict(opt_ckpt["optimizer"])

    start_epoch = opt_ckpt["epoch"] + 1
    global_step = opt_ckpt["global_step"]

    for g in optimizer.param_groups:
        g["lr"] = opt_ckpt["lr"]

    if use_hypernet:
        hyper_ckpt = os.path.join(output_dir, f"hypernet_epoch_{epoch_to_resume}.pt")
        print(f"Loading hypernet weights from {hyper_ckpt}")
        hypernet = load_hypernet(hyper_ckpt, model, device)
        hypernet.train()

        optimizer_hyper = torch.optim.Adam(hypernet.parameters(), lr=opt_ckpt["lr_hyper"])
        optimizer_hyper.load_state_dict(opt_ckpt["optimizer_hyper"])

        for g in optimizer_hyper.param_groups:
            g["lr"] = opt_ckpt["lr_hyper"]

        return model, start_epoch, global_step, optimizer, hypernet, optimizer_hyper

    return model, start_epoch, global_step, optimizer