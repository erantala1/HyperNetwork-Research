import os
import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
from dataloaders import *
from losses import *
from models import *
from sde import *
from utils import *
from hypernet import *

# ==============================================================================
# CONFIG
# ==============================================================================
output_dir = "/glade/derecho/scratch/erantala/project_runs/model_chkpts/diffusion/baseline_test"
os.makedirs(output_dir, exist_ok=True)

data_dir = "/glade/derecho/scratch/cainslie/beta-channel-turbulence/data_low_res/data_lowres"
data = load_turbulence_data(data_dir, stop_idx=12799, normalize=True)

batch_size = 2
dt = 1
num_epochs = 100
save_every = 5
learning_rate = 1e-4
num_Taus = 4

# fraction of timeline used for training (rest is validation)
train_frac = 0.80

hyperparameters = {
    "data_shape": (2, 256, 256),
    #"data_shape": (1, 256, 256), # test for no concatentation, only hypernetwork conditioning
    "dim_mults": [2, 4, 4, 8, 16],
    "hidden_size": 16,
    "heads": 32,
    "dim_head": 64,
    "dropout_rate": 0.1,
    "num_res_blocks": 2,
    "attn_resolutions": [32, 16, 8],
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# SPLIT DATA INTO TRAIN / VAL (by time/index order)
# ==============================================================================
all_indices = sorted(list(data.keys()))
cut = int(len(all_indices) * train_frac)

train_indices = all_indices[:cut]
val_indices = all_indices[cut:]

train_data = {k: data[k] for k in train_indices}
val_data = {k: data[k] for k in val_indices}

print(f"Total frames: {len(all_indices)}")
print(f"Train frames: {len(train_indices)} (idx {train_indices[0]}..{train_indices[-1]})")
print(f"Val frames:   {len(val_indices)} (idx {val_indices[0]}..{val_indices[-1]})")

# ==============================================================================
# MODEL + HYPERNET + SDE
# ==============================================================================
model = createModel(**hyperparameters).to(device).float()
vpsde = VPSDE(beta_min=0.01, beta_max=55.0, T=1.0, schedule_type="power", power=6).to(device)
noise_std = 0.01

use_hypernet = False # set to false when just using UNET
if use_hypernet:
    C_state = 1
    preprocess_step = PoolPyramidConditioner(C=C_state, sizes=(32, 16, 8), add_moments=True).to(device)
    rank = 1
    num_mlp_layers = 16
    in_dim = preprocess_step.out_dim
    hyper_hidden_scale = 0.3
    hypernet = HyperNetwork(num_mlp_layers, in_dim, hyper_hidden_scale, rank, model, device).to(device)

# ==============================================================================
# LOSS
# ==============================================================================
def score_loss_fn(model, t, x, y, params=None):
    return continuousScoreLoss(model, t, x, y, vpsde, noise_std, params=params)

def batch_loss(model, taus, input_batch, x_dt):
    B = input_batch.shape[0]
    losses = []
    for b in range(B):
        x = input_batch[b]
        y = x_dt[b]
        tau_losses = []
        for k in range(taus.shape[0]):
            t = taus[k]
            tau_losses.append(score_loss_fn(model, t, x, y))
        losses.append(torch.stack(tau_losses))
    return torch.stack(losses).mean()

def batch_loss_hyper(model, hypernet, preprocess_step, taus, input_batch, x_dt):
    B = input_batch.shape[0]
    # x_t_batch: (B,1,H,W)
    x_t_batch = input_batch[:, 1:2]
    # preprocess returns (B, in_dim)
    z = preprocess_step(x_t_batch)
    params_batched = hypernet(z)
    losses = []
    for b in range(B):
        x = input_batch[b]
        y = x_dt[b]
        x_t = x[1:2]
        z = preprocess_step(x_t)
        params = {k: v[b] for k, v in params_batched.items()}
        tau_losses = []
        for k in range(taus.shape[0]):
            t = taus[k]
            tau_losses.append(score_loss_fn(model, t, x, y, params))
        losses.append(torch.stack(tau_losses))
    return torch.stack(losses).mean()

# ==============================================================================
# OPTIMIZER + SCHEDULE
# ==============================================================================
initial_lr = learning_rate
decay_steps = 120000
end_lr = 1e-5

def cosine_decay(step, initial_lr, end_lr, decay_steps):
    if step >= decay_steps:
        return end_lr
    progress = step / decay_steps
    return end_lr + (initial_lr - end_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

gradient_clip_norm = 0.25
optimizer = optim.AdamW(model.parameters(), lr=initial_lr)

if use_hypernet:
    initial_lr_hyper = 1e-5
    end_hyper_lr = 1e-6
    optimizer_hyper = optim.AdamW(hypernet.parameters(), lr=initial_lr_hyper)

def update_step(model, optimizer, step, taus, input_batch, x_dt):
    lr = cosine_decay(step, initial_lr, end_lr, decay_steps)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    optimizer.zero_grad(set_to_none=True)
    loss = batch_loss(model, taus, input_batch, x_dt)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    return loss.detach()

def hyper_update_step(model, hypernet, preprocess_step, optimizer, optimizer_hyper, step, taus, input_batch, x_dt):
    lr = cosine_decay(step, initial_lr, end_lr, decay_steps)
    lr_hyper = cosine_decay(step, initial_lr_hyper, end_hyper_lr, decay_steps)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    for pg in optimizer_hyper.param_groups:
        pg["lr"] = lr_hyper
    optimizer.zero_grad(set_to_none=True)
    optimizer_hyper.zero_grad(set_to_none=True)
    loss = batch_loss_hyper(model, hypernet, preprocess_step, taus, input_batch, x_dt)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    torch.nn.utils.clip_grad_norm_(hypernet.parameters(), gradient_clip_norm)
    optimizer.step()
    optimizer_hyper.step()
    return loss.detach()

@torch.no_grad()
def run_validation(model, val_data, hypernet=None, preprocess_step=None):
    model.eval()
    val_loader = create_turbulence_dataloader(val_data, batch_size, dt=dt, shuffle=False)
    num_samples_val = len(val_data) - dt
    num_batches_val = num_samples_val // batch_size
    losses = []
    for _ in range(num_batches_val):
        try:
            x_t, x_nt = next(val_loader)
            if x_t.shape[0] != batch_size:
                continue
            x_t = x_t.to(device=device, dtype=torch.float32)
            x_nt = x_nt.to(device=device, dtype=torch.float32)
            x_dt = x_nt - x_t
            input_batch = torch.cat([x_dt, x_t], dim=1)
            taus = 0.01 + (1.0 - 0.01) * torch.rand(num_Taus, device=device, dtype=torch.float32)
            if use_hypernet:
                loss = batch_loss_hyper(model, hypernet, preprocess_step, taus, input_batch, x_dt)
            else:
                loss = batch_loss(model, taus, input_batch, x_dt)
            losses.append(loss.item())
        except StopIteration:
            break

    model.train()
    return float(torch.tensor(losses).mean()) if len(losses) else float("nan")

# ==============================================================================
# WANDB
# ==============================================================================
run = wandb.init(
    entity="erantala-university-of-california",
    project="HyperNet_Diffusion",
    config={
        "architecture": "Baseline UNet diffusion (turbulence)",
        "hyperparameters": hyperparameters,
        "batch_size": batch_size,
        "dt": dt,
        "num_epochs": num_epochs,
        "num_Taus": num_Taus,
        "save_every": save_every,
        "save_path": output_dir,
        "train_frac": train_frac,
        "use_hypernet": use_hypernet,
        "concatenation_cond": True,
        #"Hyper in_dim": 1346,
        #"Hyper hidden_dim": 6049,
        #"Hyper out_dim": 20165
    },
)
wandb.define_metric("epoch")
wandb.define_metric("epoch/*", step_metric="epoch")

# ==============================================================================
# TRAIN LOOP
# ==============================================================================
num_samples_train = len(train_data) - dt
num_batches_train = num_samples_train // batch_size
print(f"Starting training: {num_epochs} epochs with {num_batches_train} train batches/epoch")

global_step = 0

for epoch in range(1, num_epochs + 1):
    model.train()
    train_loader = create_turbulence_dataloader(train_data, batch_size, dt=dt, shuffle=True)

    progress_bar = tqdm(total=num_batches_train, desc=f"Epoch {epoch}/{num_epochs}")
    epoch_losses = []

    for _ in range(num_batches_train):
        taus = 0.01 + (1.0 - 0.01) * torch.rand(num_Taus, device=device, dtype=torch.float32)

        try:
            x_t, x_nt = next(train_loader)
            if x_t.shape[0] != batch_size:
                continue

            x_t = x_t.to(device=device, dtype=torch.float32)
            x_nt = x_nt.to(device=device, dtype=torch.float32)

            x_dt = x_nt - x_t
            
            input_batch = torch.cat([x_dt, x_t], dim=1)
            if use_hypernet:
                loss = hyper_update_step(model, hypernet, preprocess_step, optimizer, optimizer_hyper, global_step, taus, input_batch, x_dt)
            else:
                loss = update_step(model, optimizer, global_step, taus, input_batch, x_dt)
            
            epoch_losses.append(loss.item())
            global_step += 1

            progress_bar.set_postfix({"train_loss": f"{loss.item():.4f}"})
            progress_bar.update(1)

        except StopIteration:
            break

    progress_bar.close()
    train_loss = float(torch.tensor(epoch_losses).mean()) if len(epoch_losses) else float("nan")

    if use_hypernet:
        val_loss = run_validation(model, val_data, hypernet, preprocess_step)
    else:
        val_loss = run_validation(model, val_data)

    print(f"Epoch {epoch} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    run.log(
        {
            "epoch": epoch,
            "epoch/train_loss": train_loss,
            "epoch/val_loss": val_loss,
            "epoch/lr": optimizer.param_groups[0]["lr"],
        },
        step=epoch,
    )

    if epoch % save_every == 0:
        save_path = os.path.join(output_dir, f"model_epoch_{epoch}.pt")
        saveModel(save_path, hyperparameters, model)
        print(f"Model saved to {save_path}")
        if use_hypernet:
            torch.save(hypernet.state_dict(), os.path.join(output_dir, f"hypernet_epoch_{epoch}.pt"))

saveModel(os.path.join(output_dir, "model_final.pt"), hyperparameters, model)
if use_hypernet:
    torch.save(hypernet.state_dict(), os.path.join(output_dir, "hypernet_final.pt"))
print("Training complete")
