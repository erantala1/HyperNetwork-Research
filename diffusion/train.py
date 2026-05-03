import os
import json
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
from dataloaders import *
from losses import *
from models import *
from sde import *
from utils import *
from hypernet import *
from wrapper import *
from count_params import *

def parse_args():
    parser = argparse.ArgumentParser(description="Sweep-ready diffusion training script")

    # Paths
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="/glade/derecho/scratch/cainslie/beta-channel-turbulence/data_low_res/data_lowres")
    parser.add_argument("--entity", type=str, default="erantala-university-of-california")
    parser.add_argument("--project", type=str, default="HyperNet_Diffusion")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--group", type=str, default="derecho_sweep")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    # Data / training
    parser.add_argument("--start_idx", type=int, default=10000)
    parser.add_argument("--stop_idx", type=int, default=19999)
    parser.add_argument("--normalize", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dt", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--num_taus", type=int, default=4)
    parser.add_argument("--val_num_taus", type=int, default=32)
    parser.add_argument("--train_frac", type=float, default=0.95)
    parser.add_argument("--gradient_clip_norm", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1234)

    # Model hyperparameters
    parser.add_argument("--data_channels", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--dim_mults", type=str, default="2,4,4,8,8,16")
    parser.add_argument("--hidden_size", type=int, default=4)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--dim_head", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--attn_resolutions", type=str, default="16")

    # Optimization / noise
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--end_lr", type=float, default=1e-5)
    parser.add_argument("--noise_std", type=float, default=0.0)
    parser.add_argument("--beta_min", type=float, default=0.01)
    parser.add_argument("--beta_max", type=float, default=55.0)
    parser.add_argument("--sde_T", type=float, default=1.0)
    parser.add_argument("--schedule_type", type=str, default="power")
    parser.add_argument("--power", type=float, default=5.0)

    # Hypernetwork
    parser.add_argument("--use_hypernet", type=int, default=1)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--c_state", type=int, default=1)
    parser.add_argument("--hyper_hidden_width", type=int, default=512)
    parser.add_argument("--num_mlp_layers", type=int, default=2)
    parser.add_argument("--hyper_learning_rate", type=float, default=1e-4)
    parser.add_argument("--hyper_end_lr", type=float, default=1e-6)
    parser.add_argument("--decay_steps", type=int, default = 20000)

    # Resume
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--resume_epoch", type=int, default=0)

    return parser.parse_args()

def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def score_loss_fn(model, t, x, y, vpsde, noise_std, wrapper=None, params=None):
    return continuousScoreLoss(model, t, x, y, vpsde, noise_std, wrapper=wrapper, params=params)


def batch_loss(model, taus, input_batch, x_dt, vpsde, noise_std, randomness="different"):
    def single_loss(x, y, t):
        return score_loss_fn(model, t, x, y, vpsde, noise_std)

    per_tau = torch.vmap(
        lambda t, x, y: single_loss(x, y, t),
        in_dims=(0, None, None),
        randomness=randomness,
    )

    per_batch = torch.vmap(
        lambda x, y: per_tau(taus, x, y),
        in_dims=(0, 0),
        randomness=randomness,
    )

    tau_losses = per_batch(input_batch, x_dt)
    return tau_losses.mean()

def batch_loss_hyper(model, wrapper, taus, x_dt, x_t, vpsde, noise_std, randomness="different", noise=0.0):
    x_t = x_t + noise * torch.randn_like(x_t)
    params_batched = wrapper.make_params(x_t)
    input_batch = torch.cat([x_dt, x_t], dim=1)

    def single_loss(x, y, t, params):
        return score_loss_fn(model, t, x, y, vpsde, noise_std, wrapper=wrapper, params=params)

    per_tau = torch.vmap(
        lambda t, x, y, params: single_loss(x, y, t, params),
        in_dims=(0, None, None, None),
        randomness=randomness,
    )

    per_batch = torch.vmap(
        lambda x, y, params: per_tau(taus, x, y, params),
        in_dims=(0, 0, {k: 0 for k in params_batched}),
        randomness=randomness,
    )

    tau_losses = per_batch(input_batch, x_dt, params_batched)
    return tau_losses.mean()


def update_step(model, optimizer, taus, x_dt, x_t, vpsde, noise_std, gradient_clip_norm):
    optimizer.zero_grad(set_to_none=True)
    input_batch = torch.cat([x_dt, x_t], dim=1)
    loss = batch_loss(model, taus, input_batch, x_dt, vpsde, noise_std)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    return loss.detach()


def hyper_update_step(model, hypernet, wrapper, optimizer, optimizer_hyper, taus, x_dt, x_t,
                      vpsde, noise_std, gradient_clip_norm):
    optimizer.zero_grad(set_to_none=True)
    optimizer_hyper.zero_grad(set_to_none=True)
    loss = batch_loss_hyper(model, wrapper, taus, x_dt, x_t, vpsde, noise_std, randomness="different", noise=noise_std)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    torch.nn.utils.clip_grad_norm_(hypernet.parameters(), gradient_clip_norm)
    optimizer.step()
    optimizer_hyper.step()
    return loss.detach()

@torch.inference_mode()
def reverse_sde_sampler(model_state, vpsde, condition_x_t, delta_shape, num_steps, eps, device, wrapper=None, hypernet=None):
    use_hnet = (hypernet is not None)
    T = float(vpsde.T)
    time_steps = torch.linspace(T, eps, steps=num_steps, device=device, dtype=torch.float32)
    dt = (eps - T) / num_steps
    dt_tensor = torch.tensor(dt, device=device)
    y_t = vpsde.prior_sampling(delta_shape, device).to(device)

    if use_hnet:
        params = wrapper.make_params(condition_x_t)
        params = {k: v.squeeze(0) for k, v in params.items()}

    for i in range(num_steps - 1):
        t = time_steps[i]
        if use_hnet:
            model_input = torch.cat([y_t, condition_x_t], dim=0)
            #model_input = y_t
            score = wrapper(model_input, t, params) 
        else:
            model_input = torch.cat([y_t, condition_x_t], dim=0)
            score = model_state(t, model_input)

        beta_t = vpsde.beta(t)
        drift = -0.5 * beta_t * y_t - beta_t * score

        diffusion_coeff = torch.sqrt(beta_t)
        noise = torch.randn(y_t.shape, device=device, dtype=y_t.dtype)
        y_t = y_t + drift * dt + diffusion_coeff * torch.sqrt(-dt_tensor) * noise

    return y_t

@torch.no_grad()
def run_validation(model, val_data, batch_size, dt, device, vpsde, sde_eps, hypernet=None, wrapper=None,):
    model.eval()
    if hypernet is not None:
        hypernet.eval()

    val_loader = create_turbulence_dataloader(val_data, batch_size, dt=dt, shuffle=False)
    num_samples_val = len(val_data) - dt
    num_batches_val = max(0, num_samples_val // batch_size)

    losses = []
    progress_bar = tqdm(total=num_batches_val, desc=f"Sampling Validation")
    for _ in range(num_batches_val):
        try:
            x_t, x_nt = next(val_loader)
            if x_t.shape[0] != batch_size:
                continue
            x_t = x_t.to(device=device, dtype=torch.float32)
            x_nt = x_nt.to(device=device, dtype=torch.float32)
            x_dt = x_nt - x_t
            delta_shape = tuple(x_dt.shape[1:])

            pred_deltas = torch.vmap(
                lambda cond_x: reverse_sde_sampler(
                    model_state=model, vpsde=vpsde,
                    condition_x_t=cond_x,
                    delta_shape=delta_shape,
                    num_steps=1000, eps=sde_eps,
                    device=device, wrapper=wrapper,
                    hypernet=hypernet), in_dims=0,
                randomness="different",
            )(x_t)

            loss = torch.mean((pred_deltas - x_dt) ** 2)
            losses.append(loss.item())
            progress_bar.set_postfix({"sample_loss": f"{loss.item():.4f}"})
            progress_bar.update(1)
        except StopIteration:
            break
    progress_bar.close()
    model.train()
    if hypernet is not None:
        hypernet.train()
    return float(torch.tensor(losses).mean()) if losses else float("nan")

def main():
    args = parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    dim_mults = parse_int_list(args.dim_mults)
    attn_resolutions = parse_int_list(args.attn_resolutions)

    hyperparameters = {
        "data_shape": (args.data_channels, args.image_size, args.image_size),
        "dim_mults": dim_mults,
        "hidden_size": args.hidden_size,
        "heads": args.heads,
        "dim_head": args.dim_head,
        "dropout_rate": args.dropout_rate,
        "num_res_blocks": args.num_res_blocks,
        "attn_resolutions": attn_resolutions,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Resolved args:")
    print(json.dumps(vars(args), indent=2, sort_keys=True))

    data = load_turbulence_data(args.data_dir, start_idx=args.start_idx, stop_idx=args.stop_idx, normalize=bool(args.normalize))

    all_indices = sorted(list(data.keys()))
    cut = int(len(all_indices) * args.train_frac)
    train_indices = all_indices[:cut]
    val_indices = all_indices[cut:]

    train_data = {k: data[k] for k in train_indices}
    val_data = {k: data[k] for k in val_indices}

    print(f"Total frames: {len(all_indices)}")
    print(f"Train frames: {len(train_indices)} (idx {train_indices[0]}..{train_indices[-1]})")
    print(f"Val frames:   {len(val_indices)} (idx {val_indices[0]}..{val_indices[-1]})")

    model = createModel(**hyperparameters).to(device).float()
    vpsde = VPSDE(beta_min=args.beta_min,
        beta_max=args.beta_max,
        T=args.sde_T,
        schedule_type=args.schedule_type,
        power=args.power,
    ).to(device)

    use_hypernet = bool(args.use_hypernet)
    hypernet = None
    wrapper = None
    
    if use_hypernet:
        hypernet = HyperNetwork(args.hyper_hidden_width, args.num_mlp_layers, model, device, args.rank).to(device)
        wrapper = Step(model, hypernet)
        wrapper = torch.compile(wrapper)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.decay_steps,
        eta_min=args.end_lr,
    )
    
    optimizer_hyper = None
    scheduler_hyper = None
    
    if use_hypernet:
        optimizer_hyper = optim.Adam(hypernet.parameters(), lr=args.hyper_learning_rate)
        scheduler_hyper = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_hyper,
            T_max=args.decay_steps,
            eta_min=args.hyper_end_lr,
        )
    
    run_name = args.run_name
    if run_name is None:
        run_name = (
            f"hs{args.hidden_size}_hd{args.heads}_"
            f"lr{args.learning_rate}_ns{args.noise_std}_e{args.num_epochs}_"
            f"hhw{args.hyper_hidden_width}_mlp{args.num_mlp_layers}"
        )

    if args.wandb_mode == "disabled":
        os.environ["WANDB_MODE"] = "disabled"
    elif args.wandb_mode == "offline":
        os.environ["WANDB_MODE"] = "offline"

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=run_name,
        group=args.group,
        notes=args.notes,
        config={
            **vars(args),
            "resolved_hyperparameters": hyperparameters,
            "device": str(device),
        },
    )
    wandb.define_metric("epoch")
    wandb.define_metric("epoch/*", step_metric="epoch")

    num_samples_train = len(train_data) - args.dt
    num_batches_train = max(0, num_samples_train // args.batch_size)


    if bool(args.resume):
        if use_hypernet:
            model, start_epoch, global_step, optimizer, hypernet, optimizer_hyper = resume_training(
                args.output_dir,
                args.resume_epoch,
                device,
                use_hypernet,
                hypernet=hypernet,
            )
            wrapper = Step(model, hypernet)
            wrapper = torch.compile(wrapper)
        else:
            model, start_epoch, global_step, optimizer = resume_training(
                args.output_dir,
                args.resume_epoch,
                device,
                use_hypernet,
            )
    else:
        start_epoch = 1
        global_step = 0
        print(f"Starting training: {args.num_epochs} epochs with {num_batches_train} train batches/epoch")

    best_val_loss = float("inf")
    
    count_parameters(model)
    if use_hypernet:
        count_parameters(hypernet)

    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()
        train_loader = create_turbulence_dataloader(train_data, args.batch_size, dt=args.dt, shuffle=True)

        progress_bar = tqdm(total=num_batches_train, desc=f"Epoch {epoch}/{args.num_epochs}")
        epoch_losses = []

        for _ in range(num_batches_train):
            taus = 0.001 + (1.0 - 0.001) * torch.rand(args.num_taus, device=device, dtype=torch.float32)

            try:
                x_t, x_nt = next(train_loader)
                if x_t.shape[0] != args.batch_size:
                    continue

                x_t = x_t.to(device=device, dtype=torch.float32)
                x_nt = x_nt.to(device=device, dtype=torch.float32)
                x_dt = x_nt - x_t
 
                if use_hypernet:
                    loss = hyper_update_step(
                        model, hypernet, wrapper, optimizer, optimizer_hyper,
                        taus, x_dt, x_t, vpsde, args.noise_std, 
                        args.gradient_clip_norm
                    )
                else:
                    loss = update_step(
                        model, optimizer, taus, x_dt, x_t,
                        vpsde, args.noise_std, args.gradient_clip_norm
                    )

                scheduler.step()
                if use_hypernet:
                    scheduler_hyper.step()

                epoch_losses.append(loss.item())
                global_step += 1
                progress_bar.set_postfix({"train_loss": f"{loss.item():.4f}"})
                progress_bar.update(1)

            except StopIteration:
                break

        progress_bar.close()

        train_loss = float(torch.tensor(epoch_losses).mean()) if epoch_losses else float("nan")
        val_loss = run_validation(
            model, val_data, args.batch_size, args.dt, device,
            vpsde, sde_eps = 1e-3, hypernet=hypernet, wrapper=wrapper
        )

        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f} - Sampling Loss: {val_loss:.4f}")

        log_dict = {
            "epoch": epoch,
            "epoch/train_loss": train_loss,
            "epoch/val_loss": val_loss,
            "epoch/lr": optimizer.param_groups[0]["lr"],
        }
        if use_hypernet:
            log_dict["epoch/hyper_lr"] = optimizer_hyper.param_groups[0]["lr"]
        run.log(log_dict, step=epoch)

        
        if (epoch % args.save_every == 0) or (val_loss < best_val_loss):
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            print("Saving checkpoint")
            save_path = os.path.join(args.output_dir, f"model_epoch_{epoch}.pt")
            saveModel(save_path, hyperparameters, model)
            print(f"Model saved to {save_path}")
        
            if use_hypernet:
                hypernet_save_path = os.path.join(args.output_dir, f"hypernet_epoch_{epoch}.pt")
                torch.save(hypernet.state_dict(), hypernet_save_path)
                torch.save(
                    {
                        "optimizer": optimizer.state_dict(),
                        "optimizer_hyper": optimizer_hyper.state_dict(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "lr_hyper": optimizer_hyper.param_groups[0]["lr"],
                        "epoch": epoch,
                        "global_step": global_step,
                    },
                    os.path.join(args.output_dir, f"optimizer_lr_epoch_gs_{epoch}.pt"),
                )
            else:
                torch.save(
                    {
                        "optimizer": optimizer.state_dict(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        "global_step": global_step,
                    },
                    os.path.join(args.output_dir, f"optimizer_lr_epoch_gs_{epoch}.pt"),
                )

    saveModel(os.path.join(args.output_dir, "model_final.pt"), hyperparameters, model)
    if use_hypernet:
        torch.save(hypernet.state_dict(), os.path.join(args.output_dir, "hypernet_final.pt"))
    print("Training complete")
    run.finish()


if __name__ == "__main__":
    main()
