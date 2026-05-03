import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
from torch.func import functional_call
from wrapper import Step
from utils import loadModel, load_hypernet
from dataloaders import load_turbulence_data
from sde import VPSDE
import os, sys

# --- Output Directories ---
BASE_DIR = "/glade/derecho/scratch/erantala/project_runs/model_chkpts/diffusion"
output_base_dir = "/glade/u/home/erantala/jacobian_research"
model_name = "tau_updates_no_concat_w16_h16_r2_l6_taus4_bs8" # change per experiment

plot_dir = os.path.join(output_base_dir, "diffusion_plots")
plot_dir = os.path.join(plot_dir, model_name)
video_dir = os.path.join(output_base_dir, "video")
numpy_dir = os.path.join(output_base_dir, "numpy_data")
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)
os.makedirs(numpy_dir, exist_ok=True)
print(f"Output base directory: {os.path.abspath(output_base_dir)}")

#modelfile = os.path.join(BASE_DIR, "width_4_24_heads_dimhead_64/model_epoch_150.pt") # best model
#hypernetfile = os.path.join(BASE_DIR, "width_4_24_heads_dimhead_64/hypernet_epoch_150.pt")

modelfile = os.path.join(BASE_DIR, "tau_updates_no_concat_w16_h16_r2_l6_taus4_bs8/model_epoch_50.pt")
hypernetfile = os.path.join(BASE_DIR, "tau_updates_no_concat_w16_h16_r2_l6_taus4_bs8/hypernet_epoch_50.pt")

start_data_idx = 20000
num_rollout_steps = 5

# --- Sampling Parameters ---
num_sampling_steps = 1000
sde_eps = 1e-3

# --- Plotting and Video Parameters ---
#save_milestones = [1, 50, 100, 250, 500, 750, 1000, 1500, 2000, 3000]
save_milestones = [1, 5, 10, 20, 50, 75, 100, 150, 200]
output_video_filename = os.path.join(video_dir, f"{model_name}_comparison.mp4")
output_numpy_filename = os.path.join(numpy_dir, f"{model_name}_predictions.npz")
fps = 30

# --- Define constant color bar limits ---
state_vmin, state_vmax = -8.0, 8.0
delta_vmin, delta_vmax = -5.0, 5.0
diff_state_vmin, diff_state_vmax = -8.0, 8.0
diff_delta_vmin, diff_delta_vmax = -5.0, 5.0
hist_bins = 50

# --- Control Flags ---
RUN_SIMULATION = True
RUN_POSTPROCESSING = True
USE_HYPERNET = True
# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
#                  Sampler: reverse SDE (Euler–Maruyama)
# ==============================================================================

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
            #model_input = torch.cat([y_t, condition_x_t], dim=0)
            model_input = y_t
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

def backwards_euler(wrapper, condition_x_t, num_iters):
    params = wrapper.make_params(condition_x_t)
    params = {k: v.squeeze(0) for k, v in params.items()}
    x_nt = condition_x_t + wrapper(condition_x_t, params)
    iter = 0
    while iter < num_iters:
        x_nt = condition_x_t + wrapper(x_nt, params)
        iter += 1
    return x_nt

# ==============================================================================
#                  PHASE 1: RUN SIMULATION & SAVE RESULTS
# ==============================================================================

if RUN_SIMULATION:
    print("\n--- PHASE 1: Starting Simulation ---")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Loading model from {modelfile}...")
    try:
        model = loadModel(modelfile)
        model.to(device)
        model.eval()
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    if USE_HYPERNET:
        print(f"Loading hypernetwork model from {hypernetfile}...")
        hypernet = load_hypernet(hypernetfile, model, device)
        hypernet.eval()
        wrapper = Step(model, hypernet)
        wrapper = torch.compile(wrapper)
    else:
        hypernet = None
        model = torch.compile(model)

    
    vpsde = VPSDE(beta_min=0.01, beta_max=55.0, T=1.0, schedule_type="power", power=5).to(device)

    data_dir = "/glade/derecho/scratch/cainslie/beta-channel-turbulence/data_low_res/data_lowres"

    print(f"Loading initial state data (index {start_data_idx})...")
    try:
        
        initial_data = load_turbulence_data(
            data_dir, start_idx=start_data_idx, stop_idx=start_data_idx, normalize=True
        )

        if start_data_idx not in initial_data:
            raise ValueError(f"Index {start_data_idx} not found in loaded data.")

        current_predicted_x = torch.tensor(
            initial_data[start_data_idx], dtype=torch.float32, device=device
        )

        current_predicted_x = current_predicted_x.unsqueeze(0)

        delta_shape = tuple(current_predicted_x.shape)  # (1,H,W)
        print("Initial state loaded.")
    except Exception as e:
        print(f"Error loading initial data: {e}")
        raise

    print(f"Starting autoregressive rollout for {num_rollout_steps} steps...")
    predicted_states_list_np = []

    start_time = time.time()
    with torch.inference_mode():
        for k in tqdm(range(num_rollout_steps), desc="Rollout Simulation"):
            condition_x_t = current_predicted_x 
            
            predicted_delta = reverse_sde_sampler(
                model_state=model,
                vpsde=vpsde,
                condition_x_t=condition_x_t,
                delta_shape=delta_shape,
                num_steps=num_sampling_steps,
                eps=sde_eps,
                device=device,
                hypernet=hypernet,
            )
            
            current_predicted_x = condition_x_t + predicted_delta

            #current_predicted_x = backwards_euler(wrapper, condition_x_t, num_iters=10)

            predicted_states_list_np.append(current_predicted_x.detach().cpu().numpy())

    end_time = time.time()
    print(f"Simulation finished in {end_time - start_time:.2f} seconds.")

    print("Consolidating simulation results...")
    all_predicted_states_np = np.stack(predicted_states_list_np, axis=0)

    print(f"Saving predicted states ({all_predicted_states_np.shape}) to {output_numpy_filename}")
    try:
        np.savez_compressed(output_numpy_filename, predicted_states=all_predicted_states_np)
        print("NumPy results saved successfully.")
    except Exception as e:
        print(f"Error saving NumPy results: {e}")

    print("--- PHASE 1: Simulation Complete ---")

else:
    print("\n--- PHASE 1: Skipping Simulation (RUN_SIMULATION=False) ---")


# ==============================================================================
#           PHASE 2: LOAD RESULTS & GENERATE PLOTS/VIDEO
# ==============================================================================

if RUN_POSTPROCESSING:
    print("\n--- PHASE 2: Starting Post-Processing (Plots and Video) ---")

    print(f"Loading predicted states from {output_numpy_filename}...")
    try:
        loaded_data = np.load(output_numpy_filename)
        all_predicted_states_np = loaded_data["predicted_states"]
        print(f"Loaded predicted states shape: {all_predicted_states_np.shape}")
        num_rollout_steps = all_predicted_states_np.shape[0]
        print(f"Number of rollout steps loaded: {num_rollout_steps}")
    except Exception as e:
        print(f"Error loading NumPy results from {output_numpy_filename}: {e}")
        raise

    data_dir = "/glade/derecho/scratch/cainslie/beta-channel-turbulence/data_low_res/data_lowres"
    stop_data_idx = start_data_idx + num_rollout_steps
    max_available_idx = 40000
    if stop_data_idx > max_available_idx:
        print(
            f"Warning: Required stop index {stop_data_idx} for ground truth exceeds max available {max_available_idx}."
        )
        stop_data_idx = max_available_idx
        print("Adjusting comparison range to max available index.")
        num_rollout_steps = min(num_rollout_steps, stop_data_idx - start_data_idx)

    print(f"Loading ground truth data from index {start_data_idx} to {stop_data_idx}...")
    try:
        data = load_turbulence_data(
            data_dir, start_idx=start_data_idx, stop_idx=stop_data_idx, normalize=True
        )
        print(f"Ground truth data loaded. Number of samples: {len(data)}")
        if len(data) < num_rollout_steps + 1:
            print(
                f"Warning: Loaded ground truth ({len(data)} samples) is less than required for {num_rollout_steps} steps + initial state."
            )
            num_rollout_steps = len(data) - 1
            print(f"Adjusting comparison range to {num_rollout_steps} steps.")
    except Exception as e:
        print(f"Error loading ground truth data: {e}")
        raise

    # --- Initialize Video Writer ---
    print(f"Initializing video writer for {output_video_filename} at {fps} FPS...")
    try:
        writer = imageio.get_writer(output_video_filename, fps=fps, macro_block_size=None)
    except Exception as e:
        print(f"Error initializing video writer: {e}")
        raise
    print("Generating video frames and milestone plots...")
    start_time_post = time.time()
    for k in tqdm(range(num_rollout_steps), desc="Post-Processing"):
        current_step_number = k + 1
        target_idx = start_data_idx + current_step_number
        pred_idx = k

        if target_idx not in data or (target_idx - 1) not in data:
            print(
                f"\nSkipping step {current_step_number}, target data index {target_idx} or previous not available."
            )
            continue

        current_predicted_x_np = all_predicted_states_np[pred_idx]  # (1,H,W)

        target_x_nt_np = np.asarray(data[target_idx], dtype=np.float32)
        target_x_nt_np = np.expand_dims(target_x_nt_np, axis=0)  # (1,H,W)

        # --- Video frame ---
        try:
            fig_video, axes_video = plt.subplots(1, 2, figsize=(10, 5))
            axes_video[0].imshow(
                np.squeeze(target_x_nt_np),
                cmap="viridis",
                vmin=state_vmin,
                vmax=state_vmax,
            )
            axes_video[0].set_title(f"Ground Truth (t+{current_step_number})")
            axes_video[0].set_xticks([])
            axes_video[0].set_yticks([])

            axes_video[1].imshow(
                np.squeeze(current_predicted_x_np),
                cmap="viridis",
                vmin=state_vmin,
                vmax=state_vmax,
            )
            axes_video[1].set_title(f"Prediction (t+{current_step_number})")
            axes_video[1].set_xticks([])
            axes_video[1].set_yticks([])

            fig_video.suptitle(f"Rollout Step {current_step_number}", fontsize=12)
            plt.tight_layout(rect=[0, 0.03, 1, 0.93])
            fig_video.canvas.draw()
            frame_array = np.asarray(fig_video.canvas.buffer_rgba())
            writer.append_data(frame_array)
            plt.close(fig_video)
        except Exception as e:
            print(f"\nError generating video frame at step {current_step_number}: {e}")

        # --- Milestone plots ---
        if current_step_number in save_milestones:
            print(f"\nGenerating detailed plot for step {current_step_number}...")
            try:
                target_dt_np = np.asarray(
                    data[target_idx] - data[target_idx - 1], dtype=np.float32
                )
                target_dt_np = np.expand_dims(target_dt_np, axis=0)

                if k == 0:
                    prev_predicted_x_np = np.asarray(
                        data[start_data_idx], dtype=np.float32
                    )
                    prev_predicted_x_np = np.expand_dims(prev_predicted_x_np, axis=0)
                else:
                    prev_predicted_x_np = all_predicted_states_np[pred_idx - 1]

                predicted_delta_np = current_predicted_x_np - prev_predicted_x_np

                mse_state = np.mean((current_predicted_x_np - target_x_nt_np) ** 2)
                mse_delta = np.mean((predicted_delta_np - target_dt_np) ** 2)
                print(f"  MSE State vs Target at step {current_step_number}: {mse_state:.6f}")
                print(f"  MSE Delta vs Target at step {current_step_number}: {mse_delta:.6f}")

                fig_milestone, axes_milestone = plt.subplots(2, 4, figsize=(24, 10))

                # == ROW 1: States ==
                im00 = axes_milestone[0, 0].imshow(
                    np.squeeze(target_x_nt_np),
                    cmap="viridis",
                    vmin=state_vmin,
                    vmax=state_vmax,
                )
                axes_milestone[0, 0].set_title("Target State")
                fig_milestone.colorbar(im00, ax=axes_milestone[0, 0], fraction=0.046, pad=0.04)

                im01 = axes_milestone[0, 1].imshow(
                    np.squeeze(current_predicted_x_np),
                    cmap="viridis",
                    vmin=state_vmin,
                    vmax=state_vmax,
                )
                axes_milestone[0, 1].set_title("Predicted State")
                fig_milestone.colorbar(im01, ax=axes_milestone[0, 1], fraction=0.046, pad=0.04)

                diff_state = target_x_nt_np - current_predicted_x_np
                im02 = axes_milestone[0, 2].imshow(
                    np.squeeze(diff_state),
                    cmap="coolwarm",
                    vmin=diff_state_vmin,
                    vmax=diff_state_vmax,
                )
                axes_milestone[0, 2].set_title("State Difference")
                fig_milestone.colorbar(im02, ax=axes_milestone[0, 2], fraction=0.046, pad=0.04)

                target_state_flat = target_x_nt_np.flatten()
                pred_state_flat = current_predicted_x_np.flatten()
                axes_milestone[0, 3].hist(
                    target_state_flat,
                    bins=hist_bins,
                    range=(state_vmin, state_vmax),
                    alpha=0.7,
                    label="Target",
                    color="blue",
                )
                axes_milestone[0, 3].hist(
                    pred_state_flat,
                    bins=hist_bins,
                    range=(state_vmin, state_vmax),
                    alpha=0.7,
                    label="Pred",
                    color="red",
                )
                axes_milestone[0, 3].set_title("State Histograms")
                axes_milestone[0, 3].legend()
                axes_milestone[0, 3].grid(axis="y", ls="--", alpha=0.7)

                # == ROW 2: Deltas (dt) ==
                im10 = axes_milestone[1, 0].imshow(
                    np.squeeze(target_dt_np),
                    cmap="viridis",
                    vmin=delta_vmin,
                    vmax=delta_vmax,
                )
                axes_milestone[1, 0].set_title("Target dt")
                fig_milestone.colorbar(im10, ax=axes_milestone[1, 0], fraction=0.046, pad=0.04)

                im11 = axes_milestone[1, 1].imshow(
                    np.squeeze(predicted_delta_np),
                    cmap="viridis",
                    vmin=delta_vmin,
                    vmax=delta_vmax,
                )
                axes_milestone[1, 1].set_title("Predicted dt")
                fig_milestone.colorbar(im11, ax=axes_milestone[1, 1], fraction=0.046, pad=0.04)

                diff_dt = target_dt_np - predicted_delta_np
                vmax_dt_diff = np.clip(np.max(np.abs(diff_dt)), a_min=None, a_max=delta_vmax)
                im12 = axes_milestone[1, 2].imshow(
                    np.squeeze(diff_dt),
                    cmap="coolwarm",
                    vmin=-vmax_dt_diff,
                    vmax=vmax_dt_diff,
                )
                axes_milestone[1, 2].set_title("Delta dt Difference")
                fig_milestone.colorbar(im12, ax=axes_milestone[1, 2], fraction=0.046, pad=0.04)

                target_dt_flat = target_dt_np.flatten()
                pred_dt_flat = predicted_delta_np.flatten()
                axes_milestone[1, 3].hist(
                    target_dt_flat,
                    bins=hist_bins,
                    range=(delta_vmin, delta_vmax),
                    alpha=0.7,
                    label="Target",
                    color="blue",
                )
                axes_milestone[1, 3].hist(
                    pred_dt_flat,
                    bins=hist_bins,
                    range=(delta_vmin, delta_vmax),
                    alpha=0.7,
                    label="Pred",
                    color="red",
                )
                axes_milestone[1, 3].set_title("Delta Histograms")
                axes_milestone[1, 3].legend()
                axes_milestone[1, 3].grid(axis="y", ls="--", alpha=0.7)

                fig_milestone.suptitle(
                    f"Autoregressive Rollout: Step {current_step_number}", fontsize=16
                )
                plt.subplots_adjust(hspace=0.4, wspace=0.3)

                plot_filename = os.path.join(plot_dir, f"rollout_step_{current_step_number:04d}.png")
                plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
                plt.close(fig_milestone)
                print(f"  Saved detailed plot to: {plot_filename}")

            except Exception as e:
                print(f"\nError generating detailed plot at step {current_step_number}: {e}")

    try:
        writer.close()
        print("\nVideo generation complete.")
        print(f"Video saved to: {output_video_filename}")
    except Exception as e:
        print(f"\nError closing video writer: {e}")

    end_time_post = time.time()
    print(f"Post-processing finished in {end_time_post - start_time_post:.2f} seconds.")
    print("--- PHASE 2: Post-Processing Complete ---")

else:
    print("\n--- PHASE 2: Skipping Post-Processing (RUN_POSTPROCESSING=False) ---")

print("\nScript finished.")
