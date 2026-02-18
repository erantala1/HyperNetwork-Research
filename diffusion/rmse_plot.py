import os
import numpy as np
import matplotlib.pyplot as plt
from torch.func import functional_call
import torch
from fvcore.nn import FlopCountAnalysis
from utils import loadModel, load_hypernet  # same as sampler

from dataloaders import load_turbulence_data

# ==============================================================================
# CONFIG
# ==============================================================================

data_dir = "/glade/derecho/scratch/cainslie/beta-channel-turbulence/data_low_res/data_lowres"

# Must match the start idx used when generating EACH rollout file.
# If different rollouts used different start idx, set per-rollout below in ROLLOUTS.
DEFAULT_START_DATA_IDX = 14000

# Rollout files to compare.
# Add as many as you want.
# label: legend label
# path:  .npz file containing predicted_states
# start_idx: optional override if that rollout used a different start_data_idx
ROLLOUTS = [
    {
        "label": "baseline_width_16",
        "path": "/glade/u/home/erantala/jacobian_research/numpy_data/baseline_gamma_5_predictions.npz",
        "start_idx": DEFAULT_START_DATA_IDX,
    },

    {
        "label": "hyper_unet_width_8",
        "path": "/glade/u/home/erantala/jacobian_research/numpy_data/hypernet_unet_width_8_predictions.npz",
        "start_idx": DEFAULT_START_DATA_IDX,
    },
    
    {
        "label": "unet_width_8",
        "path": "/glade/u/home/erantala/jacobian_research/numpy_data/unet_width_8_predictions.npz",
        "start_idx": DEFAULT_START_DATA_IDX,
    },
]

# Output plot
out_dir = "/glade/u/home/erantala/jacobian_research/diffusion_plots/rmse_curves/test"
os.makedirs(out_dir, exist_ok=True)
out_png = os.path.join(out_dir, "testing_rollout_v2_rmse_comparison.png")

# Compute delta RMSE too?
COMPUTE_DELTA_RMSE = False

FLOP_MODELS = [
    {
        "label": "baseline_width_16",
        "model_ckpt": "/glade/derecho/scratch/erantala/project_runs/model_chkpts/diffusion/baseline_v3/model_epoch_85.pt",
        "hyper_ckpt": None,
    },
    {
        "label": "hyper_unet_width_8",
        "model_ckpt": "/glade/derecho/scratch/erantala/project_runs/model_chkpts/diffusion/width8_hyper/model_epoch_130.pt",
        "hyper_ckpt": "/glade/derecho/scratch/erantala/project_runs/model_chkpts/diffusion/width8_hyper/hypernet_epoch_130.pt",
    },
    {
        "label": "unet_width_8",
        "model_ckpt": "/glade/derecho/scratch/erantala/project_runs/model_chkpts/diffusion/width8_no_hyper/model_epoch_90.pt",
        "hyper_ckpt": None,
    },
]

out_flops_png = os.path.join(out_dir, "flops_per_ddpm_step.png")
# ==============================================================================
# HELPERS
# ==============================================================================
# ==============================================================================
# FLOP HELPERS
# ==============================================================================

def _to_torch(x_np: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x_np, dtype=torch.float32, device=device)

@torch.inference_mode()
def compute_flops_one_ddpm_step(model, condition_x_t, device, hypernet=None, preprocess_step=None):
    model = model.to(device).eval()

    # Create y_t with same shape as condition_x_t
    # condition_x_t is (1,H,W) in your sampler
    y_t = torch.randn_like(condition_x_t)

    # model_input becomes (2,H,W) after cat on dim=0
    model_input = torch.cat([y_t, condition_x_t], dim=0)

    # time scalar like sampler
    t = torch.tensor(0.5, device=device, dtype=torch.float32)

    flops_total = 0.0

    # --- optional hypernet FLOPs ---
    if hypernet is not None and preprocess_step is not None:
        hypernet = hypernet.to(device).eval()

        # hypernet input: z = preprocess_step(condition_x_t)
        z = preprocess_step(condition_x_t)

        # Count hypernet FLOPs (forward only)
        flops_h = FlopCountAnalysis(hypernet, (z,))
        flops_total += float(flops_h.total())

        # (Optional) actually run it once so any lazy init happens outside UNet FLOPs
        _ = hypernet(z)

    # --- UNet FLOPs (normal forward) ---
    flops_m = FlopCountAnalysis(model, (t, model_input))
    flops_total += float(flops_m.total())

    # (Optional) run once for warmup symmetry
    _ = model(t, model_input)

    return flops_total

def _ensure_shape_TCHW(pred_states: np.ndarray) -> np.ndarray:
    """
    Expect predicted_states to be (T, 1, H, W) per your sampler.
    Handle some common variants robustly.
    """
    if pred_states.ndim == 4:
        return pred_states
    raise ValueError(f"predicted_states expected 4D (T,C,H,W), got shape {pred_states.shape}")


def compute_rmse_curves(pred_states: np.ndarray, gt_states: np.ndarray) -> np.ndarray:
    """
    pred_states: (T, C, H, W) corresponds to steps 1..T
    gt_states:   (T, C, H, W) ground truth at t+1..t+T
    returns rmse: (T,)
    """
    assert pred_states.shape == gt_states.shape, (pred_states.shape, gt_states.shape)
    diff = pred_states - gt_states
    mse = np.mean(diff**2, axis=(1, 2, 3))
    rmse = np.sqrt(mse)
    return rmse


def compute_delta_rmse_curves(pred_states: np.ndarray, gt_states: np.ndarray, gt_prev0: np.ndarray) -> np.ndarray:
    """
    pred_states: (T,C,H,W) predicted states at t+1..t+T
    gt_states:   (T,C,H,W) GT states at t+1..t+T
    gt_prev0:    (C,H,W)   GT state at t (the start frame)

    delta definitions match your sampler post-processing:
      pred_dt[k] = pred_state[k] - pred_state[k-1], with pred_state[-1] = gt_prev0 for k=0
      gt_dt[k]   = gt_state[k]   - gt_state[k-1], with gt_state[-1] = gt_prev0 for k=0
    """
    T = pred_states.shape[0]
    pred_prev = np.concatenate([gt_prev0[None, ...], pred_states[:-1]], axis=0)
    gt_prev   = np.concatenate([gt_prev0[None, ...], gt_states[:-1]], axis=0)
    pred_dt = pred_states - pred_prev
    gt_dt   = gt_states   - gt_prev
    diff = pred_dt - gt_dt
    mse = np.mean(diff**2, axis=(1, 2, 3))
    rmse = np.sqrt(mse)
    return rmse

# ==============================================================================
# EXTRA METRICS: HISTOGRAM + SPECTRA
# ==============================================================================

def flatten_vorticity(states: np.ndarray) -> np.ndarray:
    """
    states: (T, 1, H, W)
    returns: (T*H*W,)
    """
    return states[:, 0, :, :].reshape(-1)


def compute_vorticity_histogram(states: np.ndarray, bins=200, vmin=-6.0, vmax=6.0):
    """
    Returns bin centers and PDF values (density) over all times.
    """
    vals = flatten_vorticity(states)
    hist, edges = np.histogram(vals, bins=bins, range=(vmin, vmax), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def compute_latitudinal_spectrum(states: np.ndarray):
    """
    Latitudinal spectrum: FFT along latitude (H dimension), averaged over time and longitude (W).

    states: (T, 1, H, W)
    returns:
      k:  (H//2+1,)
      Pk: (H//2+1,)
    """
    omega = states[:, 0, :, :]  # (T, H, W)
    T, H, W = omega.shape

    # rFFT along latitude axis (H)
    F = np.fft.rfft(omega, axis=1)  # (T, H//2+1, W)
    P = (np.abs(F) ** 2).mean(axis=(0, 2))  # avg over time and longitude -> (H//2+1,)

    k = np.fft.rfftfreq(H) * H
    return k, P


def compute_tke_spectrum(states: np.ndarray, eps=1e-12):
    """
    Turbulent kinetic energy (TKE) spectrum from vorticity via streamfunction inversion:

      ω̂ = -k^2 ψ̂  => ψ̂ = -ω̂ / k^2
      û = i ky ψ̂, v̂ = -i kx ψ̂
      E(k) = 0.5 (|û|^2 + |v̂|^2) binned by radial wavenumber |k|.

    states: (T, 1, H, W)
    returns:
      k_bins: (K,)
      E_k:    (K,)
    """
    omega = states[:, 0, :, :]  # (T, H, W)
    T, H, W = omega.shape

    ky = np.fft.fftfreq(H) * H  # (-H/2..H/2)
    kx = np.fft.fftfreq(W) * W
    KY, KX = np.meshgrid(ky, kx, indexing="ij")  # (H, W)
    K2 = KX**2 + KY**2
    K = np.sqrt(K2)

    # integer radial bins (0..kmax)
    kmax = int(np.max(K))
    k_int = np.clip(K.astype(np.int32), 0, kmax)

    E_accum = np.zeros(kmax + 1, dtype=np.float64)
    N_accum = np.zeros(kmax + 1, dtype=np.float64)

    # avoid division by zero at k=0
    K2_safe = K2.copy()
    K2_safe[0, 0] = 1.0

    for t in range(T):
        w_hat = np.fft.fft2(omega[t])  # (H, W), complex
        psi_hat = -w_hat / (K2_safe + eps)

        u_hat = 1j * KY * psi_hat
        v_hat = -1j * KX * psi_hat

        E_hat = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2)  # (H, W)

        # bin by radial k
        E_flat = E_hat.ravel()
        k_flat = k_int.ravel()

        E_accum += np.bincount(k_flat, weights=E_flat, minlength=kmax + 1)
        N_accum += np.bincount(k_flat, weights=np.ones_like(E_flat), minlength=kmax + 1)

    E_k = E_accum / np.maximum(N_accum, 1.0)
    k_bins = np.arange(kmax + 1)

    # drop k=0 (often singular / not meaningful)
    return k_bins[1:], E_k[1:]



# ==============================================================================
# MAIN
# ==============================================================================

def main():
    results = []

    for item in ROLLOUTS:
        label = item["label"]
        path = item["path"]
        start_idx = int(item.get("start_idx", DEFAULT_START_DATA_IDX))

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing rollout file: {path}")

        loaded = np.load(path)
        if "predicted_states" not in loaded:
            raise KeyError(f"{path} does not contain key 'predicted_states'. Keys: {list(loaded.keys())}")

        pred_states = _ensure_shape_TCHW(loaded["predicted_states"]).astype(np.float32)
        T = pred_states.shape[0]

        # Load GT frames needed for comparison
        stop_idx = start_idx + T
        gt_dict = load_turbulence_data(data_dir, start_idx=start_idx, stop_idx=stop_idx, normalize=True)

        # Build GT tensor aligned to predicted steps:
        # predicted step k corresponds to GT at (start_idx + (k+1))
        gt_list = []
        for k in range(T):
            gt_idx = start_idx + (k + 1)
            if gt_idx not in gt_dict:
                raise KeyError(f"GT index {gt_idx} missing from loaded data (start={start_idx}, stop={stop_idx}).")
            gt_list.append(np.asarray(gt_dict[gt_idx], dtype=np.float32)[None, ...])  # (1,H,W)

        gt_states = np.stack(gt_list, axis=0)  # (T,1,H,W)

        # Compute RMSE
        rmse_state = compute_rmse_curves(pred_states, gt_states)

        if COMPUTE_DELTA_RMSE:
            if start_idx not in gt_dict:
                # if load_turbulence_data doesn't include start_idx frame, load it directly
                gt0_dict = load_turbulence_data(data_dir, start_idx=start_idx, stop_idx=start_idx, normalize=True)
                if start_idx not in gt0_dict:
                    raise KeyError(f"Could not load GT start frame {start_idx} for delta RMSE.")
                gt_prev0 = np.asarray(gt0_dict[start_idx], dtype=np.float32)[None, ...]  # (1,H,W)
            else:
                gt_prev0 = np.asarray(gt_dict[start_idx], dtype=np.float32)[None, ...]  # (1,H,W)

            rmse_delta = compute_delta_rmse_curves(pred_states, gt_states, gt_prev0)
        else:
            rmse_delta = None

        results.append(
            {
                "label": label,
                "rmse_state": rmse_state,
                "rmse_delta": rmse_delta,
                "T": T,
                "pred_states": pred_states,
                "gt_states": gt_states,
            }
        )
        print(f"[ok] {label}: loaded T={T} from {path} (start_idx={start_idx})")

    # Plot
    plt.figure(figsize=(10, 5))
    for r in results:
        T_MAX = 100
        T_plot = min(r["T"], T_MAX)
        x = np.arange(1, T_plot + 1)
        plt.plot(x, r["rmse_state"][:T_plot], label=r["label"])

    plt.xlabel("Rollout step")
    plt.ylabel("State RMSE")
    plt.title("Autoregressive rollout RMSE vs step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[saved] {out_png}")

    if COMPUTE_DELTA_RMSE:
        out_png_dt = os.path.join(out_dir, "rollout_delta_rmse_comparison.png")
        plt.figure(figsize=(10, 5))
        for r in results:
            x = np.arange(1, r["T"][:T_plot] + 1)
            plt.plot(x, r["rmse_delta"], label=r["label"])
        plt.xlabel("Rollout step")
        plt.ylabel("Delta RMSE")
        plt.title("Autoregressive rollout Δ RMSE vs step")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png_dt, dpi=150)
        print(f"[saved] {out_png_dt}")
        # ==============================================================================
    # EXTRA PLOTS: Histogram + Spectra (one plot, all models + GT)
    # ==============================================================================

    # Use a common horizon so GT/model comparisons align
    T_common = min(r["T"] for r in results)
    print(f"[info] Using T_common={T_common} for histogram/spectra comparisons")

    # Use GT from the first rollout (same start_idx setup) and truncate
    gt_common = results[0]["gt_states"][:T_common]

    # 1) Total histogram of vorticity values
    out_hist = os.path.join(out_dir, "rollout_vorticity_histogram_all_models.png")
    plt.figure(figsize=(7, 5))

    x_gt, pdf_gt = compute_vorticity_histogram(gt_common, bins=250, vmin=-6.0, vmax=6.0)
    plt.plot(x_gt, pdf_gt, label="truth", linewidth=2)

    for r in results:
        pred_common = r["pred_states"][:T_common]
        x_m, pdf_m = compute_vorticity_histogram(pred_common, bins=250, vmin=-6.0, vmax=6.0)
        plt.plot(x_m, pdf_m, label=r["label"], alpha=0.9)

    plt.yscale("log")
    plt.xlabel("Vorticity")
    plt.ylabel("PDF")
    plt.title("Total vorticity histogram over rollout")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_hist, dpi=150)
    print(f"[saved] {out_hist}")

    # 2) Latitudinal spectrum averaged across the rollout
    out_lat = os.path.join(out_dir, "rollout_latitudinal_spectrum_all_models.png")
    plt.figure(figsize=(7, 5))

    k_gt, P_gt = compute_latitudinal_spectrum(gt_common)
    k_cut = int((2/3) * np.max(k_gt))
    mask_lat = (k_gt >= 1) & (k_gt <= k_cut)

    plt.loglog(k_gt[mask_lat], P_gt[mask_lat], label="truth", linewidth=2)

    for r in results:
        k_m, P_m = compute_latitudinal_spectrum(r["pred_states"][:T_common])
        plt.loglog(k_m[mask_lat], P_m[mask_lat], label=r["label"], alpha=0.9)

    plt.xlabel("Wavenumber")
    plt.ylabel("Latitudinal spectrum (power)")
    plt.title("Latitudinal spectrum averaged over rollout")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_lat, dpi=150)
    print(f"[saved] {out_lat}")

    # 3) TKE spectrum averaged across the rollout
    out_tke = os.path.join(out_dir, "rollout_tke_spectrum_all_models.png")
    plt.figure(figsize=(7, 5))

    k_gt, E_gt = compute_tke_spectrum(gt_common)
    k_cut = int((2/3) * np.max(k_gt))
    mask_tke = (k_gt >= 1) & (k_gt <= k_cut)

    plt.loglog(k_gt[mask_tke], E_gt[mask_tke], label="truth", linewidth=2)
    for r in results:
        pred_common = r["pred_states"][:T_common]
        k_m, E_m = compute_tke_spectrum(pred_common)
        plt.loglog(k_m[mask_tke], E_m[mask_tke], label=r["label"], alpha=0.9)

    plt.xlabel("Wavenumber")
    plt.ylabel("TKE spectrum")
    plt.title("TKE spectrum averaged over rollout")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_tke, dpi=150)
    print(f"[saved] {out_tke}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cond_np = gt_common[0]  # (1,H,W)
    condition_x_t = _to_torch(cond_np, device=device)

    flops_g = []
    labels = []

    print("\n--- FLOP COUNT (one DDPM model eval) ---")
    for item in FLOP_MODELS:
        label = item["label"]
        model_ckpt = item["model_ckpt"]
        hyper_ckpt = item["hyper_ckpt"]

        print(f"\n[label={label}] Loading model: {model_ckpt}")
        model_tmp = loadModel(model_ckpt).to(device).eval()

        if hyper_ckpt is not None:
            print(f"[label={label}] Loading hypernet: {hyper_ckpt}")
            hyper_tmp, preprocess_step = load_hypernet(hyper_ckpt, model_tmp, device)
            hyper_tmp = hyper_tmp.to(device).eval()
        else:
            hyper_tmp, preprocess_step = None, None

        # Warmup (so any lazy init doesn't distort timing, though FLOPs should be stable)
        _ = compute_flops_one_ddpm_step(model_tmp, condition_x_t, device, hyper_tmp, preprocess_step)

        flops_total = compute_flops_one_ddpm_step(model_tmp, condition_x_t, device, hyper_tmp, preprocess_step)

        gflops = flops_total / 1e9
        print(f"[label={label}] FLOPs per DDPM step: {gflops:.4f} GFLOPs")

        labels.append(label)
        flops_g.append(gflops)

    # Plot like your example
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, flops_g)
    plt.title("FLOP Count per DDPM Step", fontsize=18, fontweight="bold")
    plt.ylabel("FLOPs (GFLOPs)", fontsize=16, fontweight="bold")
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis="y", alpha=0.3)

    # Annotate values on bars
    for b, val in zip(bars, flops_g):
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(out_flops_png, dpi=150)
    print(f"[saved] {out_flops_png}")

if __name__ == "__main__":
    main()
