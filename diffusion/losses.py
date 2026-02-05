import torch
from torch.func import functional_call

#continuous score loss from Song et al 2003
def continuousScoreLoss(model, t, x, y, vpsde, noise_std, params=None):
    # y is target field (e.g. x_dt), shape (C,H,W)
    noisy_y = vpsde.forward_sample(y, t)
    mean, std = vpsde.marginal_prob(y, t)
    target_score = -(noisy_y - mean) / std 
    # true score has denom std^2, model prediction is scaled by std in loss

    # x is conditioning input, expected to be concatenated [y, x_t]
    # take x_t as channel 1, x is (2,H,W) 
    condition_x_t = x[1:2, :, :] # (1,H,W)
    condition_noise = torch.randn_like(condition_x_t) * noise_std
    noisy_x_t = condition_x_t + condition_noise

    # toggle between these for concat conditioning vs only noisy input with hypernet conditioning
    model_input = torch.cat([noisy_y, noisy_x_t], dim=0) # (2,H,W)
    #model_input = noisy_y # (1,H,W)

    if params is None:
        pred_score = model(t, model_input)
    else:
        # functional_call for hypernetwork modified params
        pred_score = functional_call(model, params, (t, model_input))

    return torch.mean((std * pred_score - target_score) ** 2)
