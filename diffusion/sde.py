import torch
import torch.nn as nn
import math

class VPSDE(nn.Module):
    beta_min: float
    beta_max: float
    T: float = 1.0
    schedule_type: str = "linear"
    gamma: float = 1.0
    power: float = 3.1
    
    def __init__(self, beta_min=0.1, beta_max=25.0, T=1.0, schedule_type="linear", gamma=1.0, power=3.1):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self.schedule_type = schedule_type
        self.gamma = gamma
        self.power = power

    def beta(self, t):
        """Beta schedule function."""
        # Ensure t is a tensor and get device/dtype
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        device = t.device
        dtype = t.dtype
        
        beta_min = torch.tensor(self.beta_min, device=device, dtype=dtype)
        beta_max = torch.tensor(self.beta_max, device=device, dtype=dtype)
        
        if self.schedule_type == "linear":
            return beta_min + t * (beta_max - beta_min)
        elif self.schedule_type == "cosine":
            cos_t = torch.cos((t * math.pi) / 2)
            normalized = 1 - (cos_t ** 2)  # Goes from 0 to 1
            return beta_min + normalized * (beta_max - beta_min)
        elif self.schedule_type == "power":
            return beta_min + (beta_max - beta_min) * (t ** self.power)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def beta_integral(self, t):
        """Analytical integral of beta from 0 to t."""
        # Ensure t is a tensor and get device/dtype
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        device = t.device
        dtype = t.dtype
        
        beta_min = torch.tensor(self.beta_min, device=device, dtype=dtype)
        beta_max = torch.tensor(self.beta_max, device=device, dtype=dtype)
        
        if self.schedule_type == "linear":
            # For linear schedule: ∫(beta_min + s*(beta_max-beta_min))ds from 0 to t
            return beta_min * t + 0.5 * (beta_max - beta_min) * t ** 2
        
        elif self.schedule_type == "cosine":
            # For cosine schedule: ∫(beta_min + (1-cos²(πs/2))*(beta_max-beta_min))ds from 0 to t
            return beta_min * t + (beta_max - beta_min) * (t - (2/math.pi) * torch.sin(math.pi * t / 2))
        elif self.schedule_type == "power":
            # Integral of t^p from 0 to t
            return beta_min * t + (beta_max - beta_min) * (t ** (self.power + 1)) / (self.power + 1)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def alpha(self, t):
        """Alpha function: exp(-0.5 * beta_integral)."""
        return torch.exp(-0.5 * self.beta_integral(t))

    def drift(self, x, t):
        """Drift term for the SDE."""
        return -0.5 * self.beta(t) * x

    def diffusion(self, x, t):
        """Diffusion coefficient for the SDE."""
        return torch.sqrt(self.beta(t)) * self.gamma

    def marginal_prob(self, x, t):
        """Marginal probability parameters for data at time t."""
        mean = self.alpha(t) * x
        #std = torch.max(torch.sqrt(1.0 - self.alpha(t)**2) * self.gamma, torch.sqrt(torch.tensor(1e-6)))
        std = torch.sqrt(1.0 - self.alpha(t)**2) * self.gamma
        return mean, std

    def prior_sampling(self, shape, device):
        """Sample from the prior distribution."""
        T_tensor = torch.tensor(self.T, device=device, dtype=torch.float32)
        _, std = self.marginal_prob(torch.zeros(shape, device=device, dtype=torch.float32), T_tensor)
        return torch.randn(shape, device=device, dtype=torch.float32) * std

    def forward_sample(self, x, t):
        mean, std = self.marginal_prob(x, t)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype)
        return mean + std * noise
