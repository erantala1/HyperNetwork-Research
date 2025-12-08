import numpy as np
import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F



################################################################
#  1d Fourier Integral Operator
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super(SpectralConv1d, self).__init__()
        """
        Initializes the 1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        Args:
            in_channels (int): input channels to the FNO layer
            out_channels (int): output channels of the FNO layer
            modes (int): number of Fourier modes to multiply, at most floor(N/2) + 1
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = (1 / (in_channels*out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))
    # Complex multiplication
    def compl_mul1d(self, input, weights):
        """
        Complex multiplication of the Fourier modes.
        [batch, in_channels, x], [in_channel, out_channels, x] -> [batch, out_channels, x]
            Args:
                input (torch.Tensor): input tensor of size [batch, in_channels, x]
                weights (torch.Tensor): weight tensor of size [in_channels, out_channels, x]
            Returns:
                torch.Tensor: output tensor with shape [batch, out_channels, x]
        """
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fourier transformation, multiplication of relevant Fourier modes, backtransformation
        Args:
            x (torch.Tensor): input to forward pass os shape [batch, in_channels, x]
        Returns:
            torch.Tensor: output of size [batch, out_channels, x]
        """
        batchsize = x.shape[0]
        # Fourier transformation
        x_ft = torch.fft.rfft(x) 

        # Multiply relevant Fourier modes

        out_ft = torch.zeros(batchsize, self.out_channels, self.modes,  device=x.device, dtype=torch.cfloat)
        out_ft = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    def __init__(self, modes, width, time_future, time_history):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        input shape: (batchsize, x=s, c=1)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes = modes
        self.width = width
        self.time_future = time_future
        self.time_history = time_history
        self.fc0 = nn.Linear(self.time_history+1, self.width)
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes) #initializes with uniform distro from 1/width^2
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes)
        self.w0 = nn.Conv1d(self.width, self.width, 1) #initializes with uniform distro from -+sqrt(1/width)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.fc1 = nn.Linear(self.width, 128) #I dont understand why this and fc2 are here, also the 128 seems like a hyperparameter
        self.fc2 = nn.Linear(128, self.time_future)

    def forward(self, u):
        if len(u.shape)==2:
            u = u.unsqueeze(0) #when there is no batchsize
        grid = self.get_grid(u.shape, u.device)
        x = torch.cat((u, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
    
    def forward_with_params(self, u, params):
        """
        Batched forward that uses per-sample dynamic parameters in `params`
        for fc0, w0-w3, fc1, fc2 while leaving conv0-conv3 untouched.

        u:      [B, S, time_history+1]   (input sequence; S = spatial grid size)
        params: dict with keys:
            "fc0.weight"   [B, width, time_history+1]
            "fc0.bias"     [B, width]
            "w0.weight"    [B, width, width, 1]
            "w0.bias"      [B, width]
            "w1.weight"
            "w2.weight"
            "w3.weight"
            "fc1.weight"   [B, 128, width]
            "fc1.bias"     [B, 128]
            "fc2.weight"   [B, time_future, 128]
            "fc2.bias"     [B, time_future]
        """

        grid = self.get_grid(u.shape, u.device)
        x = torch.cat((u, grid), dim=-1)

        W0 = params["fc0.weight"]
        b0 = params["fc0.bias"]

        x = torch.einsum("bst, bwt -> bsw", x, W0) + b0.unsqueeze(1)

        x = x.permute(0, 2, 1)

        for layer_idx in [0, 1, 2, 3]:
            conv = getattr(self, f"conv{layer_idx}")
            x1 = conv(x)

            W = params[f"w{layer_idx}.weight"]
            b = params[f"w{layer_idx}.bias"]

            W = W.squeeze(-1)

            x2 = torch.einsum("b i s , b o i -> b o s", x, W) + b.unsqueeze(-1)

            x = x1 + x2

            if layer_idx < 3:
                x = F.gelu(x)

        x = x.permute(0, 2, 1)

        W1 = params["fc1.weight"]
        b1 = params["fc1.bias"]

        x = torch.einsum("bsw, baw -> bsa", x, W1) + b1.unsqueeze(1)
        x = F.gelu(x)

        W2 = params["fc2.weight"]
        b2 = params["fc2.bias"]

        x = torch.einsum("bsf, btf -> bst", x, W2) + b2.unsqueeze(1)

        return x   # [B, S, time_future]
