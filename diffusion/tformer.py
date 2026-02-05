import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

# --- Previously Defined Building Blocks ---

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) implementation for PyTorch.
    """
    def __init__(self, embedding_size: int):
        super().__init__()
        self.embedding_size = embedding_size
        half_dim = embedding_size // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        """
        Apply rotary positional embedding to input.
        x: (..., seq_len, head_dim)
        """
        seq_len = x.shape[-2]
        device = x.device
        dtype = x.dtype
        
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = torch.cos(emb)
        sin = torch.sin(emb)
        
        # Apply rotation
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1).reshape(x.shape)
        return x * cos[..., None, :] + rotated * sin[..., None, :]


class MultiheadAttention(nn.Module):
    """
    An implementation of Multi-head Attention that applies Rotary Positional
    Embeddings (RoPE) to the query and key vectors.
    """
    def __init__(
        self,
        num_heads: int,
        num_channels: int,
        rope: RotaryPositionalEmbedding,
        *,
        key=None,
    ):
        super().__init__()
        if num_channels % num_heads != 0:
            raise ValueError(f"num_channels ({num_channels}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.num_channels = num_channels
        self.rope = rope
        
        self.query_proj = nn.Linear(num_channels, num_channels, bias=False)
        self.key_proj = nn.Linear(num_channels, num_channels, bias=False)
        self.value_proj = nn.Linear(num_channels, num_channels, bias=False)
        self.output_proj = nn.Linear(num_channels, num_channels, bias=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        seq_len, channels = x.shape
        head_dim = channels // self.num_heads

        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)

        # Reshape for multi-head attention
        query = rearrange(query, 's (h d) -> h s d', h=self.num_heads)
        key = rearrange(key, 's (h d) -> h s d', h=self.num_heads)
        value = rearrange(value, 's (h d) -> h s d', h=self.num_heads)

        # Apply Rotary Positional Embeddings
        query = self.rope(query)
        key = self.rope(key)
        
        # Perform attention
        # PyTorch's scaled_dot_product_attention expects (batch, num_heads, seq_len, head_dim)
        # We have (num_heads, seq_len, head_dim), so we add batch dimension and then squeeze
        query = query.unsqueeze(0)  # (1, num_heads, seq_len, head_dim)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)
        
        results = F.scaled_dot_product_attention(query, key, value)
        results = results.squeeze(0)  # (num_heads, seq_len, head_dim)
        
        # Combine heads
        results = rearrange(results, 'h s d -> s (h d)')
        
        out = self.output_proj(results)
        return out

class SwiGLU(nn.Module):
    """
    A SwiGLU-based feed-forward network.
    """
    def __init__(self, num_channels: int, widening_factor: int, *, key=None):
        super().__init__()
        hidden_dim = num_channels * widening_factor
        
        self.gate_proj = nn.Linear(num_channels, hidden_dim, bias=False)
        self.up_proj = nn.Linear(num_channels, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, num_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated_x = F.silu(self.gate_proj(x))
        up_x = self.up_proj(x)
        fused_x = gated_x * up_x
        out = self.down_proj(fused_x)
        return out

class TransformerBlock(nn.Module):
    """
    A Transformer block using Pre-LN, SwiGLU, and Rotary Positional Embeddings.
    """
    def __init__(self, num_heads: int, num_channels: int, ff_widening_factor: int, *, key=None):
        super().__init__()
        
        head_dim = num_channels // num_heads
        rope = RotaryPositionalEmbedding(embedding_size=head_dim)
        
        self.pre_attn_norm = nn.LayerNorm(num_channels)
        self.attention = MultiheadAttention(num_heads, num_channels, rope=rope, key=key)
        self.pre_ff_norm = nn.LayerNorm(num_channels)
        self.ff_layer = SwiGLU(num_channels, ff_widening_factor, key=key)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.pre_attn_norm(x)
        x = x + self.attention(x_norm)

        x_norm = self.pre_ff_norm(x)
        x = x + self.ff_layer(x_norm)
        
        return x

def sinusoidal_time_embedding(time: float, embedding_dim: int, max_period: int = 10000):
    if embedding_dim % 2 != 0:
        raise ValueError(f"Embedding dimension {embedding_dim} must be even.")

    half_dim = embedding_dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half_dim) / half_dim)
    args = time * freqs
    return torch.concatenate([torch.sin(args), torch.cos(args)])


class PatchifyAndEmbed(nn.Module):
    """
    Handles patchification, projection, and time embedding for a multi-channel image.
    """
    def __init__(
        self, model_embedding_dim: int, patch_size: int, num_channels: int, *, key=None,
    ):
        super().__init__()
        raw_patch_dim = num_channels * patch_size * patch_size
        self.patch_proj = nn.Linear(raw_patch_dim, model_embedding_dim, bias=True)
        self.model_embedding_dim = model_embedding_dim
        self.patch_size = patch_size

    def forward(self, t: float, x: torch.Tensor):
        patches = rearrange(
            x, 'c (h p1) (w p2) -> (h w) (p1 p2 c)',
            p1=self.patch_size, p2=self.patch_size,
        )
        projected_patches = self.patch_proj(patches)
        time_embedding = sinusoidal_time_embedding(t, self.model_embedding_dim).to(projected_patches.device)
        final_embeddings = projected_patches + time_embedding
        return final_embeddings

# --- New Top-Level Model ---

class TransformerModel(nn.Module):
    """
    A complete Transformer-based model for image processing tasks, conditioned on time.

    This model takes a multi-channel image and a time float, processes them through
    an embedding layer, a series of Transformer blocks, and finally projects the
    output back into an image format.
    """
    # Configuration
    patch_size: int = 16
    
    def __init__(
        self,
        num_blocks: int = 16,
        num_heads: int = 16,
        ff_widening_factor: int = 3,
        *,
        key=None,
    ):
        """
        Initializes the full Transformer model.

        Args:
            num_blocks: The number of sequential Transformer blocks (model depth).
            num_heads: The number of attention heads in each Transformer block.
            ff_widening_factor: The widening factor for the SwiGLU FFNs.
            key: A PyTorch generator for reproducible initialization (optional).
        """
        super().__init__()
        
        # --- Define layer dimensions based on architecture ---
        # Input image is (2, 256, 256)
        input_channels = 2
        initial_embedding_dim = 512
        model_dim = 1024 # The main dimension for the transformer blocks
        # The final output has 1 channel. Patch features = 16*16*1 = 256
        output_patch_features = self.patch_size * self.patch_size * 1 

        # 1. Input Embedding Layer
        self.patch_embed = PatchifyAndEmbed(
            model_embedding_dim=initial_embedding_dim,
            patch_size=self.patch_size,
            num_channels=input_channels,
            key=key
        )

        # 2. Up-Projection Layer (512 -> 1024)
        self.up_proj = nn.Linear(initial_embedding_dim, model_dim)
        
        # 3. Core Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                num_heads=num_heads,
                num_channels=model_dim,
                ff_widening_factor=ff_widening_factor,
                key=key,
            )
            for _ in range(num_blocks)
        ])
        
        # 4. Down-Projection Layer (1024 -> 256)
        self.down_proj = nn.Linear(model_dim, output_patch_features)

    def forward(
        self, t: float, x: torch.Tensor, *, key=None
    ) -> torch.Tensor:
        """
        Performs the forward pass of the model.

        Args:
            t: A scalar time value (typically in [0, 1]).
            x: The input image array of shape (2, 256, 256).
            key: A generator (optional, not used in this forward pass).

        Returns:
            The processed output image of shape (1, 256, 256).
        """
        # 1. Get initial time-aware embeddings
        # (2, 256, 256) -> (256, 512)
        x = self.patch_embed(t, x)
        
        # 2. Up-project to model dimension and apply GeLU
        # (256, 512) -> (256, 1024)
        x = F.gelu(self.up_proj(x))
        
        # 3. Pass through the stack of Transformer blocks
        # (256, 1024) -> (256, 1024)
        for block in self.transformer_blocks:
            x = block(x)
            
        # 4. Down-project to output patch dimension and apply GeLU
        # (256, 1024) -> (256, 256)
        #x = F.gelu(self.down_proj(x))
        x = self.down_proj(x)
        
        # 5. "Un-patch" the sequence back into an image
        # (256, 256) -> (1, 256, 256)
        # We rearrange the sequence of patches (h*w) each with (p1*p2*c) features
        # back into a final image of shape (c, h*p1, w*p2).
        output = rearrange(
            x, 
            '(h w) (p1 p2 c) -> c (h p1) (w p2)', 
            h=(256 // self.patch_size), 
            w=(256 // self.patch_size), 
            p1=self.patch_size, 
            p2=self.patch_size,
            c=1
        )
        
        return output
