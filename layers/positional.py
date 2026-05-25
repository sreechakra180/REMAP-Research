import torch
import torch.nn as nn
import math
from typing import Tuple

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed Sinusoidal Positional Encoding.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Positional encoding added to input
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class LearnedPositionalEncoding(nn.Module):
    """
    Learnable Positional Encoding.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        return x + self.pe(positions).unsqueeze(0)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute the frequency tensor for complex exponentials (RoPE)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape frequency tensor for broadcasting over batch and heads."""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor, 
    xk: torch.Tensor, 
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Positional Embedding to queries and keys.
    xq, xk: (batch, seq_len, n_heads, head_dim)
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE).
    
    Computes rotations on query and key vectors.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        freqs_cis = precompute_freqs_cis(dim, max_seq_len, base)
        self.register_buffer('freqs_cis', freqs_cis)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: Queries of shape (batch, seq_len, n_heads, head_dim)
            k: Keys of shape (batch, seq_len, n_heads, head_dim)
        Returns:
            Rotated queries and keys
        """
        seq_len = q.size(1)
        freqs_cis = self.freqs_cis[:seq_len]
        return apply_rotary_emb(q, k, freqs_cis)
