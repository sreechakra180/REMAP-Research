import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Computes:
        y = (x / RMS(x)) * gamma
    where RMS(x) = sqrt( 1/d * sum(x_i^2) + eps )
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return x_normed * self.weight

class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN).
    
    Conditions normalization on an external signal (cond).
    Computes:
        y = LayerNorm(x) * (1 + gamma(cond)) + beta(cond)
    """
    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * dim)
        
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            cond: Conditioning tensor of shape (batch_size, cond_dim) or (batch_size, seq_len, cond_dim)
        """
        params = self.proj(cond)
        
        if params.dim() == 2 and x.dim() == 3:
            params = params.unsqueeze(1)
            
        gamma, beta = params.chunk(2, dim=-1)
        
        x_norm = self.norm(x)
        return x_norm * (1.0 + gamma) + beta
