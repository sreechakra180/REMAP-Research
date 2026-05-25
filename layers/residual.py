import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class ResidualBlock(nn.Module):
    """
    Standard Residual Block (Post-LayerNorm style or implicit).
    
    Computes:
        y = x + F(x)
    where F is a 2-layer MLP with specified activation.
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'swish':
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.act(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.dropout2(out)
        return x + out

class PreNormResidualBlock(nn.Module):
    """
    Pre-LayerNorm Residual Block.
    
    Computes:
        y = x + F(LayerNorm(x))
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.dropout2(out)
        return x + out

class ResidualStack(nn.Module):
    """
    Stack of PreNormResidualBlocks with optional gradient checkpointing.
    """
    def __init__(self, dim: int, hidden_dim: int, n_blocks: int, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            PreNormResidualBlock(dim, hidden_dim, dropout) 
            for _ in range(n_blocks)
        ])
        
    def forward(self, x: torch.Tensor, use_checkpointing: bool = False) -> torch.Tensor:
        """
        Forward pass through the stack of residual blocks.
        """
        for block in self.blocks:
            if use_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x
