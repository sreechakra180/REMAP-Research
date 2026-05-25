import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any
from torch.utils.checkpoint import checkpoint

from ..layers import MultiHeadAttention, RMSNorm, SinusoidalPositionalEncoding

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.norm1 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN architecture
        attn_out, _ = self.attn(self.norm1(x), mask=mask)
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        return x

class TransformerBackbone(nn.Module):
    """
    F0 Object Layer implementation for Sequence/Text tasks.
    Uses custom MultiHeadAttention, RMSNorm, and SinusoidalPositionalEncoding.
    """
    def __init__(self, d_model: int = 256, n_heads: int = 8, n_layers: int = 6, 
                 d_ff: int = 1024, dropout: float = 0.1, max_seq_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.gradient_checkpointing = False
        
        self.pos_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_seq_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                return_intermediates: bool = False) -> Any:
        x = self.pos_encoding(x)
        
        intermediates = []
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # use_reentrant=False is recommended for PyTorch 2.x
                x = checkpoint(layer, x, mask, use_reentrant=False)
            else:
                x = layer(x, mask)
                
            if return_intermediates:
                intermediates.append(x)
                
        x = self.norm(x)
        
        if return_intermediates:
            return x, intermediates
        return x

class MLPBackbone(nn.Module):
    """
    F0 Object Layer implementation with a simple feed-forward architecture
    with residual connections.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 256, 256], 
                 dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        self.gradient_checkpointing = False
        self.input_dim = input_dim
        
        acts = {
            'gelu': nn.GELU,
            'relu': nn.ReLU,
            'swish': nn.SiLU,
            'silu': nn.SiLU
        }
        act_layer = acts.get(activation.lower(), nn.GELU)
        
        self.layers = nn.ModuleList()
        in_dim = input_dim
        for h_dim in hidden_dims:
            block = nn.Sequential(
                nn.Linear(in_dim, h_dim),
                RMSNorm(h_dim),
                act_layer(),
                nn.Dropout(dropout)
            )
            self.layers.append(block)
            
            if in_dim != h_dim:
                self.layers.append(nn.Linear(in_dim, h_dim))
            else:
                self.layers.append(nn.Identity())
            in_dim = h_dim
            
        self.output_dim = in_dim
        
    def forward(self, x: torch.Tensor, return_intermediates: bool = False) -> Any:
        intermediates = []
        
        # self.layers contains [block, projection, block, projection, ...]
        for i in range(0, len(self.layers), 2):
            block = self.layers[i]
            proj = self.layers[i+1]
            
            residual = proj(x)
            
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
                
            x = x + residual
            
            if return_intermediates:
                intermediates.append(x)
                
        if return_intermediates:
            return x, intermediates
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetBackbone(nn.Module):
    """
    F0 Object Layer implementation for Image tasks using a 4-stage residual network.
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64, n_blocks: List[int] = [2, 2, 2, 2]):
        super().__init__()
        self.gradient_checkpointing = False
        self.in_planes = base_channels

        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, base_channels, n_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, base_channels*2, n_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, base_channels*4, n_blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, base_channels*8, n_blocks[3], stride=2)
        
        self.output_dim = base_channels * 8 * BasicBlock.expansion

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, return_intermediates: bool = False) -> Any:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        intermediates = []
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        
        for layer in layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
                
            if return_intermediates:
                intermediates.append(x)

        if return_intermediates:
            return x, intermediates
        return x
