import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (MHA) block with optional Flash Attention support.
    
    Computes:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, use_flash: bool = True):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
            
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        self.use_flash = use_flash
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        use_checkpointing: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            use_checkpointing: Whether to use gradient checkpointing
        """
        if use_checkpointing and self.training:
            return checkpoint(self._forward_impl, x, mask, use_reentrant=False)
        return self._forward_impl(x, mask)
        
    def _forward_impl(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=mask, 
                dropout_p=self.dropout if self.training else 0.0
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
            if mask is not None:
                if mask.dtype == torch.bool:
                    scores = scores.masked_fill(~mask, float('-inf'))
                else:
                    scores = scores + mask
                    
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout_layer(attn)
            out = torch.matmul(attn, v)
            
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    """
    Cross-Attention block for memory retrieval.
    
    Queries come from x, Keys and Values come from context.
    Computes:
        Attention(Q_x, K_ctx, V_ctx) = softmax(Q_x K_ctx^T / sqrt(d_k)) V_ctx
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
            
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        context: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        use_checkpointing: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Query input tensor (batch_size, q_len, d_model)
            context: Key/Value input tensor (batch_size, kv_len, d_model)
            mask: Optional attention mask
            use_checkpointing: Whether to use gradient checkpointing
        """
        if use_checkpointing and self.training:
            return checkpoint(self._forward_impl, x, context, mask, use_reentrant=False)
        return self._forward_impl(x, context, mask)
        
    def _forward_impl(self, x: torch.Tensor, context: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, q_len, _ = x.size()
        kv_len = context.size(1)
        
        q = self.q_proj(x).view(batch_size, q_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(context).view(batch_size, kv_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(context).view(batch_size, kv_len, self.n_heads, self.d_k).transpose(1, 2)
        
        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=mask, 
                dropout_p=self.dropout if self.training else 0.0
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
            if mask is not None:
                if mask.dtype == torch.bool:
                    scores = scores.masked_fill(~mask, float('-inf'))
                else:
                    scores = scores + mask
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout_layer(attn)
            out = torch.matmul(attn, v)
            
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        return self.out_proj(out)
