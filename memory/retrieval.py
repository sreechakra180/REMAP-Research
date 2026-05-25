import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AssociativeRetrieval(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = key_dim // n_heads
        
        assert self.head_dim * n_heads == key_dim, "key_dim must be divisible by n_heads"
        
        self.q_proj = nn.Linear(query_dim, key_dim)
        self.k_proj = nn.Linear(key_dim, key_dim)
        self.v_proj = nn.Linear(value_dim, value_dim)
        
        self.out_proj = nn.Linear(value_dim, value_dim)
        
    def forward(self, query, memory_keys, memory_values):
        batch_size = query.size(0)
        seq_len = memory_keys.size(1)
        
        q = self.q_proj(query).view(batch_size, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory_keys).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory_values).view(batch_size, seq_len, self.n_heads, self.v_proj.out_features // self.n_heads).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1)
        
        out = self.out_proj(context)
        return out
