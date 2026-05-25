import torch
from remap_net.layers import (
    MultiHeadAttention, CrossAttention,
    ResidualBlock, PreNormResidualBlock, ResidualStack,
    RMSNorm, AdaptiveLayerNorm,
    SinusoidalPositionalEncoding, LearnedPositionalEncoding, RotaryPositionalEncoding
)

def test_layers():
    print("Testing Attention Layers...")
    mha = MultiHeadAttention(d_model=64, n_heads=8)
    x = torch.randn(2, 10, 64)
    out = mha(x)
    assert out.shape == (2, 10, 64)
    
    ca = CrossAttention(d_model=64, n_heads=8)
    ctx = torch.randn(2, 20, 64)
    out = ca(x, ctx)
    assert out.shape == (2, 10, 64)
    
    print("Testing Residual Layers...")
    res = ResidualBlock(64, 128)
    out = res(x)
    assert out.shape == (2, 10, 64)
    
    res_stack = ResidualStack(64, 128, 2)
    out = res_stack(x)
    assert out.shape == (2, 10, 64)
    
    print("Testing Normalization Layers...")
    rms = RMSNorm(64)
    out = rms(x)
    assert out.shape == (2, 10, 64)
    
    adaln = AdaptiveLayerNorm(64, 32)
    cond = torch.randn(2, 32)
    out = adaln(x, cond)
    assert out.shape == (2, 10, 64)
    
    print("Testing Positional Layers...")
    sin_pe = SinusoidalPositionalEncoding(64)
    out = sin_pe(x)
    assert out.shape == (2, 10, 64)
    
    lrn_pe = LearnedPositionalEncoding(64)
    out = lrn_pe(x)
    assert out.shape == (2, 10, 64)
    
    rope = RotaryPositionalEncoding(8)
    q = torch.randn(2, 10, 8, 8)
    k = torch.randn(2, 10, 8, 8)
    q_out, k_out = rope(q, k)
    assert q_out.shape == (2, 10, 8, 8)
    assert k_out.shape == (2, 10, 8, 8)

    print("All tests passed successfully!")

if __name__ == '__main__':
    test_layers()
