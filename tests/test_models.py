import pytest
import torch
import torch.nn as nn
from remap_net.models import TransformerBackbone, MLPBackbone, ResNetBackbone, REMAPNet
from torch.cuda.amp import autocast

def test_transformer_backbone_forward(device, small_config):
    model = TransformerBackbone(
        input_dim=small_config["input_dim"],
        hidden_dim=small_config["hidden_dim"],
        num_layers=small_config["num_layers"],
        num_heads=small_config["num_heads"]
    ).to(device)
    
    x = torch.randn(4, small_config["seq_len"], small_config["input_dim"]).to(device)
    out = model(x)
    assert out.shape == (4, small_config["seq_len"], small_config["hidden_dim"])

def test_transformer_backbone_backward(device, small_config):
    model = TransformerBackbone(
        input_dim=small_config["input_dim"],
        hidden_dim=small_config["hidden_dim"],
        num_layers=small_config["num_layers"],
        num_heads=small_config["num_heads"]
    ).to(device)
    
    x = torch.randn(4, small_config["seq_len"], small_config["input_dim"]).to(device)
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    for param in model.parameters():
        assert param.grad is not None

def test_mlp_backbone_forward_backward(device, small_config):
    model = MLPBackbone(
        input_dim=small_config["input_dim"],
        hidden_dim=small_config["hidden_dim"],
        num_layers=small_config["num_layers"]
    ).to(device)
    
    # MLP expects 2D or flattened seq
    x = torch.randn(4, small_config["input_dim"]).to(device)
    out = model(x)
    assert out.shape == (4, small_config["hidden_dim"])
    
    loss = out.sum()
    loss.backward()
    for param in model.parameters():
        assert param.grad is not None

def test_resnet_backbone_forward_backward(device):
    model = ResNetBackbone(
        in_channels=3,
        hidden_dim=32,
        num_layers=2
    ).to(device)
    
    x = torch.randn(4, 3, 32, 32).to(device)
    out = model(x)
    assert len(out.shape) == 2
    assert out.shape[1] == 32
    
    loss = out.sum()
    loss.backward()
    for param in model.parameters():
        assert param.grad is not None

def test_remap_net_creation(device, small_config):
    try:
        model = REMAPNet(
            input_dim=small_config["input_dim"],
            hidden_dim=small_config["hidden_dim"],
            output_dim=small_config["output_dim"]
        ).to(device)
    except TypeError:
        model = REMAPNet(small_config).to(device)
    
    assert model is not None
    assert isinstance(model, nn.Module)

def test_mixed_precision(device, small_config):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for mixed precision test")
        
    try:
        model = REMAPNet(
            input_dim=small_config["input_dim"],
            hidden_dim=small_config["hidden_dim"],
            output_dim=small_config["output_dim"]
        ).to(device)
    except TypeError:
        model = REMAPNet(small_config).to(device)
        
    x = torch.randn(4, small_config["seq_len"], small_config["input_dim"]).to(device)
    
    with autocast():
        out = model(x)
        assert out.dtype in (torch.float16, torch.bfloat16)
