import pytest
import torch
import torch.nn as nn
from remap_net.abstraction import InformationBottleneck, MINEEstimator

def test_information_bottleneck_loss(device):
    ib = InformationBottleneck(input_dim=32, hidden_dim=16).to(device)
    x = torch.randn(10, 32).to(device)
    y = torch.randn(10, 16).to(device) # target or similar
    
    z, mu, logvar = ib(x)
    assert z.shape == (10, 16)
    
    loss = ib.compute_loss(x, z, mu, logvar)
    assert loss.dim() == 0 # scalar
    assert loss.requires_grad

def test_mine_estimator(device):
    mine = MINEEstimator(x_dim=32, z_dim=16, hidden_dim=32).to(device)
    x = torch.randn(10, 32).to(device)
    z = torch.randn(10, 16).to(device)
    
    # Mutual information estimation
    mi = mine(x, z)
    assert mi.dim() == 0
    assert mi.requires_grad
    
    # Test with shuffled z to approximate marginal
    z_shuffle = z[torch.randperm(10)]
    mi_shuffled = mine(x, z_shuffle)
    # Typically MI with matched pairs is higher than with shuffled pairs, though not guaranteed in a single random batch
    assert mi_shuffled.dim() == 0

def test_abstraction_quality_metric(device):
    ib = InformationBottleneck(input_dim=32, hidden_dim=16).to(device)
    x = torch.randn(10, 32).to(device)
    z, mu, logvar = ib(x)
    
    if hasattr(ib, "abstraction_quality"):
        quality = ib.abstraction_quality(x, z)
        assert isinstance(quality, float) or quality.dim() == 0
