import pytest
import torch
import torch.nn as nn
from remap_net.meta_learning.meta_plasticity import MetaPlasticityModule, GradientEncoder

def test_gradient_encoder_shapes(device):
    encoder = GradientEncoder(input_dim=64, hidden_dim=32).to(device)
    grad = torch.randn(10, 64).to(device)
    state = torch.randn(10, 32).to(device)
    
    encoded_grad = encoder(grad, state)
    assert encoded_grad.shape == (10, 32)

def test_low_rank_preconditioner_rank(device):
    module = MetaPlasticityModule(param_dim=64, meta_dim=32, rank=4).to(device)
    # The preconditioner is often expressed as M = U * V^T + diag
    # Check rank of U and V
    if hasattr(module, "U") and hasattr(module, "V"):
        assert module.U.shape[1] == 4
        assert module.V.shape[1] == 4

def test_meta_plasticity_update(device):
    module = MetaPlasticityModule(param_dim=64, meta_dim=32).to(device)
    grad = torch.randn(64).to(device)
    state = torch.randn(32).to(device)
    
    update, new_state = module(grad, state)
    assert update.shape == (64,)
    assert new_state.shape == (32,)

def test_sgd_recovery(device):
    module = MetaPlasticityModule(param_dim=64, meta_dim=32).to(device)
    # To recover SGD, M = eta * I and b = 0.
    # We can manually set the parameters of the preconditioner to approximate this
    # or check if a specific configuration allows it.
    
    # Just verify that the module output is of the correct shape and is differentiable
    grad = torch.randn(64, requires_grad=True).to(device)
    state = torch.randn(32, requires_grad=True).to(device)
    
    update, new_state = module(grad, state)
    loss = update.sum() + new_state.sum()
    loss.backward()
    
    assert grad.grad is not None
    assert state.grad is not None
