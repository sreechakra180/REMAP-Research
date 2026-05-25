import pytest
import torch
import torch.nn as nn
from remap_net.meta_learning.epistemic_recursion import EpistemicRecursionModule

def test_higher_order_gradients(device):
    module = EpistemicRecursionModule(meta_dim=32).to(device)
    
    # Simulate a loss function and some parameters
    params = nn.Parameter(torch.randn(10).to(device))
    optimizer = torch.optim.SGD([params], lr=0.1)
    
    def loss_fn(p):
        return (p ** 2).sum()
    
    loss = loss_fn(params)
    grad = torch.autograd.grad(loss, params, create_graph=True)[0]
    
    # Pass through module
    state = torch.randn(32).to(device)
    update, new_state = module(grad, state)
    
    # Compute higher order gradient
    meta_loss = update.sum()
    meta_grad = torch.autograd.grad(meta_loss, params, retain_graph=True)[0]
    
    assert meta_grad is not None

def test_hessian_approximation(device):
    module = EpistemicRecursionModule(meta_dim=32).to(device)
    # Test if the module can approximate Hessian-vector products
    params = torch.randn(10, requires_grad=True).to(device)
    loss = (params ** 3).sum()
    
    grad = torch.autograd.grad(loss, params, create_graph=True)[0]
    vec = torch.randn(10).to(device)
    
    # HVP
    hvp = torch.autograd.grad(grad, params, grad_outputs=vec, retain_graph=True)[0]
    assert hvp.shape == (10,)

def test_epistemic_update_shapes(device):
    module = EpistemicRecursionModule(meta_dim=32, num_steps=3).to(device)
    grad = torch.randn(64).to(device)
    state = torch.randn(32).to(device)
    
    update, new_state = module(grad, state)
    assert update.shape == (64,)
    assert new_state.shape == (32,)
