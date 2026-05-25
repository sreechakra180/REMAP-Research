import pytest
import torch
import torch.nn as nn
from remap_net.models import REMAPNet

def test_training_step_phase1(remap_model, sample_batch):
    x, y = sample_batch
    
    if hasattr(remap_model, "training_step_phase1"):
        loss = remap_model.training_step_phase1(x, y)
        assert loss.dim() == 0
        assert loss.requires_grad

def test_training_step_phase2(remap_model, sample_batch):
    x, y = sample_batch
    
    if hasattr(remap_model, "training_step_phase2"):
        loss = remap_model.training_step_phase2(x, y)
        assert loss.dim() == 0
        assert loss.requires_grad

def test_training_step_phase3(remap_model, sample_batch):
    x, y = sample_batch
    
    if hasattr(remap_model, "training_step_phase3"):
        loss = remap_model.training_step_phase3(x, y)
        assert loss.dim() == 0
        assert loss.requires_grad

def test_loss_computation(remap_model, sample_batch):
    x, y = sample_batch
    # A standard forward pass
    out = remap_model(x)
    
    # If the model handles sequential outputs, get the last step or use specific loss
    if out.dim() == 3:
        out = out[:, -1, :]
        
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out, y)
    
    assert loss.dim() == 0
    loss.backward()
    
    # Check if gradients flow
    has_grad = False
    for param in remap_model.parameters():
        if param.grad is not None:
            has_grad = True
            break
            
    assert has_grad
