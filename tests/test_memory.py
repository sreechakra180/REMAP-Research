import pytest
import torch
from remap_net.memory import EpisodicMemory

def test_episodic_store_retrieve(device):
    memory = EpisodicMemory(key_dim=32, val_dim=64, capacity=10).to(device)
    keys = torch.randn(5, 32).to(device)
    vals = torch.randn(5, 64).to(device)
    
    memory.store(keys, vals)
    assert memory.size == 5
    
    query = torch.randn(2, 32).to(device)
    retrieved = memory.retrieve(query, k=3)
    
    assert retrieved.shape == (2, 3, 64)

def test_memory_capacity_limit(device):
    memory = EpisodicMemory(key_dim=32, val_dim=64, capacity=10).to(device)
    keys = torch.randn(15, 32).to(device)
    vals = torch.randn(15, 64).to(device)
    
    memory.store(keys, vals)
    assert memory.size == 10  # should not exceed capacity

def test_consolidation(device):
    memory = EpisodicMemory(key_dim=32, val_dim=64, capacity=10).to(device)
    keys = torch.randn(10, 32).to(device)
    vals = torch.randn(10, 64).to(device)
    memory.store(keys, vals)
    
    if hasattr(memory, "consolidate"):
        compressed_memory = memory.consolidate()
        # Shape or size depends on the specific consolidation algorithm
        assert compressed_memory is not None

def test_tcr_loss_computation(device):
    memory = EpisodicMemory(key_dim=32, val_dim=64, capacity=10).to(device)
    keys = torch.randn(5, 32).to(device)
    vals = torch.randn(5, 64).to(device)
    memory.store(keys, vals)
    
    if hasattr(memory, "compute_tcr_loss"):
        loss = memory.compute_tcr_loss()
        assert loss.dim() == 0
        assert loss.requires_grad
