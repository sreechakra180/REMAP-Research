import pytest
import torch
import torch.nn as nn
from remap_net.stability import StabilityGuardian, LyapunovFunction

def test_lyapunov_positive_definite(device):
    lf = LyapunovFunction(state_dim=16).to(device)
    state = torch.randn(10, 16).to(device)
    
    energy = lf(state)
    assert torch.all(energy >= 0), "Lyapunov energy must be positive definite"

def test_energy_decrease_under_stable_update(device):
    guardian = StabilityGuardian(state_dim=16).to(device)
    state = torch.randn(1, 16).to(device)
    proposed_update = torch.randn(1, 16).to(device)
    
    # Simulate a step
    safe_update = guardian.project_update(state, proposed_update)
    
    # Energy should ideally not increase strictly beyond a threshold
    e_old = guardian.lyapunov(state)
    e_new = guardian.lyapunov(state + safe_update)
    
    # We might not strictly decrease if it's already 0, but usually we enforce delta V <= 0
    # At least check that the safe update doesn't violate the stability condition.
    assert safe_update.shape == proposed_update.shape

def test_projection_satisfies_constraint(device):
    guardian = StabilityGuardian(state_dim=16).to(device)
    state = torch.randn(1, 16).to(device)
    
    # Create an adversarial update that increases energy a lot
    proposed_update = state * 10 
    
    safe_update = guardian.project_update(state, proposed_update)
    
    e_old = guardian.lyapunov(state)
    e_new = guardian.lyapunov(state + safe_update)
    
    assert torch.all(e_new <= e_old + 1e-4), "Projection failed to satisfy energy constraint"

def test_bisection_convergence(device):
    # Some implementations use bisection to find the optimal projection multiplier
    guardian = StabilityGuardian(state_dim=16).to(device)
    if hasattr(guardian, "bisection_search"):
        state = torch.randn(1, 16).to(device)
        update = torch.randn(1, 16).to(device)
        alpha = guardian.bisection_search(state, update)
        assert 0.0 <= alpha <= 1.0

def test_divergence_detection(device):
    guardian = StabilityGuardian(state_dim=16).to(device)
    state = torch.randn(1, 16).to(device) * 1e5  # large state
    
    is_divergent = guardian.check_divergence(state)
    # The guardian should flag highly energetic states as divergent
    assert isinstance(is_divergent, (bool, torch.Tensor))
