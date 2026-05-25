import pytest
import torch
import torch.nn as nn
from remap_net.models import REMAPNet, TransformerBackbone, MLPBackbone, ResNetBackbone
from remap_net.meta_learning.meta_plasticity import MetaPlasticityModule
from remap_net.meta_learning.epistemic_recursion import EpistemicRecursionModule
from remap_net.stability import StabilityGuardian
from remap_net.abstraction import InformationBottleneck
from remap_net.memory import EpisodicMemory

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def small_config():
    return {
        "input_dim": 16,
        "hidden_dim": 32,
        "output_dim": 10,
        "num_layers": 2,
        "num_heads": 2,
        "seq_len": 8,
        "meta_dim": 16,
        "memory_capacity": 100,
        "beta": 1.0,
        "learning_rate": 0.001
    }

@pytest.fixture
def sample_batch(small_config, device):
    batch_size = 4
    x = torch.randn(batch_size, small_config["seq_len"], small_config["input_dim"], device=device)
    y = torch.randint(0, small_config["output_dim"], (batch_size,), device=device)
    return x, y

@pytest.fixture
def remap_model(small_config, device):
    # Depending on exact constructor, mock it or use standard params
    try:
        model = REMAPNet(
            input_dim=small_config["input_dim"],
            hidden_dim=small_config["hidden_dim"],
            output_dim=small_config["output_dim"],
            backbone_type="transformer"
        ).to(device)
    except TypeError:
        # Fallback if config is passed as dict
        model = REMAPNet(small_config).to(device)
    return model

@pytest.fixture
def simple_linear_model(device):
    return nn.Linear(16, 10).to(device)
