import pytest
import torch
import numpy as np
import random
from remap_net.models import REMAPNet

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def test_seed_determinism(device, small_config):
    set_seed(42)
    try:
        model1 = REMAPNet(
            input_dim=small_config["input_dim"],
            hidden_dim=small_config["hidden_dim"],
            output_dim=small_config["output_dim"]
        ).to(device)
    except TypeError:
        model1 = REMAPNet(small_config).to(device)
        
    x = torch.randn(2, small_config["seq_len"], small_config["input_dim"]).to(device)
    out1 = model1(x)
    
    set_seed(42)
    try:
        model2 = REMAPNet(
            input_dim=small_config["input_dim"],
            hidden_dim=small_config["hidden_dim"],
            output_dim=small_config["output_dim"]
        ).to(device)
    except TypeError:
        model2 = REMAPNet(small_config).to(device)
        
    x2 = torch.randn(2, small_config["seq_len"], small_config["input_dim"]).to(device)
    out2 = model2(x2)
    
    assert torch.allclose(out1, out2)

def test_config_snapshot(tmp_path):
    import json
    config = {
        "input_dim": 16,
        "hidden_dim": 32,
        "output_dim": 10
    }
    
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
        
    with open(config_path, "r") as f:
        loaded_config = json.load(f)
        
    assert loaded_config == config
