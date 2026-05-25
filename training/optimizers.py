import torch
import torch.optim as optim

def build_optimizer(params, config):
    """Builds a standard optimizer."""
    opt_type = config.get('optimizer', 'Adam')
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 0.0)
    
    if opt_type == 'Adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_type == 'AdamW':
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif opt_type == 'SGD':
        momentum = config.get('momentum', 0.9)
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_type}")

def build_meta_optimizer(model, config):
    """Builds separate optimizers for theta, phi, and psi parameters."""
    
    # Group parameters
    theta_params = []
    phi_params = []
    psi_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'plasticity' in name or 'f1' in name:
            phi_params.append(param)
        elif 'stability' in name or 'f2' in name:
            psi_params.append(param)
        else:
            theta_params.append(param)
            
    # Build optimizers
    theta_opt = build_optimizer(theta_params, config.get('theta_optim', {'optimizer': 'AdamW', 'lr': 1e-3}))
    phi_opt = build_optimizer(phi_params, config.get('phi_optim', {'optimizer': 'Adam', 'lr': 1e-4}))
    psi_opt = build_optimizer(psi_params, config.get('psi_optim', {'optimizer': 'Adam', 'lr': 1e-5}))
    
    return theta_opt, phi_opt, psi_opt
