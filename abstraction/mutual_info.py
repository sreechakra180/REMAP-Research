import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MINEEstimator(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, z):
        # Joint samples
        joint = torch.cat([x, z], dim=-1)
        t_joint = self.net(joint)
        
        # Marginal samples
        z_shuffled = z[torch.randperm(z.shape[0])]
        marginal = torch.cat([x, z_shuffled], dim=-1)
        t_marginal = self.net(marginal)
        
        # Lower bound on MI
        mi_lb = t_joint.mean() - torch.log(torch.exp(t_marginal).mean() + 1e-8)
        return mi_lb

class VariationalMIEstimator(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim=128):
        super().__init__()
        self.q_mu = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, x_dim)
        )
        self.q_logvar = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, x_dim)
        )
        
    def forward(self, x, z):
        mu = self.q_mu(z)
        logvar = self.q_logvar(z)
        
        var = torch.exp(logvar)
        log_q = -0.5 * torch.sum((x - mu)**2 / var + logvar + math.log(2 * math.pi), dim=1)
        # Returns expected log q(x|z), which forms a lower bound when added to H(X)
        return log_q.mean()

def compute_mi(x, z, method='mine', model=None):
    if model is None:
        return torch.tensor(0.0)
    return model(x, z)
