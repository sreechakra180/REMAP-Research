import torch
import torch.nn as nn
from .mutual_info import MINEEstimator

class InformationBottleneck(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, output_dim, beta=1.0, I_min=0.1):
        super().__init__()
        self.beta = beta
        self.I_min = I_min
        
        self.enc_mu = nn.Linear(input_dim, bottleneck_dim)
        self.enc_logvar = nn.Linear(input_dim, bottleneck_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        self.mi_estimator_xy = MINEEstimator(input_dim, bottleneck_dim)
        
    def encode(self, x):
        mu = self.enc_mu(x)
        logvar = self.enc_logvar(x)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x, y=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        
        ib_loss = None
        if y is not None:
            ib_loss = self.compute_ib_loss(x, z, mu, logvar, y, reconstruction)
            
        return z, reconstruction, ib_loss
        
    def compute_ib_loss(self, x, z, mu, logvar, y, y_pred):
        # min_g H(g(X)) - beta*I(g(X); Y) s.t. I(X; g(X)) >= I_min
        
        # D_KL(q(z|x) || p(z)) is an upper bound on I(X; Z)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        # I(Z;Y) using cross entropy as a proxy
        ce_loss = nn.CrossEntropyLoss()(y_pred, y)
        i_zy_approx = -ce_loss 
        
        # I(X; Z) approx via MI estimator
        i_xz = self.mi_estimator_xy(x, z)
        
        # Lagrangian multiplier for I_min constraint
        constraint_penalty = torch.relu(self.I_min - i_xz).pow(2)
        
        loss = kl_div - self.beta * i_zy_approx + 10.0 * constraint_penalty
        return loss
