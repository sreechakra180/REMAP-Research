import torch
import torch.nn as nn

class LatentAbstractionModule(nn.Module):
    def __init__(self, input_dim, abstraction_dims=[128, 64, 32], beta=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.abstraction_dims = abstraction_dims
        self.beta = beta
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        prev_dim = input_dim
        for dim in abstraction_dims:
            self.encoders.append(nn.Sequential(
                nn.Linear(prev_dim, dim),
                nn.ReLU()
            ))
            self.decoders.append(nn.Sequential(
                nn.Linear(dim, prev_dim),
                nn.ReLU()
            ))
            prev_dim = dim
            
    def forward(self, x):
        features = x
        outputs = []
        for enc, dec in zip(self.encoders, self.decoders):
            z_level = enc(features)
            recon = dec(z_level)
            outputs.append((z_level, recon))
            features = z_level
        return outputs
        
    def generate_hierarchy(self):
        return {
            "levels": len(self.abstraction_dims),
            "dims": self.abstraction_dims
        }
