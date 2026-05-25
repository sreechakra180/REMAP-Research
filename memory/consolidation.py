import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans
import warnings

class MemoryConsolidation(nn.Module):
    def __init__(self, state_dim, compressed_dim, n_prototypes=100):
        super().__init__()
        self.state_dim = state_dim
        self.compressed_dim = compressed_dim
        self.n_prototypes = n_prototypes
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, compressed_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        )
        
        self.register_buffer('prototypes', torch.randn(n_prototypes, compressed_dim))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.kmeans = MiniBatchKMeans(n_clusters=n_prototypes, n_init=1, random_state=42)
            
        self.is_kmeans_init = False
        
    def consolidate(self, episodic_buffer):
        if len(episodic_buffer) == 0:
            return None
            
        device = next(self.parameters()).device
        states = []
        for entry in episodic_buffer:
            s = entry['state']
            if not isinstance(s, torch.Tensor):
                s = torch.tensor(s, dtype=torch.float32)
            states.append(s)
            
        states_tensor = torch.stack(states).to(device)
        
        with torch.no_grad():
            compressed = self.encoder(states_tensor)
            
        compressed_np = compressed.cpu().numpy()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not self.is_kmeans_init:
                self.kmeans.fit(compressed_np)
                self.is_kmeans_init = True
            else:
                self.kmeans.partial_fit(compressed_np)
            
        self.prototypes.data = torch.tensor(self.kmeans.cluster_centers_, dtype=torch.float32, device=device)
        
        return compressed
        
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
