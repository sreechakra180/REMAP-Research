import torch
import random

class CompressedMemoryStore:
    def __init__(self, dim, compressed_dim, capacity=5000):
        self.dim = dim
        self.compressed_dim = compressed_dim
        self.capacity = capacity
        
        self.memory = []
        
        # Projection matrix for compression
        self.proj_matrix = torch.randn(dim, compressed_dim) / (compressed_dim ** 0.5)
        self.inv_proj_matrix = torch.linalg.pinv(self.proj_matrix)
        
    def store(self, representations):
        if isinstance(representations, list):
            representations = torch.stack(representations)
            
        with torch.no_grad():
            compressed = torch.matmul(representations.cpu(), self.proj_matrix)
            
        for i in range(compressed.size(0)):
            if len(self.memory) >= self.capacity:
                self.memory.pop(0)
            self.memory.append(compressed[i])
            
    def retrieve(self, query=None, top_k=None):
        if len(self.memory) == 0:
            return torch.empty(0, self.dim)
            
        if top_k is None or top_k >= len(self.memory):
            compressed = torch.stack(self.memory)
        else:
            sampled_indices = random.sample(range(len(self.memory)), top_k)
            compressed = torch.stack([self.memory[i] for i in sampled_indices])
            
        decompressed = torch.matmul(compressed, self.inv_proj_matrix)
        return decompressed
