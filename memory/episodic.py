import torch
import numpy as np
import random

class EpisodicMemory:
    def __init__(self, capacity=10000, state_dim=256):
        self.capacity = capacity
        self.state_dim = state_dim
        self.buffer = []
        
    def store(self, state, task_id, gradient, timestamp):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        
        entry = {
            'state': state.cpu().detach() if isinstance(state, torch.Tensor) else state,
            'task_id': task_id,
            'gradient': gradient.cpu().detach() if isinstance(gradient, torch.Tensor) else gradient,
            'timestamp': timestamp
        }
        self.buffer.append(entry)
        
    def sample(self, batch_size, strategy='uniform'):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        if strategy == 'uniform':
            samples = random.sample(self.buffer, batch_size)
        elif strategy == 'priority':
            weights = np.linspace(0.1, 1.0, len(self.buffer))
            weights = weights / weights.sum()
            indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=weights)
            samples = [self.buffer[i] for i in indices]
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
            
        return samples
        
    def get_recent(self, K):
        K = min(K, len(self.buffer))
        return self.buffer[-K:] if K > 0 else []
        
    def get_context(self, query, top_k=5):
        if len(self.buffer) == 0:
            return []
            
        top_k = min(top_k, len(self.buffer))
        
        if isinstance(query, torch.Tensor):
            query = query.cpu().detach().numpy()
            
        states = np.stack([entry['state'].numpy() if isinstance(entry['state'], torch.Tensor) else entry['state'] for entry in self.buffer])
        # Simple dot product similarity
        similarities = np.dot(states, query.T).flatten()
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.buffer[i] for i in top_indices]
        
    def __len__(self):
        return len(self.buffer)
