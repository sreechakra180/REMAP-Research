import random
from collections import namedtuple
import numpy as np
import torch
from torch.utils.data import DataLoader

TaskBatch = namedtuple("TaskBatch", ["support_x", "support_y", "query_x", "query_y"])

def set_data_seed(seed: int):
    """Set the seed for data generation and sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EpisodeSampler:
    """Generic N-way K-shot sampler for few-shot learning."""
    def __init__(self, dataset, n_way: int, k_shot: int, q_query: int, n_episodes: int):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.n_episodes = n_episodes
        self.classes = list(self.dataset.class_to_indices.keys())
        
    def __iter__(self):
        for _ in range(self.n_episodes):
            sampled_classes = random.sample(self.classes, self.n_way)
            support_indices = []
            query_indices = []
            
            for c in sampled_classes:
                indices = self.dataset.class_to_indices[c]
                sampled_indices = random.sample(indices, self.k_shot + self.q_query)
                support_indices.extend(sampled_indices[:self.k_shot])
                query_indices.extend(sampled_indices[self.k_shot:])
                
            yield support_indices, query_indices
            
    def __len__(self):
        return self.n_episodes

def create_data_loader(dataset, config) -> DataLoader:
    """Create a DataLoader for the given dataset and configuration."""
    return DataLoader(
        dataset,
        batch_size=config.get("batch_size", 32),
        shuffle=config.get("shuffle", True),
        num_workers=config.get("num_workers", 4),
        pin_memory=config.get("pin_memory", True)
    )
