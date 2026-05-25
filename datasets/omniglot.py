import os
import random
from typing import Tuple, List, Dict
import torch
from torch.utils.data import Dataset
from torchvision.datasets import Omniglot as TorchvisionOmniglot
import torchvision.transforms as transforms
from .utils import EpisodeSampler, TaskBatch

class OmniglotDataset(Dataset):
    """Omniglot dataset for few-shot learning."""
    
    def __init__(self, root: str, background: bool = True, transform=None):
        self.root = root
        self.background = background
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
            
        self.dataset = TorchvisionOmniglot(
            root=self.root, 
            background=self.background, 
            download=True, 
            transform=self.transform
        )
        
        self.class_to_indices = {}
        for i in range(len(self.dataset)):
            _, label = self.dataset[i]
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(i)
            
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]

class OmniglotEpisodeSampler:
    """Creates few-shot episodes for Omniglot."""
    def __init__(self, dataset: OmniglotDataset, n_way: int, k_shot: int, q_query: int, n_episodes: int):
        self.dataset = dataset
        self.sampler = EpisodeSampler(dataset, n_way, k_shot, q_query, n_episodes)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        
    def __iter__(self):
        for support_indices, query_indices in self.sampler:
            support_x = torch.stack([self.dataset[i][0] for i in support_indices])
            support_y = torch.tensor([i // self.k_shot for i in range(len(support_indices))], dtype=torch.long)
            
            query_x = torch.stack([self.dataset[i][0] for i in query_indices])
            query_y = torch.tensor([i // self.q_query for i in range(len(query_indices))], dtype=torch.long)
            
            yield TaskBatch(support_x, support_y, query_x, query_y)
            
    def __len__(self):
        return len(self.sampler)
