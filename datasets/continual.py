import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
import torchvision.transforms as transforms
import numpy as np

class SplitCIFAR10(Dataset):
    """CIFAR-10 split into 5 tasks (2 classes per task)."""
    def __init__(self, root: str, task_id: int, train: bool = True):
        self.dataset = CIFAR10(root=root, train=train, download=True, 
                               transform=transforms.ToTensor())
        self.classes = [task_id * 2, task_id * 2 + 1]
        
        indices = [i for i, (_, label) in enumerate(self.dataset) if label in self.classes]
        self.data = [self.dataset.data[i] for i in indices]
        self.targets = [self.dataset.targets[i] for i in indices]
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx: int):
        return transforms.ToTensor()(self.data[idx]), self.targets[idx]

class SplitCIFAR100(Dataset):
    """CIFAR-100 split into 20 tasks (5 classes per task)."""
    def __init__(self, root: str, task_id: int, train: bool = True):
        self.dataset = CIFAR100(root=root, train=train, download=True,
                                transform=transforms.ToTensor())
        self.classes = list(range(task_id * 5, (task_id + 1) * 5))
        
        indices = [i for i, (_, label) in enumerate(self.dataset) if label in self.classes]
        self.data = [self.dataset.data[i] for i in indices]
        self.targets = [self.dataset.targets[i] for i in indices]
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx: int):
        return transforms.ToTensor()(self.data[idx]), self.targets[idx]

class PermutedMNIST(Dataset):
    """MNIST with random permutations for continual learning."""
    def __init__(self, root: str, task_id: int, train: bool = True):
        self.dataset = MNIST(root=root, train=train, download=True,
                             transform=transforms.ToTensor())
        
        np.random.seed(task_id)
        self.permutation = np.random.permutation(28 * 28)
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx: int):
        img, target = self.dataset[idx]
        img = img.view(-1)[self.permutation].view(1, 28, 28)
        return img, target

class SequentialDomainDataset(Dataset):
    """Generic N-domain sequential dataset."""
    def __init__(self, datasets: list):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cumulative_lengths = np.cumsum(self.lengths)
        
    def __len__(self):
        return self.cumulative_lengths[-1]
        
    def __getitem__(self, idx: int):
        task_id = np.searchsorted(self.cumulative_lengths, idx, side='right')
        if task_id == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_lengths[task_id - 1]
        return self.datasets[task_id][local_idx]
