import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, List, Dict, Any

class RecursiveArithmeticDataset(Dataset):
    """Generates recursive arithmetic problems for reasoning."""
    
    def __init__(self, num_samples: int = 10000, max_depth: int = 3, max_val: int = 10):
        self.num_samples = num_samples
        self.max_depth = max_depth
        self.max_val = max_val
        self.data = self._generate_data(num_samples)
        
    def _generate_expr(self, depth: int) -> Tuple[str, int]:
        if depth == 0 or np.random.rand() < 0.2:
            val = np.random.randint(1, self.max_val)
            return str(val), val
            
        op = np.random.choice(['+', '*'])
        left_str, left_val = self._generate_expr(depth - 1)
        right_str, right_val = self._generate_expr(depth - 1)
        
        if op == '+':
            val = left_val + right_val
        else:
            val = left_val * right_val
            
        return f"({left_str}{op}{right_str})", val
        
    def _generate_data(self, num_samples: int) -> List[Tuple[str, int]]:
        data = []
        for _ in range(num_samples):
            depth = np.random.randint(1, self.max_depth + 1)
            expr, val = self._generate_expr(depth)
            data.append((expr, val))
        return data
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.data[idx]

class SymbolicReasoningDataset(Dataset):
    """Symbolic rule learning tasks."""
    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x = torch.randn(10)
        y = (x.sum() > 0).long()
        return x, y

class SinusoidRegressionDataset(Dataset):
    """Standard meta-learning sinusoid regression benchmark."""
    def __init__(self, num_samples: int = 1000, num_points: int = 50):
        self.num_samples = num_samples
        self.num_points = num_points
        self.data = []
        for _ in range(num_samples):
            amplitude = np.random.uniform(0.1, 5.0)
            phase = np.random.uniform(0, np.pi)
            x = np.random.uniform(-5.0, 5.0, num_points)
            y = amplitude * np.sin(x + phase)
            self.data.append((torch.tensor(x, dtype=torch.float32).unsqueeze(1), 
                              torch.tensor(y, dtype=torch.float32).unsqueeze(1)))
            
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]
