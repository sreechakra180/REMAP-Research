import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .utils import EpisodeSampler, TaskBatch
from PIL import Image
from typing import Tuple

class MiniImageNetDataset(Dataset):
    """miniImageNet dataset for few-shot learning."""
    
    def __init__(self, root: str, split: str = 'train', transform=None):
        """
        Args:
            root: Path to the dataset.
            split: One of 'train', 'val', 'test'.
            transform: Optional transforms to apply.
        """
        self.root = root
        self.split = split
        if transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(84),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(92),
                    transforms.CenterCrop(84),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
            
        file_path = os.path.join(root, f"mini-imagenet-cache-{split}.pkl")
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                self.images = data['image_data']
                self.labels = data['class_dict']
                
            self.data = []
            self.targets = []
            self.class_to_indices = {}
            idx = 0
            for class_name, indices in self.labels.items():
                self.class_to_indices[class_name] = []
                for i in indices:
                    self.data.append(self.images[i])
                    self.targets.append(class_name)
                    self.class_to_indices[class_name].append(idx)
                    idx += 1
        else:
            # Create dummy data for testing if real dataset is not found
            self.data = [np.zeros((84, 84, 3), dtype=np.uint8) for _ in range(600)]
            self.targets = [i // 600 for i in range(600)]
            self.class_to_indices = {0: list(range(600))}
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
