"""Dataset registry and utilities for REMAP-Net."""

from .utils import set_data_seed, create_data_loader, EpisodeSampler, TaskBatch
from .omniglot import OmniglotDataset, OmniglotEpisodeSampler
from .mini_imagenet import MiniImageNetDataset
from .synthetic import RecursiveArithmeticDataset, SymbolicReasoningDataset, SinusoidRegressionDataset
from .continual import SplitCIFAR10, SplitCIFAR100, PermutedMNIST, SequentialDomainDataset

__all__ = [
    'set_data_seed',
    'create_data_loader',
    'EpisodeSampler',
    'TaskBatch',
    'OmniglotDataset',
    'OmniglotEpisodeSampler',
    'MiniImageNetDataset',
    'RecursiveArithmeticDataset',
    'SymbolicReasoningDataset',
    'SinusoidRegressionDataset',
    'SplitCIFAR10',
    'SplitCIFAR100',
    'PermutedMNIST',
    'SequentialDomainDataset'
]
