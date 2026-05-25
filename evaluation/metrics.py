import torch
import numpy as np
from typing import List, Dict, Union

def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute standard classification accuracy."""
    _, preds = torch.max(outputs, 1)
    return float((preds == targets).float().mean())

def top_k_accuracy(outputs: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """Compute top-k accuracy."""
    _, preds = outputs.topk(k, 1, True, True)
    preds = preds.t()
    correct = preds.eq(targets.view(1, -1).expand_as(preds))
    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
    return float(correct_k.mul_(1.0 / targets.size(0)))

def f1_score(outputs: torch.Tensor, targets: torch.Tensor, average: str = 'macro') -> float:
    """Compute F1 score."""
    from sklearn.metrics import f1_score as sk_f1
    _, preds = torch.max(outputs, 1)
    return float(sk_f1(targets.cpu().numpy(), preds.cpu().numpy(), average=average))

def few_shot_accuracy(support_y: torch.Tensor, query_preds: torch.Tensor, query_y: torch.Tensor) -> float:
    """Compute few-shot episodic accuracy."""
    return accuracy(query_preds, query_y)

def backward_transfer(acc_matrix: np.ndarray) -> float:
    """Compute backward transfer from accuracy matrix."""
    n_tasks = acc_matrix.shape[0]
    if n_tasks <= 1:
        return 0.0
    bwt = 0.0
    for i in range(1, n_tasks):
        for j in range(i):
            bwt += acc_matrix[i, j] - acc_matrix[j, j]
    return float(bwt / (n_tasks * (n_tasks - 1) / 2))

def forward_transfer(acc_matrix: np.ndarray, random_acc: List[float] = None) -> float:
    """Compute forward transfer from accuracy matrix."""
    n_tasks = acc_matrix.shape[0]
    if n_tasks <= 1:
        return 0.0
    if random_acc is None:
        random_acc = [0.1] * n_tasks
    fwt = 0.0
    for i in range(1, n_tasks):
        fwt += acc_matrix[i-1, i] - random_acc[i]
    return float(fwt / (n_tasks - 1))
