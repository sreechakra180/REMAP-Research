import torch
import torch.nn.functional as F

def differential_entropy(samples, k=5):
    """
    Estimates differential entropy of samples using a Kozachenko-Leonenko estimator.
    """
    n, d = samples.shape
    if n <= k:
        return torch.tensor(0.0, device=samples.device)
    
    # Compute pairwise distances
    dists = torch.cdist(samples, samples)
    dists.fill_diagonal_(float('inf'))
    
    # Get k-th nearest neighbor distance
    kth_dists = torch.topk(dists, k, largest=False)[0][:, -1]
    
    # Entropy estimate
    entropy = (d * torch.log(kth_dists + 1e-8)).mean() + torch.log(torch.tensor(n - 1.0, device=samples.device))
    return entropy

def categorical_entropy(probs, eps=1e-8):
    """
    Computes entropy of a categorical distribution.
    """
    return -torch.sum(probs * torch.log(probs + eps), dim=-1).mean()

def conditional_entropy(joint_samples, x_dim):
    """
    Estimates H(Y|X) given joint samples [X, Y].
    """
    x = joint_samples[:, :x_dim]
    h_xy = differential_entropy(joint_samples)
    h_x = differential_entropy(x)
    return h_xy - h_x
