import torch
from sklearn.metrics import normalized_mutual_info_score
from collections import Counter
from sklearn.cluster import KMeans
from .entropy import differential_entropy
import warnings

def normalized_mutual_info(pred_labels, true_labels):
    return normalized_mutual_info_score(true_labels, pred_labels)

def cluster_purity(pred_labels, true_labels):
    total = len(pred_labels)
    cluster_mapping = {}
    for pl, tl in zip(pred_labels, true_labels):
        if pl not in cluster_mapping:
            cluster_mapping[pl] = []
        cluster_mapping[pl].append(tl)
        
    correct = 0
    for pl, trues in cluster_mapping.items():
        count = Counter(trues)
        most_common = count.most_common(1)[0][1]
        correct += most_common
        
    return correct / total

def compression_ratio(input_dim, abstraction_dim):
    return input_dim / float(abstraction_dim)

def abstraction_quality(z, x, y):
    """
    Computes Q_abs(g) = [I(g(X); Y) / H(Y)] * [H(X) / H(g(X))] (Eq. 11)
    """
    n_classes = len(torch.unique(y))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmeans = KMeans(n_clusters=n_classes, n_init=10, random_state=42)
        z_labels = kmeans.fit_predict(z.detach().cpu().numpy())
    
    nmi = normalized_mutual_info_score(y.detach().cpu().numpy(), z_labels)
    
    h_x = differential_entropy(x.view(x.size(0), -1))
    h_z = differential_entropy(z.view(z.size(0), -1))
    
    if h_z.item() <= 0:
        h_z = torch.tensor(1e-4, device=z.device)
        
    q_abs = nmi * (h_x / h_z)
    return q_abs.item()
