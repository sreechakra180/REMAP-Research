import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import os

from .style import set_ieee_style, get_color, LINEWIDTH, MARKERSIZE

def plot_tsne_latent_space(features: np.ndarray, labels: np.ndarray, save_path: str) -> None:
    """Plots t-SNE of latent space Z."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("scikit-learn not installed. Cannot plot t-SNE.")
        return
        
    set_ieee_style()
    fig, ax = plt.subplots()
    
    tsne = TSNE(n_components=2, random_state=42)
    z_tsne = tsne.fit_transform(features)
    
    scatter = ax.scatter(z_tsne[:, 0], z_tsne[:, 1], c=labels, cmap='tab10', s=MARKERSIZE, alpha=0.7)
    legend = ax.legend(*scatter.legend_elements(), title="Classes", loc="best")
    ax.add_artist(legend)
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_xticks([])
    ax.set_yticks([])
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

def plot_umap_latent_space(features: np.ndarray, labels: np.ndarray, save_path: str) -> None:
    """Plots UMAP of latent space Z."""
    try:
        import umap
    except ImportError:
        print("umap-learn not installed. Cannot plot UMAP.")
        return
        
    set_ieee_style()
    fig, ax = plt.subplots()
    
    reducer = umap.UMAP(random_state=42)
    z_umap = reducer.fit_transform(features)
    
    scatter = ax.scatter(z_umap[:, 0], z_umap[:, 1], c=labels, cmap='tab10', s=MARKERSIZE, alpha=0.7)
    legend = ax.legend(*scatter.legend_elements(), title="Classes", loc="best")
    ax.add_artist(legend)
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_xticks([])
    ax.set_yticks([])
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

def plot_abstraction_hierarchy(hierarchy: Dict[str, Any], save_path: str) -> None:
    """Plots a tree visualization of the abstraction hierarchy."""
    set_ieee_style()
    fig, ax = plt.subplots()
    
    # Placeholder for networkx tree visualization
    ax.text(0.5, 0.5, "Abstraction Hierarchy Tree\n(Placeholder)", 
            horizontalalignment='center', verticalalignment='center')
    ax.axis('off')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

def plot_information_plane(i_xz: List[float], i_zy: List[float], save_path: str, labels: Optional[List[str]] = None) -> None:
    """Plots Information Bottleneck information plane (I(X;Z) vs I(Z;Y))."""
    set_ieee_style()
    fig, ax = plt.subplots()
    
    ax.plot(i_xz, i_zy, marker='o', color=get_color('abstraction'), markersize=MARKERSIZE, linewidth=LINEWIDTH)
    
    if labels:
        for i, label in enumerate(labels):
            ax.annotate(label, (i_xz[i], i_zy[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=6)
            
    ax.set_xlabel('Compression $I(X;Z)$')
    ax.set_ylabel('Predictive Info $I(Z;Y)$')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

def plot_compression_vs_quality(results: List[Tuple[float, float]], save_path: str) -> None:
    """Plots compression vs task quality tradeoff."""
    set_ieee_style()
    fig, ax = plt.subplots()
    
    if results:
        comp, qual = zip(*results)
        ax.scatter(comp, qual, color=get_color('primary'), s=MARKERSIZE*5)
        
    ax.set_xlabel('Compression')
    ax.set_ylabel('Quality (Accuracy)')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
