import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Any
import os

from .style import set_ieee_style, get_color, LINEWIDTH

def plot_memory_utilization(episodic_sizes: List[int], lt_sizes: List[int], save_path: str) -> None:
    """Plots memory utilization (episodic and long-term) over time."""
    set_ieee_style()
    fig, ax = plt.subplots()
    
    steps = range(len(episodic_sizes))
    ax.plot(steps, episodic_sizes, label='Episodic Memory', color=get_color('secondary'), linewidth=LINEWIDTH)
    ax.plot(steps, lt_sizes, label='Long-term Memory', color=get_color('memory'), linewidth=LINEWIDTH)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Number of Items')
    ax.legend()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

def plot_retrieval_similarity(similarity_scores: List[float], save_path: str) -> None:
    """Plots the distribution of retrieval similarity scores."""
    set_ieee_style()
    fig, ax = plt.subplots()
    
    ax.hist(similarity_scores, bins=50, color=get_color('primary'), alpha=0.7)
    
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Frequency')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

def plot_memory_evolution(memory_snapshots: List[Dict[str, Any]], save_path: str) -> None:
    """Plots how memory contents evolve over time (e.g., prototype drift)."""
    set_ieee_style()
    fig, ax = plt.subplots()
    
    # Placeholder for tracking specific prototypes over time
    ax.text(0.5, 0.5, "Memory Evolution\n(Track specific prototypes)", 
            horizontalalignment='center', verticalalignment='center')
    ax.axis('off')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

def plot_forgetting_heatmap(accuracy_matrix: np.ndarray, save_path: str) -> None:
    """Plots task accuracy over time to visualize catastrophic forgetting."""
    set_ieee_style()
    fig, ax = plt.subplots()
    
    # accuracy_matrix: Tasks (rows) x Time steps (cols)
    im = ax.imshow(accuracy_matrix, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label='Accuracy')
    
    ax.set_xlabel('Time Step (Evaluations)')
    ax.set_ylabel('Task ID')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
