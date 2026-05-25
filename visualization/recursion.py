import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
import os

from .style import set_ieee_style, get_color, LINEWIDTH, MARKERSIZE

def plot_recursive_update_flow(update_magnitudes_per_level: Dict[str, List[float]], save_path: str) -> None:
    """Plots the flow of update magnitudes from F2 -> F1 -> F0."""
    set_ieee_style()
    fig, ax = plt.subplots()
    
    levels = ['F2', 'F1', 'F0']
    colors = ['f2_color', 'f1_color', 'f0_color']
    
    for level, color_key in zip(levels, colors):
        if level in update_magnitudes_per_level:
            mags = update_magnitudes_per_level[level]
            steps = range(len(mags))
            ax.plot(steps, mags, label=f'{level} Updates', color=get_color(color_key), linewidth=LINEWIDTH)
            
    ax.set_xlabel('Step')
    ax.set_ylabel('Update Magnitude')
    ax.set_yscale('log')
    ax.legend()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

def plot_update_correlation(f0_updates: List[float], f1_updates: List[float], save_path: str) -> None:
    """Plots correlation between F0 and F1 updates."""
    set_ieee_style()
    fig, ax = plt.subplots()
    
    ax.scatter(f0_updates, f1_updates, alpha=0.5, s=MARKERSIZE, color=get_color('primary'))
    
    ax.set_xlabel('F0 Update Magnitude')
    ax.set_ylabel('F1 Update Magnitude')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

def plot_meta_gradient_flow(meta_grad_norms: List[float], save_path: str) -> None:
    """Plots the norm of meta-gradients over time."""
    set_ieee_style()
    fig, ax = plt.subplots()
    
    steps = range(len(meta_grad_norms))
    ax.plot(steps, meta_grad_norms, color=get_color('quinary'), linewidth=LINEWIDTH)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Meta-Gradient Norm')
    ax.set_yscale('log')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
