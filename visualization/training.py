import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple
import os

from .style import set_ieee_style, get_color, LINEWIDTH

def plot_loss_curves(train_losses: List[float], val_losses: List[float], save_path: str) -> None:
    """Plots training and validation loss curves."""
    set_ieee_style()
    fig, ax = plt.subplots()
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train Loss', color=get_color('primary'), linewidth=LINEWIDTH)
    if val_losses:
        ax.plot(epochs, val_losses, label='Validation Loss', color=get_color('secondary'), linewidth=LINEWIDTH)
        
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

def plot_learning_rate_schedule(lr_history: List[float], save_path: str) -> None:
    """Plots learning rate schedule over training."""
    set_ieee_style()
    fig, ax = plt.subplots()
    
    steps = range(len(lr_history))
    ax.plot(steps, lr_history, color=get_color('tertiary'), linewidth=LINEWIDTH)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

def plot_gradient_norms(grad_norms_per_level: Dict[str, List[float]], save_path: str) -> None:
    """Plots gradient norms for F0, F1, F2 over time."""
    set_ieee_style()
    fig, ax = plt.subplots()
    
    for level, color_key in zip(['F0', 'F1', 'F2'], ['f0_color', 'f1_color', 'f2_color']):
        if level in grad_norms_per_level:
            norms = grad_norms_per_level[level]
            steps = range(len(norms))
            ax.plot(steps, norms, label=f'{level} Gradient Norm', color=get_color(color_key), linewidth=LINEWIDTH)
            
    ax.set_xlabel('Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_yscale('log')
    ax.legend()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

def plot_phase_transitions(metrics: List[float], phase_boundaries: List[int], save_path: str) -> None:
    """Plots metrics with vertical lines indicating phase transitions."""
    set_ieee_style()
    fig, ax = plt.subplots()
    
    steps = range(len(metrics))
    ax.plot(steps, metrics, color=get_color('primary'), linewidth=LINEWIDTH)
    
    for boundary in phase_boundaries:
        ax.axvline(x=boundary, color='k', linestyle='--', alpha=0.5, linewidth=LINEWIDTH)
        
    ax.set_xlabel('Step')
    ax.set_ylabel('Metric')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

def plot_meta_update_magnitudes(update_history: Dict[str, List[float]], save_path: str) -> None:
    """Plots magnitudes of meta-updates over time."""
    set_ieee_style()
    fig, ax = plt.subplots()
    
    for name, history in update_history.items():
        steps = range(len(history))
        ax.plot(steps, history, label=name, linewidth=LINEWIDTH)
        
    ax.set_xlabel('Step')
    ax.set_ylabel('Update Magnitude')
    ax.set_yscale('log')
    ax.legend()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
