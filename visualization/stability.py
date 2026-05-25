import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import os

from .style import set_ieee_style, get_color, LINEWIDTH

def plot_lyapunov_energy(v_history: List[float], save_path: str) -> None:
    """Plots Lyapunov energy V(t) over training."""
    set_ieee_style()
    fig, ax = plt.subplots()
    
    steps = range(len(v_history))
    ax.plot(steps, v_history, color=get_color('stable'), linewidth=LINEWIDTH)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Lyapunov Energy $V(t)$')
    ax.set_yscale('log')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

def plot_energy_components(v_theta: List[float], v_phi: List[float], v_psi: List[float], save_path: str) -> None:
    """Plots stacked energy components over time."""
    set_ieee_style()
    fig, ax = plt.subplots()
    
    steps = range(len(v_theta))
    ax.stackplot(steps, v_theta, v_phi, v_psi, labels=['$V_{\\theta}$', '$V_{\\phi}$', '$V_{\\psi}$'],
                 colors=[get_color('f0_color'), get_color('f1_color'), get_color('f2_color')],
                 alpha=0.8)
                 
    ax.set_xlabel('Step')
    ax.set_ylabel('Energy Components')
    ax.legend(loc='upper right')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

def plot_stability_landscape_2d(guardian: Any, param_range: Tuple[float, float, int], save_path: str) -> None:
    """Plots a 2D contour of the Lyapunov energy landscape."""
    set_ieee_style()
    fig, ax = plt.subplots()
    
    # Placeholder for actual landscape generation
    x = np.linspace(param_range[0], param_range[1], param_range[2])
    y = np.linspace(param_range[0], param_range[1], param_range[2])
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2  # Dummy energy landscape
    
    contour = ax.contourf(X, Y, Z, cmap='viridis', levels=20)
    fig.colorbar(contour, ax=ax, label='Lyapunov Energy')
    
    ax.set_xlabel('Parameter 1')
    ax.set_ylabel('Parameter 2')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

def plot_projection_events(projection_log: List[Tuple[int, float]], save_path: str) -> None:
    """Plots when and how much projections occur to maintain stability."""
    set_ieee_style()
    fig, ax = plt.subplots()
    
    if projection_log:
        steps, magnitudes = zip(*projection_log)
        ax.bar(steps, magnitudes, color=get_color('unstable'), width=1.0)
        
    ax.set_xlabel('Step')
    ax.set_ylabel('Projection Magnitude')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

def plot_convergence_rate(v_history: List[float], save_path: str) -> None:
    """Plots estimated convergence rate $\gamma$ over time."""
    set_ieee_style()
    fig, ax = plt.subplots()
    
    # Estimate gamma: V(t+1) <= (1-gamma) V(t)
    v_history_np = np.array(v_history)
    if len(v_history_np) > 1:
        gamma = 1.0 - (v_history_np[1:] / (v_history_np[:-1] + 1e-10))
        steps = range(1, len(v_history_np))
        ax.plot(steps, gamma, color=get_color('tertiary'), linewidth=LINEWIDTH)
        
    ax.set_xlabel('Step')
    ax.set_ylabel('Estimated Convergence Rate $\gamma$')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
