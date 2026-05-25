import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional
import os

from .style import set_ieee_style, IEEE_RCPARAMS

def create_figure_grid(n_rows: int, n_cols: int, figsize: Optional[Tuple[float, float]] = None) -> Tuple[plt.Figure, np.ndarray]:
    """Creates a standard figure grid optimized for publication."""
    set_ieee_style()
    if figsize is None:
        # Default single column size, adjust height based on rows
        base_width = IEEE_RCPARAMS['figure.figsize'][0]
        figsize = (base_width, base_width * 0.75 * n_rows / n_cols)
        
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    return fig, axes

def add_significance_brackets(ax: plt.Axes, x1: float, x2: float, y: float, p_value: float, h: float = 0.05) -> None:
    """Adds statistical significance brackets and asterisks."""
    if p_value < 0.001:
        text = '***'
    elif p_value < 0.01:
        text = '**'
    elif p_value < 0.05:
        text = '*'
    else:
        text = 'ns'
        
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.0, c='k')
    ax.text((x1+x2)*.5, y+h, text, ha='center', va='bottom', color='k')

def save_publication_figure(fig: plt.Figure, save_path: str, formats: List[str] = ['pdf', 'png', 'svg']) -> None:
    """Saves a figure in multiple publication-ready formats."""
    base_path, _ = os.path.splitext(save_path)
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    
    for fmt in formats:
        fig.savefig(f"{base_path}.{fmt}", format=fmt, bbox_inches='tight', dpi=300)

def create_comparison_bar_chart(methods: List[str], metrics: Dict[str, List[float]], save_path: str) -> None:
    """Creates a grouped bar chart comparing methods across metrics."""
    set_ieee_style()
    fig, ax = plt.subplots()
    
    n_methods = len(methods)
    n_metrics = len(metrics)
    
    bar_width = 0.8 / n_methods
    index = np.arange(n_metrics)
    
    for i, method in enumerate(methods):
        values = [metrics[metric][i] for metric in metrics]
        ax.bar(index + i * bar_width, values, bar_width, label=method)
        
    ax.set_xticks(index + bar_width * (n_methods - 1) / 2)
    ax.set_xticklabels(list(metrics.keys()))
    ax.legend()
    
    save_publication_figure(fig, save_path)
    plt.close(fig)

def create_radar_chart(methods: List[str], dimensions: List[str], scores: List[List[float]], save_path: str) -> None:
    """Creates a radar chart for multidimensional comparison."""
    set_ieee_style()
    
    N = len(dimensions)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], dimensions)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1.0"], color="grey", size=7)
    plt.ylim(0, 1)
    
    for i, method in enumerate(methods):
        values = scores[i]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, linestyle='solid', label=method)
        ax.fill(angles, values, alpha=0.1)
        
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    save_publication_figure(fig, save_path)
    plt.close(fig)
