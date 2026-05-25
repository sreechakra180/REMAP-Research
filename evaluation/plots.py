import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any

# IEEE Plotting standards
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 12

def plot_training_curves(metrics: Dict[str, List[float]], save_path: str):
    """Plot training metrics over epochs/steps."""
    plt.figure(figsize=(3.5, 2.5)) # IEEE single column width is ~3.5 inches
    
    for metric_name, values in metrics.items():
        plt.plot(values, label=metric_name)
        
    plt.xlabel('Training Steps')
    plt.ylabel('Value')
    plt.title('Training Dynamics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_few_shot_comparison(results: Dict[str, float], baselines: Dict[str, float], save_path: str):
    """Plot bar chart comparing few-shot performance."""
    plt.figure(figsize=(3.5, 2.5))
    
    names = list(baselines.keys()) + ['REMAP-Net']
    scores = list(baselines.values()) + [results.get('score', 0)]
    
    x = np.arange(len(names))
    plt.bar(x, scores, color=['gray']*len(baselines) + ['blue'])
    
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('Accuracy (%)')
    plt.title('Few-Shot Performance')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_ablation_bars(ablation_results: Dict[str, Dict], save_path: str):
    """Plot bar chart for ablation results."""
    plt.figure(figsize=(3.5, 2.5))
    
    names = list(ablation_results.keys())
    scores = [res.get('score', 0) for res in ablation_results.values()]
    
    plt.barh(names, scores, color='steelblue')
    plt.xlabel('Performance')
    plt.title('Ablation Study')
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_stability_trajectory(V_history: List[float], save_path: str):
    """Plot Lyapunov energy bounded stability monitoring over time."""
    plt.figure(figsize=(3.5, 2.5))
    
    plt.plot(V_history, color='darkred', label='$V(\\theta)$')
    plt.axhline(y=max(V_history)*1.1, color='k', linestyle='--', label='Boundary')
    
    plt.xlabel('Epochs')
    plt.ylabel('Lyapunov Energy $V(\\theta)$')
    plt.title('Stability Monitoring')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_forgetting_curves(acc_matrix: np.ndarray, save_path: str):
    """Plot sequential task performance showing catastrophic forgetting."""
    plt.figure(figsize=(3.5, 2.5))
    
    n_tasks = acc_matrix.shape[0]
    for i in range(n_tasks):
        # Only plot from task i onwards
        x = np.arange(i, n_tasks)
        y = acc_matrix[x, i]
        plt.plot(x, y, marker='o', markersize=4, label=f'Task {i+1}')
        
    plt.xlabel('Tasks Trained')
    plt.ylabel('Task Accuracy')
    plt.title('Catastrophic Forgetting Mitigation')
    if n_tasks <= 5:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
