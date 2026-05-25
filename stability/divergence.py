import torch
import numpy as np
from typing import List, Dict, Optional

class DivergenceDetector:
    """Monitors metrics to detect model divergence."""
    def __init__(self, window_size: int = 100, grad_norm_threshold: float = 100.0, loss_spike_factor: float = 5.0, energy_growth_threshold: float = 1.1):
        self.window_size = window_size
        self.grad_norm_threshold = grad_norm_threshold
        self.loss_spike_factor = loss_spike_factor
        self.energy_growth_threshold = energy_growth_threshold
        
        self.loss_history: List[float] = []
        
        self.last_grad_norm_diverged = False
        self.last_loss_spiked = False
        self.last_energy_grown = False

    def check_gradient_norm(self, grad: torch.Tensor) -> bool:
        """Checks if gradient norm exceeds threshold."""
        norm = torch.norm(grad).item()
        self.last_grad_norm_diverged = norm > self.grad_norm_threshold
        return self.last_grad_norm_diverged

    def check_loss_spike(self, current_loss: float, history: Optional[List[float]] = None) -> bool:
        """Checks if current loss represents an anomalous spike compared to history."""
        if history is not None:
            self.loss_history = history[-self.window_size:]
            
        if len(self.loss_history) < 10:
            self.loss_history.append(current_loss)
            self.last_loss_spiked = False
            return False
            
        median_loss = float(np.median(self.loss_history))
        self.last_loss_spiked = current_loss > (median_loss * self.loss_spike_factor)
        
        self.loss_history.append(current_loss)
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
            
        return self.last_loss_spiked

    def check_energy_growth(self, V_current: float, V_history: List[float]) -> bool:
        """Checks if Lyapunov energy is growing beyond threshold."""
        if not V_history:
            self.last_energy_grown = False
            return False
            
        v_prev = V_history[-1]
        if v_prev <= 0:
            self.last_energy_grown = False
            return False
            
        growth_ratio = V_current / v_prev
        self.last_energy_grown = growth_ratio > self.energy_growth_threshold
        return self.last_energy_grown

    def get_report(self) -> Dict[str, bool]:
        """Returns divergence detection flags."""
        return {
            "grad_norm_diverged": self.last_grad_norm_diverged,
            "loss_spiked": self.last_loss_spiked,
            "energy_grown": self.last_energy_grown,
            "is_diverging": self.last_grad_norm_diverged or self.last_loss_spiked or self.last_energy_grown
        }
