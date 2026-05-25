import torch
import numpy as np
from typing import Dict, List, Any

class StabilityMonitor:
    """Monitors stability metrics and generates reports during training."""
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        
        self.energy_history: List[float] = []
        self.components_history: Dict[str, List[float]] = {"V_theta": [], "V_phi": [], "V_psi": []}
        
        self.violation_steps: List[int] = []
        self.violation_magnitudes: List[float] = []
        
        self.projection_steps: List[int] = []
        self.projection_ratios: List[float] = []

    def log_energy(self, step: int, V: float, V_components: Dict[str, float]):
        self.energy_history.append(V)
        for k, v in V_components.items():
            if k in self.components_history:
                self.components_history[k].append(v)

    def log_violation(self, step: int, V_before: float, V_after: float):
        if V_after > V_before:
            self.violation_steps.append(step)
            self.violation_magnitudes.append(V_after - V_before)

    def log_projection(self, step: int, original_step_size: float, projected_step_size: float):
        self.projection_steps.append(step)
        ratio = projected_step_size / (original_step_size + 1e-8)
        self.projection_ratios.append(ratio)

    def generate_report(self) -> Dict[str, Any]:
        return {
            "total_steps": len(self.energy_history),
            "final_energy": self.energy_history[-1] if self.energy_history else 0.0,
            "total_violations": len(self.violation_steps),
            "max_violation_magnitude": max(self.violation_magnitudes) if self.violation_magnitudes else 0.0,
            "total_projections": len(self.projection_steps),
            "avg_projection_ratio": float(np.mean(self.projection_ratios)) if self.projection_ratios else 1.0
        }

    def get_plot_data(self) -> Dict[str, np.ndarray]:
        return {
            "energy": np.array(self.energy_history),
            "V_theta": np.array(self.components_history["V_theta"]),
            "V_phi": np.array(self.components_history["V_phi"]),
            "V_psi": np.array(self.components_history["V_psi"]),
            "projection_ratios": np.array(self.projection_ratios)
        }

    def to_wandb(self) -> Dict[str, float]:
        metrics = {}
        if self.energy_history:
            metrics["stability/energy_total"] = self.energy_history[-1]
            
        for k, v in self.components_history.items():
            if v:
                metrics[f"stability/energy_{k}"] = v[-1]
                
        if self.projection_ratios:
            metrics["stability/projection_ratio"] = self.projection_ratios[-1]
            
        metrics["stability/cumulative_violations"] = float(len(self.violation_steps))
        
        return metrics
