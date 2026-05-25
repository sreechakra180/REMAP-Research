import torch
import torch.nn as nn
from typing import Dict, List, Optional
from .lyapunov import LyapunovFunction
from .projection import BisectionProjection

class StabilityGuardian(nn.Module):
    """
    Lyapunov Stability Guardian for REMAP-Net.
    Ensures that network updates monotonically decrease the Lyapunov energy.
    """
    def __init__(self, theta_dim: int, phi_dim: int, psi_dim: int, gamma: float = 0.01, max_bisections: int = 10, ema_decay: float = 0.999):
        super().__init__()
        self.gamma = gamma
        self.max_bisections = max_bisections
        self.ema_decay = ema_decay
        
        self.lyapunov = LyapunovFunction(theta_dim, phi_dim, psi_dim)
        
        total_dim = theta_dim + phi_dim + psi_dim
        self.register_buffer('z_star', torch.zeros(total_dim))
        
        self.violation_count = 0
        self.V_history: List[float] = []

    def certify_update(self, z_current: torch.Tensor, z_proposed: torch.Tensor) -> bool:
        """Checks if proposed update strictly reduces Lyapunov energy."""
        V_old = self.lyapunov.compute_energy(z_current, self.z_star)
        V_new = self.lyapunov.compute_energy(z_proposed, self.z_star)
        return bool((V_new <= V_old).all().item())

    def project_update(self, z_current: torch.Tensor, z_proposed: torch.Tensor, max_bisections: Optional[int] = None) -> torch.Tensor:
        """Projects z_proposed onto the valid energy level set using bisection."""
        if max_bisections is None:
            max_bisections = self.max_bisections
            
        delta_z = z_proposed - z_current
        projected_delta = BisectionProjection.project(
            z_current=z_current,
            delta_z=delta_z,
            energy_fn=self.lyapunov.compute_energy,
            z_star=self.z_star,
            max_iter=max_bisections
        )
        return z_current + projected_delta

    def compute_stability_loss(self, z_current: torch.Tensor, z_proposed: torch.Tensor) -> torch.Tensor:
        """Computes stability penalty: L_stab = max(0, V(z_new) - V(z_old))."""
        V_old = self.lyapunov.compute_energy(z_current, self.z_star)
        V_new = self.lyapunov.compute_energy(z_proposed, self.z_star)
        
        loss = torch.relu(V_new - V_old)
        
        if (loss > 0).any():
            self.violation_count += 1
            
        return loss.mean()

    def update_equilibrium(self, z: torch.Tensor):
        """Updates equilibrium reference z* via EMA."""
        if torch.all(self.z_star == 0):
            self.z_star.copy_(z.detach())
        else:
            self.z_star.mul_(self.ema_decay).add_(z.detach(), alpha=1.0 - self.ema_decay)
            
        current_V = self.lyapunov.compute_energy(z, self.z_star).mean().item()
        self.V_history.append(current_V)
        
        if len(self.V_history) > 1000:
            self.V_history.pop(0)

    def get_stability_metrics(self) -> Dict[str, float]:
        """Returns metrics about stability enforcement."""
        current_V = self.V_history[-1] if self.V_history else 0.0
        return {
            "current_V": current_V,
            "violation_count": float(self.violation_count),
            "estimated_gamma": self.check_convergence_rate(self.V_history)
        }

    def check_convergence_rate(self, V_history: List[float]) -> float:
        """Estimates convergence rate gamma from energy history."""
        if len(V_history) < 2:
            return 0.0
            
        v_curr = V_history[-1]
        v_prev = V_history[-2]
        
        if v_prev <= 0:
            return 0.0
            
        gamma_est = 1.0 - (v_curr / v_prev)
        return max(0.0, float(gamma_est))
