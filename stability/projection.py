import torch
from typing import Callable

class BisectionProjection:
    """Projection via bisection search along update direction."""
    @staticmethod
    def project(z_current: torch.Tensor, delta_z: torch.Tensor, energy_fn: Callable, z_star: torch.Tensor, max_iter: int = 10) -> torch.Tensor:
        """
        Binary search for the largest step size alpha in [0, 1] 
        such that V(z_current + alpha * delta_z) <= V(z_current).
        """
        V_current = energy_fn(z_current, z_star)
        z_proposed = z_current + delta_z
        V_proposed = energy_fn(z_proposed, z_star)
        
        if (V_proposed <= V_current).all():
            return delta_z
            
        alpha_high = 1.0
        alpha_low = 0.0
        
        for _ in range(max_iter):
            alpha = (alpha_high + alpha_low) / 2.0
            z_test = z_current + alpha * delta_z
            V_test = energy_fn(z_test, z_star)
            
            if (V_test <= V_current).all():
                alpha_low = alpha
            else:
                alpha_high = alpha
                
        return alpha_low * delta_z

class GradientProjection:
    """Gradient descent projection to minimize Lyapunov energy."""
    @staticmethod
    def project(z: torch.Tensor, energy_fn: Callable, z_star: torch.Tensor, eta: float = 0.01) -> torch.Tensor:
        """
        Takes a step in the direction of the negative energy gradient:
        z - eta * grad_V(z)
        """
        if hasattr(energy_fn, '__self__') and hasattr(energy_fn.__self__, 'compute_energy_gradient'):
            grad_V = energy_fn.__self__.compute_energy_gradient(z, z_star)
        else:
            with torch.enable_grad():
                z_req = z.detach().clone().requires_grad_(True)
                V = energy_fn(z_req, z_star)
                
                # Assume batch dimension if it's a tensor of shape > 0
                if V.dim() > 0:
                    V = V.sum()
                    
                V.backward()
                grad_V = z_req.grad
                
        return z - eta * grad_V
