import torch
import torch.nn as nn
from typing import Dict, Tuple

class LyapunovFunction(nn.Module):
    """
    Lyapunov energy function for REMAP-Net composite state.
    V(z) = (z - z*)^T P (z - z*)
    where P ≻ 0 is block-diagonal with blocks P_theta, P_phi, P_psi.
    """
    def __init__(self, theta_dim: int, phi_dim: int, psi_dim: int, P_theta_scale: float = 1.0, P_phi_scale: float = 0.1, P_psi_scale: float = 0.01):
        super().__init__()
        self.theta_dim = theta_dim
        self.phi_dim = phi_dim
        self.psi_dim = psi_dim
        
        # Parameterize Cholesky factors L to ensure P = L @ L^T is positive definite
        self.L_theta = nn.Parameter(torch.eye(theta_dim) * (P_theta_scale ** 0.5))
        self.L_phi = nn.Parameter(torch.eye(phi_dim) * (P_phi_scale ** 0.5))
        self.L_psi = nn.Parameter(torch.eye(psi_dim) * (P_psi_scale ** 0.5))
        
    def get_P_matrix(self) -> torch.Tensor:
        """Returns the full block-diagonal P matrix."""
        P_theta = self.L_theta @ self.L_theta.T
        P_phi = self.L_phi @ self.L_phi.T
        P_psi = self.L_psi @ self.L_psi.T
        
        total_dim = self.theta_dim + self.phi_dim + self.psi_dim
        P = torch.zeros(total_dim, total_dim, device=self.L_theta.device)
        
        P[:self.theta_dim, :self.theta_dim] = P_theta
        
        idx2 = self.theta_dim + self.phi_dim
        P[self.theta_dim:idx2, self.theta_dim:idx2] = P_phi
        
        P[idx2:, idx2:] = P_psi
        
        return P

    def verify_positive_definite(self) -> bool:
        """Verifies if P is positive definite via eigenvalues."""
        P = self.get_P_matrix()
        eigenvalues = torch.linalg.eigvalsh(P)
        return bool((eigenvalues > 0).all().item())

    def compute_energy_components(self, z: torch.Tensor, z_star: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes V_theta, V_phi, V_psi separately."""
        is_1d = z.dim() == 1
        if is_1d:
            z = z.unsqueeze(0)
            z_star = z_star.unsqueeze(0)
            
        diff = z - z_star
        diff_theta = diff[:, :self.theta_dim]
        
        idx2 = self.theta_dim + self.phi_dim
        diff_phi = diff[:, self.theta_dim:idx2]
        diff_psi = diff[:, idx2:]
        
        P_theta = self.L_theta @ self.L_theta.T
        P_phi = self.L_phi @ self.L_phi.T
        P_psi = self.L_psi @ self.L_psi.T
        
        V_theta = torch.sum((diff_theta @ P_theta) * diff_theta, dim=-1)
        V_phi = torch.sum((diff_phi @ P_phi) * diff_phi, dim=-1)
        V_psi = torch.sum((diff_psi @ P_psi) * diff_psi, dim=-1)
        
        if is_1d:
            V_theta = V_theta.squeeze(0)
            V_phi = V_phi.squeeze(0)
            V_psi = V_psi.squeeze(0)
            
        return {
            "V_theta": V_theta,
            "V_phi": V_phi,
            "V_psi": V_psi
        }

    def compute_energy(self, z: torch.Tensor, z_star: torch.Tensor) -> torch.Tensor:
        """Computes total Lyapunov energy V(z)."""
        components = self.compute_energy_components(z, z_star)
        return components["V_theta"] + components["V_phi"] + components["V_psi"]

    def compute_energy_gradient(self, z: torch.Tensor, z_star: torch.Tensor) -> torch.Tensor:
        """Computes ∇_z V(z) = 2 P (z - z*)."""
        is_1d = z.dim() == 1
        if is_1d:
            z = z.unsqueeze(0)
            z_star = z_star.unsqueeze(0)
            
        diff = z - z_star
        P = self.get_P_matrix()
        
        grad = 2.0 * (diff @ P)
        
        if is_1d:
            grad = grad.squeeze(0)
            
        return grad
