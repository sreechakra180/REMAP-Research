import torch
from typing import Dict, Any

class RecursiveConstraint:
    """Enforces constraints hierarchically across REMAP-Net levels (F2, F1, F0)."""
    @staticmethod
    def enforce_at_level(params: Dict[str, torch.Tensor], level: str, guardian: Any) -> torch.Tensor:
        """
        Applies constraints to the parameters at the specified level using the StabilityGuardian.
        
        Args:
            params: Dictionary containing 'z_current' and 'z_proposed' states for the level.
            level: The hierarchy level ('F2', 'F1', or 'F0').
            guardian: Instance of StabilityGuardian to use for projection.
            
        Returns:
            constrained_params: The projected parameter state that satisfies Lyapunov constraints.
        """
        valid_levels = ['F2', 'F1', 'F0']
        if level not in valid_levels:
            raise ValueError(f"Invalid level {level}. Must be one of {valid_levels}")
            
        if 'z_current' not in params or 'z_proposed' not in params:
            raise ValueError("params dictionary must contain 'z_current' and 'z_proposed' keys")
            
        z_current = params['z_current']
        z_proposed = params['z_proposed']
        
        constrained_params = guardian.project_update(z_current, z_proposed)
        
        return constrained_params
