import torch
import torch.nn as nn
import torch.nn.functional as F

class REMAPLoss(nn.Module):
    """
    Computes the full REMAP-Net objective (Eq. 14):
    L_REMAP = L_task + lambda_1 * L_TC + lambda_2 * L_stab + lambda_3 * L_abs
    """
    def __init__(self, lambda_tc=1.0, lambda_stab=0.1, lambda_abs=0.01):
        super().__init__()
        self.lambda_tc = lambda_tc
        self.lambda_stab = lambda_stab
        self.lambda_abs = lambda_abs

    def forward(self, predictions, targets, model_state=None, task_type='classification'):
        # L_task
        l_task = self.compute_task_loss(predictions, targets, task_type)
        
        # Placeholder for other losses if model_state is not provided
        if model_state is None:
            return l_task
            
        l_tc = model_state.get('l_tc', torch.tensor(0.0, device=predictions.device))
        l_stab = model_state.get('l_stab', torch.tensor(0.0, device=predictions.device))
        l_abs = model_state.get('l_abs', torch.tensor(0.0, device=predictions.device))
        
        l_remap = l_task + self.lambda_tc * l_tc + self.lambda_stab * l_stab + self.lambda_abs * l_abs
        return l_remap, l_task, l_tc, l_stab, l_abs

    def compute_task_loss(self, predictions, targets, task_type='classification'):
        if task_type == 'classification':
            return F.cross_entropy(predictions, targets)
        elif task_type == 'regression':
            return F.mse_loss(predictions, targets)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def compute_stability_loss(self, z_current, z_proposed, guardian=None):
        # L_stab
        # If guardian is provided, we can use it, but typically this is max(0, V(z_proposed) - V(z_current) + margin)
        # Assuming V is computed by guardian
        if guardian is not None:
            v_curr = guardian(z_current)
            v_prop = guardian(z_proposed)
            return F.relu(v_prop - v_curr + 1e-4) # margin = 1e-4
        else:
            return F.mse_loss(z_proposed, z_current) # Fallback

    def compute_abstraction_loss(self, abstraction_module=None):
        # L_abs
        if abstraction_module is not None and hasattr(abstraction_module, 'compute_q_abs'):
            return -abstraction_module.compute_q_abs()
        return torch.tensor(0.0)
