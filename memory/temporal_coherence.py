import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalCoherenceRegularizer(nn.Module):
    def __init__(self, lambda_kl=1.0, lambda_fisher=0.5, lambda_replay=0.5, decay_rate=0.01):
        super().__init__()
        self.lambda_kl = lambda_kl
        self.lambda_fisher = lambda_fisher
        self.lambda_replay = lambda_replay
        self.decay_rate = decay_rate
        
        self.anchors = []
        self.anchors_fisher = []
        
    def compute_tcr_loss(self, model_current, model_previous, anchors, timestamps):
        """
        L_TC (Eq. 12)
        """
        if not anchors or model_previous is None:
            return torch.tensor(0.0, device=next(model_current.parameters()).device)
            
        loss = 0.0
        total_weight = 0.0
        
        for idx, a in enumerate(anchors):
            fisher_info = self.anchors_fisher[idx] if idx < len(self.anchors_fisher) else 1.0
            t_a = timestamps[idx]
            current_time = max(timestamps) if timestamps else 0
            
            omega_a = self.compute_importance_weights(current_time, t_a, fisher_info)
            
            with torch.no_grad():
                out_prev = model_previous(a.unsqueeze(0))
            out_curr = model_current(a.unsqueeze(0))
            
            prob_prev = F.softmax(out_prev, dim=-1)
            log_prob_curr = F.log_softmax(out_curr, dim=-1)
            
            kl_div = F.kl_div(log_prob_curr, prob_prev, reduction='batchmean')
            
            loss += omega_a * kl_div
            total_weight += omega_a
            
        if total_weight > 0:
            loss = loss / total_weight
            
        return loss
        
    def compute_importance_weights(self, current_time, t_a, fisher_info):
        """
        Eq. 13: omega_a
        """
        time_diff = current_time - t_a
        decay = torch.exp(torch.tensor(-self.decay_rate * time_diff, dtype=torch.float32))
        return decay * fisher_info
        
    def update_anchors(self, model, data_batch):
        pass
        
    def get_fisher_diagonal(self, model, data_loader):
        fisher = [torch.zeros_like(p) for p in model.parameters() if p.requires_grad]
        model.eval()
        
        for x, y in data_loader:
            model.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            
            for i, p in enumerate([p for p in model.parameters() if p.requires_grad]):
                if p.grad is not None:
                    fisher[i] += p.grad.data ** 2 / len(data_loader)
                    
        return fisher
