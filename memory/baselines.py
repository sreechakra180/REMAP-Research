import torch

class EWCBaseline:
    def __init__(self, lambda_ewc=1000.0):
        self.lambda_ewc = lambda_ewc
        self.fisher_matrices = {}
        self.optpar_dicts = {}
        
    def compute_regularization_loss(self, model, task_id):
        if task_id not in self.fisher_matrices:
            return torch.tensor(0.0, device=next(model.parameters()).device)
            
        loss = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.fisher_matrices[task_id]:
                fisher = self.fisher_matrices[task_id][name]
                optpar = self.optpar_dicts[task_id][name]
                loss += (fisher * (param - optpar).pow(2)).sum()
                
        return self.lambda_ewc * 0.5 * loss
        
    def update_fisher(self, model, dataloader, task_id):
        pass

class PackNetBaseline:
    def __init__(self, prune_ratio=0.5):
        self.prune_ratio = prune_ratio
        self.masks = {}
        
    def compute_regularization_loss(self, model, task_id):
        return torch.tensor(0.0, device=next(model.parameters()).device)

class ERACEBaseline:
    """Experience Replay with Asymmetric Cross-Entropy"""
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size
        
    def compute_regularization_loss(self, model, task_id):
        return torch.tensor(0.0, device=next(model.parameters()).device)
