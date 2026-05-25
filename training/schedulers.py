import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=1e-6, eta_min=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr) for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) for base_lr in self.base_lrs]

class MetaScheduler:
    def __init__(self, theta_sched, phi_sched, psi_sched):
        self.theta_sched = theta_sched
        self.phi_sched = phi_sched
        self.psi_sched = psi_sched
        
    def step(self):
        if self.theta_sched: self.theta_sched.step()
        if self.phi_sched: self.phi_sched.step()
        if self.psi_sched: self.psi_sched.step()

class PhaseAwareScheduler:
    def __init__(self, base_scheduler, phase_multipliers):
        """
        phase_multipliers: dict mapping phase name to LR multiplier
        """
        self.base_scheduler = base_scheduler
        self.phase_multipliers = phase_multipliers
        self.current_phase = 'pretrain'
        
    def set_phase(self, phase):
        self.current_phase = phase
        
    def step(self):
        self.base_scheduler.step()
        
    def get_last_lr(self):
        lrs = self.base_scheduler.get_last_lr()
        multiplier = self.phase_multipliers.get(self.current_phase, 1.0)
        return [lr * multiplier for lr in lrs]
