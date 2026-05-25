import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import os
import json

class StabilityCallback(Callback):
    """Logs stability metrics each step."""
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if hasattr(pl_module, 'stability_guardian') and pl_module.stability_guardian is not None:
            # We assume the module computes stability metrics during training step and stores them
            if outputs and isinstance(outputs, dict) and 'l_stab' in outputs:
                trainer.logger.experiment.add_scalar("train/l_stab", outputs['l_stab'].item(), trainer.global_step)

class PhaseTransitionCallback(Callback):
    """Manages phase transitions."""
    def __init__(self, pretrain_epochs=10, meta_warmup_epochs=20):
        super().__init__()
        self.pretrain_epochs = pretrain_epochs
        self.meta_warmup_epochs = meta_warmup_epochs

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch < self.pretrain_epochs:
            phase = 'pretrain'
        elif epoch < self.pretrain_epochs + self.meta_warmup_epochs:
            phase = 'meta_warmup'
        else:
            phase = 'full'
            
        if pl_module.phase != phase:
            pl_module.phase = phase
            print(f"Transitioning to phase: {phase} at epoch {epoch}")

class MetaCheckpointCallback(Callback):
    """Saves model + meta-params + z*"""
    def __init__(self, save_dir='checkpoints'):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % 5 == 0:
            ckpt_path = os.path.join(self.save_dir, f"remap_epoch_{trainer.current_epoch}.ckpt")
            trainer.save_checkpoint(ckpt_path)

class WandBVisualizationCallback(Callback):
    """Logs visualizations."""
    def on_validation_epoch_end(self, trainer, pl_module):
        # Placeholder for logging a sample of predictions or attention weights
        pass

class ReproducibilityCallback(Callback):
    """Saves config snapshot + seed info."""
    def __init__(self, config, save_dir='logs'):
        super().__init__()
        self.config = config
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
    def on_fit_start(self, trainer, pl_module):
        config_path = os.path.join(self.save_dir, 'run_config.json')
        try:
            from omegaconf import OmegaConf
            config_dict = OmegaConf.to_container(self.config, resolve=True)
        except:
            config_dict = self.config
            
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
