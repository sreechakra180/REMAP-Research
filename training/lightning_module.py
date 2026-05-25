import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from .losses import REMAPLoss
from .optimizers import build_optimizer, build_meta_optimizer

class REMAPNetLightningModule(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.phase = 'pretrain' # 'pretrain' | 'meta_warmup' | 'full'
        
        # Lazy import of REMAPNet to avoid circular dependencies
        try:
            from remap_net.core.model import REMAPNet
            self.model = REMAPNet(config.get('model', {}))
        except ImportError:
            self.model = None
            print("Warning: Could not import REMAPNet. Model must be set manually.")
            
        self.loss_fn = REMAPLoss(
            lambda_tc=config.get('training', {}).get('lambda_tc', 1.0),
            lambda_stab=config.get('training', {}).get('lambda_stab', 0.1),
            lambda_abs=config.get('training', {}).get('lambda_abs', 0.01)
        )
        
        # Training state variables
        self.z_t = None
        self.anchor_exemplars = []

        # Turn off automatic optimization to handle manual optimization for Algorithm 1
        self.automatic_optimization = False

    def forward(self, x, task_idx=None):
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        return self.model(x, task_idx)

    def training_step(self, batch, batch_idx):
        x, y, task_idx = batch
        
        opt = self.optimizers()
        if isinstance(opt, list) or isinstance(opt, tuple):
            if len(opt) == 3:
                theta_opt, phi_opt, psi_opt = opt
            else:
                theta_opt = opt[0]
                phi_opt = None
                psi_opt = None
        else:
            theta_opt = opt
            phi_opt = None
            psi_opt = None
            
        # Forward pass
        predictions = self(x, task_idx)
        
        if self.phase == 'pretrain':
            # Phase I: standard forward + loss, SGD on theta only
            loss = self.loss_fn.compute_task_loss(predictions, y)
            theta_opt.zero_grad()
            self.manual_backward(loss)
            theta_opt.step()
            
            self.log('train_loss_phase1', loss, prog_bar=True)
            return loss
            
        elif self.phase == 'meta_warmup':
            # Phase II: meta-plasticity step, compute L_REMAP without L_stab from F2
            loss = self.loss_fn.compute_task_loss(predictions, y)
            
            theta_opt.zero_grad()
            if phi_opt: phi_opt.zero_grad()
            
            self.manual_backward(loss)
            
            theta_opt.step()
            if phi_opt: phi_opt.step()
            
            self.log('train_loss_phase2', loss, prog_bar=True)
            return loss
            
        elif self.phase == 'full':
            # Phase III: full Algorithm 1 with stability certification
            loss, l_task, l_tc, l_stab, l_abs = self.loss_fn(
                predictions, y, 
                model_state={'l_stab': torch.tensor(0.0, device=self.device)} # Placeholder for actual state
            )
            
            theta_opt.zero_grad()
            if phi_opt: phi_opt.zero_grad()
            if psi_opt: psi_opt.zero_grad()
            
            self.manual_backward(loss)
            
            theta_opt.step()
            if phi_opt: phi_opt.step()
            
            if psi_opt:
                # Step 12: proposed update
                # In PyTorch, we can step the optimizer, then check certification.
                psi_opt.step()
                # TODO: Implement Certify V(z(psi_hat)) <= V(z_t) and Bisection
                
            self.log('train_loss', loss, prog_bar=True)
            self.log('l_task', l_task)
            return loss

    def validation_step(self, batch, batch_idx):
        x, y, task_idx = batch
        predictions = self(x, task_idx)
        val_loss = self.loss_fn.compute_task_loss(predictions, y)
        self.log('val_loss', val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        train_cfg = self.config.get('training', {})
        if self.phase == 'pretrain':
            optimizer = build_optimizer(self.model.parameters(), train_cfg)
            return optimizer
        else:
            theta_opt, phi_opt, psi_opt = build_meta_optimizer(self.model, train_cfg)
            return [theta_opt, phi_opt, psi_opt]

    def on_train_epoch_start(self):
        # Allow callback or external logic to update phase
        pass
