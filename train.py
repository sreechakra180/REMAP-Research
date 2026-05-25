import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from remap_net.training import (
    REMAPNetLightningModule,
    PhaseTransitionCallback,
    MetaCheckpointCallback,
    ReproducibilityCallback
)
import os

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Set deterministic seeds
    pl.seed_everything(cfg.get('seed', 42), workers=True)
    
    # Build Lightning Module
    model = REMAPNetLightningModule(cfg)
    
    # Callbacks
    callbacks = [
        PhaseTransitionCallback(
            pretrain_epochs=cfg.training.get('pretrain_epochs', 10),
            meta_warmup_epochs=cfg.training.get('meta_warmup_epochs', 20)
        ),
        MetaCheckpointCallback(save_dir=cfg.get('checkpoint_dir', 'checkpoints')),
        ReproducibilityCallback(config=cfg)
    ]
    
    # Loggers
    os.makedirs('logs', exist_ok=True)
    tb_logger = TensorBoardLogger('logs', name='remap_net')
    loggers = [tb_logger]
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.get('epochs', 100),
        callbacks=callbacks,
        logger=loggers,
        accelerator='auto',
        devices=cfg.get('devices', 1),
        strategy=cfg.get('strategy', 'auto'),
        deterministic=True
    )
    
    # Train
    print("Trainer setup complete. Add datamodule to run trainer.fit()")

if __name__ == "__main__":
    main()
