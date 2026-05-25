import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from remap_net.training import REMAPNetLightningModule

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    ckpt_path = cfg.get('checkpoint_path', None)
    if not ckpt_path:
        raise ValueError("Must provide checkpoint_path for evaluation.")
        
    model = REMAPNetLightningModule.load_from_checkpoint(ckpt_path, config=cfg)
    model.eval()
    
    trainer = pl.Trainer(
        accelerator='auto',
        devices=1
    )
    
    print("Loaded model for evaluation. Implement testing dataloader to run trainer.test()")
    print("Generating evaluation report...")

if __name__ == "__main__":
    main()
