import torch
import numpy as np

class AbstractionHierarchy:
    def __init__(self):
        self.tree = {}
        self.levels = []
        
    def build_hierarchy(self, encoder, data_loader):
        device = next(encoder.parameters()).device
        encoder.eval()
        
        all_z = []
        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(device)
                outputs = encoder(x)
                zs = [out[0].cpu().numpy() for out in outputs]
                all_z.append(zs)
                
        levels = len(all_z[0])
        self.levels = []
        for i in range(levels):
            z_level_i = np.concatenate([batch[i] for batch in all_z], axis=0)
            self.levels.append(z_level_i)
            
        self.tree = {"depth": levels, "samples": len(self.levels[0])}
        return self.tree
        
    def visualize(self):
        return {
            "type": "hierarchy_diagram",
            "data": self.tree,
            "levels_dims": [lvl.shape[1] for lvl in self.levels] if self.levels else []
        }
        
    def get_level(self, depth):
        if 0 <= depth < len(self.levels):
            return self.levels[depth]
        return None
