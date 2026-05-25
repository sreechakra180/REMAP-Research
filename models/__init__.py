from .backbones import TransformerBackbone, MLPBackbone, ResNetBackbone
from .remap_net import REMAPNet, create_remap_net

__all__ = [
    "TransformerBackbone",
    "MLPBackbone",
    "ResNetBackbone",
    "REMAPNet",
    "create_remap_net",
    "get_model"
]

def get_model(name: str, **kwargs):
    """
    Factory function to get a specific model or backbone.
    
    Args:
        name: Name of the model ('remap_net', 'transformer', 'mlp', 'resnet')
        **kwargs: Arguments to pass to the model constructor
    """
    name = name.lower()
    if name in ['remap_net', 'remapnet']:
        return create_remap_net(**kwargs)
    elif name == 'transformer':
        return TransformerBackbone(**kwargs)
    elif name == 'mlp':
        return MLPBackbone(**kwargs)
    elif name == 'resnet':
        return ResNetBackbone(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")
