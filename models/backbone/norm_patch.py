# Replace mmcv dependency
import torch.nn as nn

def build_norm_layer(cfg, num_features):
    """Simple norm layer builder"""
    eps = cfg.get('eps', 1e-5)
    momentum = cfg.get('momentum', 0.1)
    requires_grad = cfg.get('requires_grad', True)
    
    if cfg['type'] == 'BN':
        layer = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
    elif cfg['type'] == 'SyncBN':
        layer = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
    elif cfg['type'] == 'LN':
        layer = nn.LayerNorm(num_features, eps=eps)
    else:
        raise ValueError(f"Unsupported norm type: {cfg['type']}")
    
    if not requires_grad:
        for param in layer.parameters():
            param.requires_grad = False
    
    return cfg['type'], layer
