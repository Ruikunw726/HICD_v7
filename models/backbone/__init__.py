"""
LWGANet Backbone Wrapper for HICD v6
Light-Weight Grouped Attention Network for Remote Sensing
"""

import torch
import torch.nn as nn
from .lwganet import LWGANet_L0_1242_e32_k11_GELU, LWGANet_L1_1242_e64_k11_GELU, LWGANet_L2_1242_e96_k11_RELU


class LWGANetBackbone(nn.Module):
    """
    LWGANet backbone wrapper for HICD v6
    
    Args:
        variant: LWGANet variant ('L0', 'L1', 'L2')
        pretrained: Whether to load pretrained weights
        pretrained_path: Path to pretrained weights
    """
    
    # Channel configurations for each variant
    CHANNELS = {
        'L0': [32, 32, 64, 128, 256],
        'L1': [64, 64, 128, 256, 512],
        'L2': [96, 96, 192, 384, 768]
    }
    
    def __init__(self, variant='L1', pretrained=True, pretrained_path=None):
        super().__init__()
        
        self.variant = variant
        self.channels = self.CHANNELS[variant]
        
        # Create backbone
        if variant == 'L0':
            self.backbone = LWGANet_L0_1242_e32_k11_GELU(
                num_classes=1000,  # Will not be used
                pretrained=False  # Load manually
            )
        elif variant == 'L1':
            self.backbone = LWGANet_L1_1242_e64_k11_GELU(
                num_classes=1000,
                pretrained=False
            )
        elif variant == 'L2':
            self.backbone = LWGANet_L2_1242_e96_k11_RELU(
                num_classes=1000,
                pretrained=False
            )
        else:
            raise ValueError(f"Unknown variant: {variant}")
        
        # Load pretrained weights
        if pretrained and pretrained_path:
            self._load_pretrained(pretrained_path)
    
    def _load_pretrained(self, pretrained_path):
        """Load pretrained weights"""
        import os
        if not os.path.exists(pretrained_path):
            print(f"Warning: Pretrained weights not found at {pretrained_path}")
            return
        
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'head' and 'avgpool_pre_head' keys (classification head)
        keys_to_remove = [k for k in state_dict.keys() 
                         if k.startswith('head') or k.startswith('avgpool_pre_head')]
        for k in keys_to_remove:
            del state_dict[k]
        
        # Load weights
        missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        
        print(f"Loaded pretrained weights from {pretrained_path}")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            List of multi-scale features: [C2, C3, C4, C5]
            - C2: stride 4, channels[1]
            - C3: stride 8, channels[2]
            - C4: stride 16, channels[3]
            - C5: stride 32, channels[4]
        """
        # Use forward_det to get multi-scale features
        features = self.backbone.forward_det(x)
        
        # features = [C2, C3, C4, C5]
        return features
    
    def freeze(self):
        """Freeze all parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen")
    
    def unfreeze(self, ratio=1.0):
        """
        Unfreeze parameters progressively
        
        Args:
            ratio: Ratio of parameters to unfreeze (0.0 to 1.0)
        """
        all_params = list(self.backbone.parameters())
        num_to_unfreeze = int(len(all_params) * ratio)
        
        # Unfreeze from the end (later layers first)
        for param in all_params[-num_to_unfreeze:]:
            param.requires_grad = True
        
        print(f"Unfroze {num_to_unfreeze}/{len(all_params)} parameters")
    
    def get_out_channels(self):
        """Get output channels for each scale"""
        return self.channels


if __name__ == '__main__':
    # Test
    model = LWGANetBackbone(variant='L1', pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    features = model(x)
    
    for i, feat in enumerate(features):
        print(f"C{i+2}: {feat.shape}")

