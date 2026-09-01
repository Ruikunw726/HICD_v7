"""
Task-Specific Adapters
Lightweight adapters to isolate gradients between instance and semantic branches
"""

import torch
import torch.nn as nn


class TaskSpecificAdapter(nn.Module):
    """
    Task-Specific Adapter for gradient isolation
    
    Args:
        in_channels: Input channels
        adapter_channels: Adapter hidden channels
    """
    
    def __init__(self, in_channels, adapter_channels=None):
        super().__init__()
        
        if adapter_channels is None:
            adapter_channels = in_channels // 4
        
        self.adapter = nn.Sequential(
            nn.Linear(in_channels, adapter_channels),
            nn.LayerNorm(adapter_channels),
            nn.GELU(),
            nn.Linear(adapter_channels, in_channels),
            nn.LayerNorm(in_channels)
        )
        
        # Small scale initialization for stability
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x):
        """
        Forward pass with residual connection
        
        Args:
            x: Input tensor [B, C, H, W] or [B, C]
            
        Returns:
            Adapted tensor with residual connection
        """
        if x.dim() == 4:
            # [B, C, H, W] -> [B, C, H*W] -> [B*H*W, C]
            B, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
            adapted = self.adapter(x_flat)
            adapted = adapted.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            # [B, C]
            adapted = self.adapter(x)
        
        return x + self.scale * adapted


class MultiScaleAdapter(nn.Module):
    """
    Multi-scale Task-Specific Adapter
    
    Args:
        in_channels: List of input channels for each scale
        adapter_channels: Adapter hidden channels
    """
    
    def __init__(self, in_channels, adapter_channels=None):
        super().__init__()
        
        self.adapters = nn.ModuleList([
            TaskSpecificAdapter(ch, adapter_channels)
            for ch in in_channels
        ])
    
    def forward(self, features):
        """
        Forward pass
        
        Args:
            features: List of features [C2, C3, C4, C5]
            
        Returns:
            List of adapted features
        """
        return [adapter(feat) for adapter, feat in zip(self.adapters, features)]


if __name__ == '__main__':
    # Test
    adapter = TaskSpecificAdapter(in_channels=128)
    x = torch.randn(2, 128, 32, 32)
    out = adapter(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    
    # Multi-scale test
    ms_adapter = MultiScaleAdapter([128, 128, 128, 128])
    features = [
        torch.randn(2, 128, 64, 64),
        torch.randn(2, 128, 32, 32),
        torch.randn(2, 128, 16, 16),
        torch.randn(2, 128, 8, 8)
    ]
    adapted = ms_adapter(features)
    for i, feat in enumerate(adapted):
        print(f"Scale {i+2}: {feat.shape}")
