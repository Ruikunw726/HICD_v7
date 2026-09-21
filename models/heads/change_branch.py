"""
Change Detection Branch - Binary change detection (changed vs unchanged).
Independent from state classification to avoid gradient interference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChangeDetectionBranch(nn.Module):
    """
    Binary change detection: changed vs unchanged.
    
    Accepts multi-scale features (list of tensors), fuses them,
    then outputs binary change logits.
    """
    def __init__(self, in_channels=128):
        super().__init__()
        # Multi-scale fusion: fuse 4 scales into 1
        self.ms_fuse = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.change_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 2, 1)  # [no-change, changed]
        )
    
    def forward(self, features):
        """
        Args:
            features: list of [B, C, H_i, W_i] multi-scale features
        Returns:
            change_logits: [B, 2, H, W] binary change logits
        """
        if isinstance(features, (list, tuple)):
            # Multi-scale fusion
            p2 = features[0]
            B, C, H, W = p2.shape
            p3_up = F.interpolate(features[1], size=(H, W), mode='bilinear', align_corners=True)
            p4_up = F.interpolate(features[2], size=(H, W), mode='bilinear', align_corners=True)
            p5_up = F.interpolate(features[3], size=(H, W), mode='bilinear', align_corners=True)
            fused = self.ms_fuse(torch.cat([p2, p3_up, p4_up, p5_up], dim=1))
        else:
            fused = features
        return self.change_head(fused)
