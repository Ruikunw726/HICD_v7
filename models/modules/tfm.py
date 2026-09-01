"""
Temporal Fusion Module (TFM) - V7
Fuse features from T1 and T2, while preserving independent features for interaction layer.
"""

import torch
import torch.nn as nn


class TemporalFusionModule(nn.Module):
    """
    Temporal Fusion Module for change detection (V7)
    
    Same architecture as V6, but returns T1/T2 independent features
    alongside the fused features for use in the interaction layer.
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv_t1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_t2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, feat_t1, feat_t2):
        """
        Forward pass
        
        Returns:
            fused: Fused features [B, C, H, W]
            feat_t1_out: T1 independent features [B, C, H, W]
            feat_t2_out: T2 independent features [B, C, H, W]
        """
        feat_t1_out = self.conv_t1(feat_t1)
        feat_t2_out = self.conv_t2(feat_t2)
        fused = self.conv_fuse(torch.cat([feat_t1_out, feat_t2_out], dim=1))
        return fused, feat_t1_out, feat_t2_out


class MultiScaleTemporalFusion(nn.Module):
    """
    Multi-scale Temporal Fusion Module (V7)
    
    Returns fused features AND independent T1/T2 features at each scale.
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.tfms = nn.ModuleList([
            TemporalFusionModule(in_ch, out_ch)
            for in_ch, out_ch in zip(in_channels, out_channels)
        ])
    
    def forward(self, features_t1, features_t2):
        """
        Args:
            features_t1: List of T1 features [C2, C3, C4, C5]
            features_t2: List of T2 features [C2, C3, C4, C5]
            
        Returns:
            fused_features: List of fused features [C2, C3, C4, C5]
            t1_features: List of T1 independent features [C2, C3, C4, C5]
            t2_features: List of T2 independent features [C2, C3, C4, C5]
        """
        fused_features = []
        t1_features = []
        t2_features = []
        
        for tfm, feat_t1, feat_t2 in zip(self.tfms, features_t1, features_t2):
            fused, t1, t2 = tfm(feat_t1, feat_t2)
            fused_features.append(fused)
            t1_features.append(t1)
            t2_features.append(t2)
        
        return fused_features, t1_features, t2_features
