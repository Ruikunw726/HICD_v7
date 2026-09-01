"""
Neighbor Feature Aggregation (NFA) Module
Multi-scale feature fusion for remote sensing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusionModule(nn.Module):
    """Feature fusion with identity shortcut"""
    
    def __init__(self, fuse_channels, identity_channels, out_channels):
        super().__init__()
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(fuse_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_identity = nn.Conv2d(identity_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, fused, identity):
        return self.relu(self.conv_fuse(fused) + self.conv_identity(identity))


class NeighborFeatureAggregation(nn.Module):
    """
    Neighbor Feature Aggregation for multi-scale feature fusion
    
    Args:
        in_channels: List of input channels for each scale [C1, C2, C3, C4, C5]
        out_channels: Output channels for each scale
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = out_channels // 2
        self.out_channels = out_channels
        
        # Scale 2: C2 + C3
        self.conv_scale2_c2 = self._make_conv(in_channels[1], self.mid_channels)
        self.conv_scale2_c3 = self._make_conv(in_channels[2], self.mid_channels)
        self.conv_aggregation_s2 = FeatureFusionModule(
            self.mid_channels * 2, in_channels[1], out_channels
        )
        
        # Scale 3: C2 + C3 + C4
        self.conv_scale3_c2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._make_conv(in_channels[1], self.mid_channels)
        )
        self.conv_scale3_c3 = self._make_conv(in_channels[2], self.mid_channels)
        self.conv_scale3_c4 = self._make_conv(in_channels[3], self.mid_channels)
        self.conv_aggregation_s3 = FeatureFusionModule(
            self.mid_channels * 3, in_channels[2], out_channels
        )
        
        # Scale 4: C3 + C4 + C5
        self.conv_scale4_c3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._make_conv(in_channels[2], self.mid_channels)
        )
        self.conv_scale4_c4 = self._make_conv(in_channels[3], self.mid_channels)
        self.conv_scale4_c5 = self._make_conv(in_channels[4], self.mid_channels)
        self.conv_aggregation_s4 = FeatureFusionModule(
            self.mid_channels * 3, in_channels[3], out_channels
        )
        
        # Scale 5: C4 + C5
        self.conv_scale5_c4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._make_conv(in_channels[3], self.mid_channels)
        )
        self.conv_scale5_c5 = self._make_conv(in_channels[4], self.mid_channels)
        self.conv_aggregation_s5 = FeatureFusionModule(
            self.mid_channels * 2, in_channels[4], out_channels
        )
    
    def _make_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, c2, c3, c4, c5):
        """
        Forward pass
        
        Args:
            c2: stride 4 features
            c3: stride 8 features
            c4: stride 16 features
            c5: stride 32 features
            
        Returns:
            s2, s3, s4, s5: Fused multi-scale features
        """
        # Scale 2
        c2_s2 = self.conv_scale2_c2(c2)
        c3_s2 = self.conv_scale2_c3(c3)
        c3_s2 = F.interpolate(c3_s2, scale_factor=2, mode='bilinear', align_corners=True)
        s2 = self.conv_aggregation_s2(torch.cat([c2_s2, c3_s2], dim=1), c2)
        
        # Scale 3
        c2_s3 = self.conv_scale3_c2(c2)
        c3_s3 = self.conv_scale3_c3(c3)
        c4_s3 = self.conv_scale3_c4(c4)
        c4_s3 = F.interpolate(c4_s3, scale_factor=2, mode='bilinear', align_corners=True)
        s3 = self.conv_aggregation_s3(torch.cat([c2_s3, c3_s3, c4_s3], dim=1), c3)
        
        # Scale 4
        c3_s4 = self.conv_scale4_c3(c3)
        c4_s4 = self.conv_scale4_c4(c4)
        c5_s4 = self.conv_scale4_c5(c5)
        c5_s4 = F.interpolate(c5_s4, scale_factor=2, mode='bilinear', align_corners=True)
        s4 = self.conv_aggregation_s4(torch.cat([c3_s4, c4_s4, c5_s4], dim=1), c4)
        
        # Scale 5
        c4_s5 = self.conv_scale5_c4(c4)
        c5_s5 = self.conv_scale5_c5(c5)
        s5 = self.conv_aggregation_s5(torch.cat([c4_s5, c5_s5], dim=1), c5)
        
        return s2, s3, s4, s5


if __name__ == '__main__':
    # Test
    in_channels = [64, 64, 128, 256, 512]  # LWGANet-L1
    nfa = NeighborFeatureAggregation(in_channels, out_channels=128)
    
    c2 = torch.randn(2, 64, 64, 64)
    c3 = torch.randn(2, 128, 32, 32)
    c4 = torch.randn(2, 256, 16, 16)
    c5 = torch.randn(2, 512, 8, 8)
    
    s2, s3, s4, s5 = nfa(c2, c3, c4, c5)
    print(f"s2: {s2.shape}")
    print(f"s3: {s3.shape}")
    print(f"s4: {s4.shape}")
    print(f"s5: {s5.shape}")
