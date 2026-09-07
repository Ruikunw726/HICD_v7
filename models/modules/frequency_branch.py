"""
Frequency-Aware Branch for HICD v7

Extracts frequency domain features from raw input images using DWT,
independent of the backbone. Targets small/rare classes that have
distinctive frequency signatures (tank, pier, shelter, etc.).

Key insight: backbone features lose frequency information, so we need
a parallel branch that operates on raw inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def haar_dwt_2d(x):
    """
    2D Haar Discrete Wavelet Transform.
    Args:
        x: (B, C, H, W) input tensor
    Returns:
        ll, lh, hl, hh: low-low, low-high, high-low, high-high subbands
    """
    B, C, H, W = x.shape
    # Ensure even dimensions
    if H % 2 != 0:
        x = F.pad(x, (0, 0, 0, 1))
        H += 1
    if W % 2 != 0:
        x = F.pad(x, (0, 1, 0, 0))
        W += 1
    
    # Haar wavelet filters
    # Low-pass: [1/sqrt(2), 1/sqrt(2)]
    # High-pass: [1/sqrt(2), -1/sqrt(2)]
    sqrt2 = 1.4142135623730951
    
    # Reshape for 2D DWT
    x = x.reshape(B, C, H // 2, 2, W // 2, 2)
    
    # Compute subbands
    ll = (x[:, :, :, 0, :, 0] + x[:, :, :, 0, :, 1] + 
          x[:, :, :, 1, :, 0] + x[:, :, :, 1, :, 1]) / (sqrt2 * sqrt2)
    lh = (x[:, :, :, 0, :, 0] + x[:, :, :, 0, :, 1] - 
          x[:, :, :, 1, :, 0] - x[:, :, :, 1, :, 1]) / (sqrt2 * sqrt2)
    hl = (x[:, :, :, 0, :, 0] - x[:, :, :, 0, :, 1] + 
          x[:, :, :, 1, :, 0] - x[:, :, :, 1, :, 1]) / (sqrt2 * sqrt2)
    hh = (x[:, :, :, 0, :, 0] - x[:, :, :, 0, :, 1] - 
          x[:, :, :, 1, :, 0] + x[:, :, :, 1, :, 1]) / (sqrt2 * sqrt2)
    
    return ll, lh, hl, hh


class FrequencyBranch(nn.Module):
    """
    Independent frequency domain branch.
    
    Extracts high-frequency (edges, textures, small targets) and
    low-frequency (overall structure) features from raw input images
    using multi-level Haar DWT.
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        out_dim: Output feature dimension (default: 64)
        levels: Number of DWT decomposition levels (default: 2)
    """
    
    def __init__(self, in_channels=3, out_dim=64, levels=2):
        super().__init__()
        self.levels = levels
        self.in_channels = in_channels
        
        # High-frequency branch: captures edges, textures, small targets
        # Each level produces 3 HF subbands (lh, hl, hh)
        hf_in_channels = in_channels * levels
        self.hf_conv = nn.Sequential(
            nn.Conv2d(hf_in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Low-frequency branch: captures overall structure, large areas
        # Each level produces 1 LF subband (ll)
        lf_in_channels = in_channels * levels
        self.lf_conv = nn.Sequential(
            nn.Conv2d(lf_in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Fusion: combine HF and LF features
        self.fusion = nn.Sequential(
            nn.Conv2d(128, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        
        # Temporal difference: capture change information
        self.diff_conv = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        
        # Gate: adaptive weighting to prevent interference with backbone features
        self.gate = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, img_t1, img_t2, target_size=None):
        """
        Args:
            img_t1: (B, 3, H, W) pre-war image
            img_t2: (B, 3, H, W) post-war image
            target_size: (H', W') target spatial size for alignment with backbone features
        
        Returns:
            freq_features: (B, out_dim, H', W') frequency domain features
        """
        # Extract multi-level DWT for both time points
        hf_t1, lf_t1 = self._multi_level_dwt(img_t1)
        hf_t2, lf_t2 = self._multi_level_dwt(img_t2)
        
        # Encode HF and LF features
        hf_feat_t1 = self.hf_conv(hf_t1)
        lf_feat_t1 = self.lf_conv(lf_t1)
        hf_feat_t2 = self.hf_conv(hf_t2)
        lf_feat_t2 = self.lf_conv(lf_t2)
        
        # Fuse HF + LF for each time point
        freq_t1 = self.fusion(torch.cat([hf_feat_t1, lf_feat_t1], dim=1))
        freq_t2 = self.fusion(torch.cat([hf_feat_t2, lf_feat_t2], dim=1))
        
        # Temporal difference (capture change)
        freq_diff = self.diff_conv(freq_t1 - freq_t2)
        
        # Apply gate for adaptive weighting
        gate = self.gate(freq_diff)
        freq_gated = freq_diff * gate
        
        # Downsample to align with backbone features
        if target_size is not None:
            freq_gated = F.adaptive_avg_pool2d(freq_gated, target_size)
        
        return freq_gated
    
    def _multi_level_dwt(self, x):
        """
        Multi-level Haar DWT decomposition.
        
        Args:
            x: (B, C, H, W) input tensor
        
        Returns:
            hf_features: (B, C*levels*3, H/2, W/2) high-frequency features (upsampled to level 1 size)
            lf_features: (B, C*levels, H/2, W/2) low-frequency features (upsampled to level 1 size)
        """
        hf_list = []
        lf_list = []
        target_size = None
        
        for i in range(self.levels):
            ll, lh, hl, hh = haar_dwt_2d(x)
            
            # Record target size from first level
            if i == 0:
                target_size = lh.shape[2:]
            
            # High-frequency energy: sqrt(lh^2 + hl^2 + hh^2)
            hf = torch.sqrt(lh**2 + hl**2 + hh**2 + 1e-8)
            lf = ll
            
            # Upsample to first level's spatial size for concatenation
            if i > 0:
                hf = F.interpolate(hf, size=target_size, mode='bilinear', align_corners=False)
                lf = F.interpolate(lf, size=target_size, mode='bilinear', align_corners=False)
            
            hf_list.append(hf)
            lf_list.append(lf)
            
            # Next level decomposition on LL subband
            x = ll
        
        return torch.cat(hf_list, dim=1), torch.cat(lf_list, dim=1)
