"""
Semantic Branch Head
FPN + dual heads for target region and state classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            lateral = nn.Conv2d(in_channels, out_channels, 1)
            output = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.lateral_convs.append(lateral)
            self.output_convs.append(output)
    
    def forward(self, features):
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], scale_factor=2, mode='bilinear', align_corners=True
            )
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        return outputs


class SemanticHead(nn.Module):
    """
    Semantic Branch Head.
    
    Accepts state_embeddings_weight as forward argument for cosine similarity.
    """
    
    def __init__(self, num_targets, num_states, in_channels=128):
        super().__init__()
        self.num_targets = num_targets
        self.num_states = num_states
        self.in_channels = in_channels
        
        self.fpn = FPN(
            in_channels_list=[in_channels] * 4,
            out_channels=in_channels
        )
        
        self.target_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, num_targets, 1)
        )
        
        self.state_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, num_states, 1)
        )
        
        # Change detection via FPN cosine similarity (inspired by SegEarth-OV-3)
        # No learned parameters - directly computes cosine sim between T1/T2 features

        # Multi-scale fusion: fuse P2-P5 into P2 resolution
        self.ms_fuse = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.state_proj = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels)
        )
    
    def forward(self, features, state_embeddings_weight=None, t1_features=None, t2_features=None):
        fpn_features = self.fpn(features)
        # Multi-scale fusion: upsample P3-P5 to P2 size, concatenate, fuse
        p2 = fpn_features[0]
        p3 = fpn_features[1]
        p4 = fpn_features[2]
        p5 = fpn_features[3]
        B, C, H, W = p2.shape
        p3_up = F.interpolate(p3, size=(H, W), mode='bilinear', align_corners=True)
        p4_up = F.interpolate(p4, size=(H, W), mode='bilinear', align_corners=True)
        p5_up = F.interpolate(p5, size=(H, W), mode='bilinear', align_corners=True)
        fused = self.ms_fuse(torch.cat([p2, p3_up, p4_up, p5_up], dim=1))

        target_logits = self.target_head(fused)
        state_logits = self.state_head(fused)
        
        if state_embeddings_weight is not None:
            state_proj = self.state_proj(state_embeddings_weight)
            B, C, H, W = fused.shape
            p2_flat = fused.permute(0, 2, 3, 1).reshape(-1, C)
            p2_norm = F.normalize(p2_flat, dim=-1)
            state_norm = F.normalize(state_proj, dim=-1)
            similarity = torch.mm(p2_norm, state_norm.t())
            similarity = similarity.reshape(B, H, W, self.num_states).permute(0, 3, 1, 2)
            state_logits = state_logits + similarity

        # Change detection: cosine similarity between T1/T2 features (SegEarth-OV-3 style)
        if t1_features is not None and t2_features is not None:
            # Multi-scale fusion for change detection
            t1_fpn = self.fpn(t1_features)
            t2_fpn = self.fpn(t2_features)
            t1_B, t1_C, t1_H, t1_W = t1_fpn[0].shape
            t1_p3_up = F.interpolate(t1_fpn[1], size=(t1_H, t1_W), mode='bilinear', align_corners=True)
            t1_p4_up = F.interpolate(t1_fpn[2], size=(t1_H, t1_W), mode='bilinear', align_corners=True)
            t1_p5_up = F.interpolate(t1_fpn[3], size=(t1_H, t1_W), mode='bilinear', align_corners=True)
            t1_fused = self.ms_fuse(torch.cat([t1_fpn[0], t1_p3_up, t1_p4_up, t1_p5_up], dim=1))
            t2_p3_up = F.interpolate(t2_fpn[1], size=(t1_H, t1_W), mode='bilinear', align_corners=True)
            t2_p4_up = F.interpolate(t2_fpn[2], size=(t1_H, t1_W), mode='bilinear', align_corners=True)
            t2_p5_up = F.interpolate(t2_fpn[3], size=(t1_H, t1_W), mode='bilinear', align_corners=True)
            t2_fused = self.ms_fuse(torch.cat([t2_fpn[0], t2_p3_up, t2_p4_up, t2_p5_up], dim=1))
            t1_p2 = t1_fused
            t2_p2 = t2_fused
            # Cosine similarity -> 2-class change logits
            cos_sim = F.cosine_similarity(t1_p2, t2_p2, dim=1)  # [B, H, W]
            cos_sim = cos_sim * 0.5 + 0.5  # scale to [0, 1]
            # 2-class: [no-change prob, change prob]
            change_logits = torch.stack([1 - cos_sim, cos_sim], dim=1)  # [B, 2, H, W]
        else:
            B, C, H, W = p2.shape
            change_logits = torch.zeros(B, 2, H, W, device=p2.device)

        return target_logits, state_logits, change_logits

