"""
V7 Interaction Layer
Cross-branch information flow between instance and pixel branches:
1. ChangeGuidedROI: pixel branch change heatmap guides instance branch RoI features
2. SpatialModulation: instance branch change heatmap modulates pixel branch features
3. BoxConsistencyLoss: detection boxes constrain pixel branch consistency within instances
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


class ChangeGuidedROI(nn.Module):
    """
    Pixel -> Instance interaction.
    
    For each proposal RoI, pools the change heatmap within that RoI region
    and concatenates with the RoI features. This lets the instance branch
    know whether the pixel branch detected change inside each box.
    
    The gate is initialized to 0 so early training is not disrupted.
    """

    def __init__(self, in_channels=128, change_channels=1, gate_init=0.0):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(gate_init))

    def forward(self, roi_features, proposals, change_heatmap):
        """
        Args:
            roi_features: [total_M, hidden_dim] — already refined after RoI Align
            proposals: list of [M_i, 4] tensors in [cx,cy,w,h] normalized
            change_heatmap: [B, 1, H, W] — raw logits from change_head (before softmax)
        Returns:
            enhanced_features: [total_M, hidden_dim + 1]
        """
        if len(proposals) == 0 or roi_features.shape[0] == 0:
            return roi_features

        B, _, Hf, Wf = change_heatmap.shape
        all_pooled = []
        idx = 0
        for b_idx, boxes in enumerate(proposals):
            n = len(boxes)
            if n == 0:
                continue
            boxes_abs = boxes.clone()
            boxes_abs[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * Wf
            boxes_abs[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * Hf
            boxes_abs[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * Wf
            boxes_abs[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * Hf
            batch_idx = torch.full((n, 1), b_idx, dtype=torch.float32, device=boxes.device)
            rois = torch.cat([batch_idx, boxes_abs], dim=1)
            pooled = roi_align(
                change_heatmap, rois,
                output_size=(1, 1), spatial_scale=1.0, aligned=True,
            )  # [n, 1, 1, 1]
            # Take change class probability (1 channel regardless of output)
            pooled_flat = pooled[:, :, 0, 0]  # [n, C_change]
            if pooled_flat.shape[1] > 1:
                change_prob = F.softmax(pooled_flat, dim=1)[:, 1:2]  # [n, 1]
            else:
                change_prob = pooled_flat[:, 0:1]  # [n, 1]
            all_pooled.append(change_prob)
            idx += n

        if not all_pooled:
            return torch.zeros(roi_features.shape[0], 1, device=roi_features.device)

        change_pooled = torch.cat(all_pooled, dim=0)  # [total_M, 1]
        return self.gate * change_pooled


class SpatialModulation(nn.Module):
    """
    Instance -> Pixel interaction.
    
    Uses the change heatmap as a spatial attention map to enhance pixel features
    where the model detects changes. This is a lightweight mechanism that lets
    the instance branch's detection signal flow back to the pixel branch.
    
    gate is initialized to 0 so early training is not disrupted.
    """

    def __init__(self, channels=128, gate_init=0.0):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(gate_init))

    def forward(self, pixel_features, change_heatmap):
        """
        Args:
            pixel_features: list of [B, C, H_i, W_i] or single [B, C, H, W]
            change_heatmap: [B, 1, H, W] — raw logits from change_head
        Returns:
            modulated features (same structure as input)
        """
        # Handle both 1-channel and 2-channel inputs
        if change_heatmap.shape[1] > 1:
            attention = F.softmax(change_heatmap, dim=1)[:, 1:2]  # take "change" class
        else:
            attention = torch.sigmoid(change_heatmap)

        if isinstance(pixel_features, list):
            modulated = []
            for feat in pixel_features:
                att = F.interpolate(attention, size=feat.shape[2:], mode='bilinear', align_corners=False)
                modulated.append(feat * (1.0 + self.gate * att))
            return modulated
        else:
            att = F.interpolate(attention, size=pixel_features.shape[2:], mode='bilinear', align_corners=False)
            return pixel_features * (1.0 + self.gate * att)


class BoxConsistencyLoss(nn.Module):
    """
    Within each GT box, pixel branch state predictions should be consistent.
    Penalizes variance of softmax predictions across spatial locations in each RoI.
    """

    def __init__(self, spatial_scale=1.0):
        super().__init__()
        self.spatial_scale = spatial_scale

    def forward(self, pred_state_map, gt_boxes, gt_labels):
        B, C, H, W = pred_state_map.shape
        if B == 0 or pred_state_map.numel() == 0:
            return torch.tensor(0.0, device=pred_state_map.device, requires_grad=True)

        total_loss = torch.tensor(0.0, device=pred_state_map.device, requires_grad=True)
        count = 0

        for b in range(B):
            boxes = gt_boxes[b]
            if boxes is None or len(boxes) == 0:
                continue

            boxes_abs = boxes.clone()
            boxes_abs[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * W
            boxes_abs[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * H
            boxes_abs[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * W
            boxes_abs[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * H

            batch_idx = torch.full((len(boxes), 1), b, dtype=boxes.dtype, device=boxes.device)
            rois = torch.cat([batch_idx, boxes_abs], dim=1)

            pooled = roi_align(
                pred_state_map, rois,
                output_size=(7, 7), spatial_scale=self.spatial_scale, aligned=True,
            )

            pooled_prob = F.softmax(pooled, dim=1)
            var = pooled_prob.var(dim=(2, 3))
            total_loss = total_loss + var.mean()
            count += 1

        if count > 0:
            total_loss = total_loss / count

        if not total_loss.requires_grad:
            total_loss = total_loss.clone().requires_grad_(True)

        return total_loss





