"""
Instance Refinement Module
Post-processing refinement for instance-level change detection.
No GT proposals required - works from pixel branch connected components.

Key insight from research: instance-level detection should be a refinement step,
not an end-to-end detector. BRIGHT challenge winners and OmniOVCD both validated this.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
import numpy as np
from scipy import ndimage


class InstanceRefinementModule(nn.Module):
    """
    Instance refinement module that:
    1. Extracts proposals from pixel branch (connected components)
    2. Pools RoI features
    3. Classifies state via cosine similarity with learnable embeddings
    
    Works WITHOUT instance annotations via consistency supervision:
    - Instance state should match the dominant pixel state within each RoI
    - This provides implicit training signal without GT boxes
    
    Args:
        hidden_dim: Feature dimension
        num_states: Number of change states (3: no-change/damaged/reduced)
        roi_size: RoI Align output size
        spatial_scale: Feature map spatial scale relative to input
    """
    
    def __init__(self, hidden_dim=128, num_states=3, roi_size=7, spatial_scale=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_states = num_states
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        
        # Feature refinement
        self.refine = nn.Sequential(
            nn.Linear(hidden_dim * roi_size * roi_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Learnable state embeddings (3 states)
        self.state_embeddings = nn.Parameter(
            torch.randn(num_states, hidden_dim) * 0.02
        )
        
        # Temperature for cosine similarity
        self.temperature = nn.Parameter(torch.tensor(0.07))
    
    def forward(self, features, proposals=None, state_logits_pixel=None):
        """
        Args:
            features: [B, C, H, W] instance adapter features
            proposals: list of [N_i, 4] boxes (cx, cy, w, h normalized)
                      If None, extract from state_logits_pixel via connected components
            state_logits_pixel: [B, num_states, H, W] pixel-level state predictions
                               Used to extract proposals and for consistency supervision
        
        Returns:
            state_logits: [total_N, num_states] per-instance state predictions
            roi_features: [total_N, hidden_dim] refined instance features
            proposals: list of proposals used
        """
        B, C, H, W = features.shape
        
        # Extract proposals from pixel branch if not provided
        if proposals is None and state_logits_pixel is not None:
            proposals = self._extract_proposals_from_pixel(state_logits_pixel)
        
        if not proposals or all(len(p) == 0 for p in proposals):
            device = features.device
            return (
                torch.zeros(0, self.num_states, device=device),
                torch.zeros(0, self.hidden_dim, device=device),
                proposals or [],
            )
        
        # Convert proposals to absolute RoI coordinates
        all_rois = []
        roi_counts = []
        for b_idx, boxes in enumerate(proposals):
            if len(boxes) == 0:
                roi_counts.append(0)
                continue
            boxes_abs = boxes.clone()
            boxes_abs[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * W
            boxes_abs[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * H
            boxes_abs[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * W
            boxes_abs[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * H
            batch_idx = torch.full((len(boxes), 1), b_idx, dtype=torch.float32, device=boxes.device)
            rois = torch.cat([batch_idx, boxes_abs], dim=1)
            all_rois.append(rois)
            roi_counts.append(len(boxes))
        
        if not all_rois:
            device = features.device
            return (
                torch.zeros(0, self.num_states, device=device),
                torch.zeros(0, self.hidden_dim, device=device),
                proposals,
            )
        
        all_rois = torch.cat(all_rois, dim=0)
        
        # RoI Align
        pooled = roi_align(
            features, all_rois,
            output_size=self.roi_size,
            spatial_scale=self.spatial_scale,
            aligned=True,
        )
        pooled_flat = pooled.flatten(1)
        roi_features = self.refine(pooled_flat)
        
        # Cosine similarity with learnable state embeddings
        roi_norm = F.normalize(roi_features, dim=1)
        state_norm = F.normalize(self.state_embeddings, dim=1)
        state_logits = torch.matmul(roi_norm, state_norm.t()) / self.temperature
        
        return state_logits, roi_features, proposals
    
    def _extract_proposals_from_pixel(self, state_logits_pixel, min_area=50, threshold=0.3):
        """
        Extract instance proposals from pixel branch state predictions.
        Uses connected components on the "changed" probability map.
        """
        B = state_logits_pixel.shape[0]
        proposals = []
        
        # Changed probability = 1 - P(no-change)
        probs = F.softmax(state_logits_pixel, dim=1)
        changed_prob = 1.0 - probs[:, 0]  # [B, H, W]
        
        for b in range(B):
            mask = (changed_prob[b] > threshold).cpu().numpy().astype(np.uint8)
            
            # Morphological cleanup
            from scipy.ndimage import binary_closing, binary_opening
            mask = binary_closing(mask, iterations=2).astype(np.uint8)
            mask = binary_opening(mask, iterations=1).astype(np.uint8)
            
            labeled, nf = ndimage.label(mask)
            boxes = []
            for i in range(1, nf + 1):
                ys, xs = np.where(labeled == i)
                if len(ys) < min_area:
                    continue
                h, w = mask.shape
                cx = (xs.min() + xs.max()) / 2 / w
                cy = (ys.min() + ys.max()) / 2 / h
                bw = (xs.max() - xs.min()) / w
                bh = (ys.max() - ys.min()) / h
                # Filter out tiny and huge boxes
                if 0.01 < bw < 0.8 and 0.01 < bh < 0.8:
                    boxes.append([cx, cy, bw, bh])
            
            if boxes:
                proposals.append(torch.tensor(boxes, dtype=torch.float32, device=state_logits_pixel.device))
            else:
                proposals.append(torch.zeros(0, 4, device=state_logits_pixel.device))
        
        return proposals
    
    def compute_consistency_loss(self, state_logits, proposals, state_logits_pixel):
        """
        Consistency loss: instance state should match dominant pixel state in each RoI.
        This provides training signal without GT instance annotations.
        
        For each proposal, compute the mean pixel state within the box,
        then encourage the instance prediction to match.
        """
        if not proposals or all(len(p) == 0 for p in proposals):
            return torch.tensor(0.0, device=state_logits_pixel.device, requires_grad=True)
        
        B, num_states, H, W = state_logits_pixel.shape
        pixel_probs = F.softmax(state_logits_pixel, dim=1)  # [B, num_states, H, W]
        
        all_targets = []
        idx = 0
        for b_idx, boxes in enumerate(proposals):
            if len(boxes) == 0:
                continue
            for box in boxes:
                cx, cy, bw, bh = box.tolist()
                x1 = max(0, int((cx - bw/2) * W))
                x2 = min(W, int((cx + bw/2) * W))
                y1 = max(0, int((cy - bh/2) * H))
                y2 = min(H, int((cy + bh/2) * H))
                
                if x2 <= x1 or y2 <= y1:
                    idx += 1
                    continue
                
                # Mean pixel state probability in this RoI
                roi_probs = pixel_probs[b_idx, :, y1:y2, x1:x2].mean(dim=[1, 2])
                all_targets.append(roi_probs)
                idx += 1
        
        if not all_targets:
            return torch.tensor(0.0, device=state_logits_pixel.device, requires_grad=True)
        
        target_probs = torch.stack(all_targets)  # [N, num_states]
        pred_probs = F.softmax(state_logits[:len(target_probs)], dim=1)
        
        # KL divergence as consistency loss
        loss = F.kl_div(
            pred_probs.log(), target_probs,
            reduction='batchmean'
        )
        return loss


def build_instance_head(config):
    """Factory for backward compatibility."""
    hidden_dim = config.get('hidden_dim', 128)
    num_states = config.get('num_states', 3)
    return InstanceRefinementModule(
        hidden_dim=hidden_dim,
        num_states=num_states,
        roi_size=7,
        spatial_scale=1.0,
    )
