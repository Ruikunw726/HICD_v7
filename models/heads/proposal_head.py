"""
Voting Refinement Module - Replaces independent instance classification.
Instead of learning its own state classifier, this module learns HOW to
aggregate pixel-level predictions within a proposal into a consistent
object-level prediction.

Key idea: all pixels of the SAME target class within a proposal should
have the same state. This module learns attention weights to identify
reliable pixels and suppress outliers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage


class VotingRefinementModule(nn.Module):
    def __init__(self, num_states=7, roi_size=7):
        super().__init__()
        self.num_states = num_states
        self.roi_size = roi_size

        # Lightweight attention: learns which spatial positions are more reliable
        self.attention = nn.Sequential(
            nn.Conv2d(num_states, num_states, 3, padding=1, groups=num_states),
            nn.BatchNorm2d(num_states),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_states, 1, 1),
            nn.Sigmoid(),
        )

        # Refinement conv: smooths predictions within proposal
        self.refine_conv = nn.Sequential(
            nn.Conv2d(num_states, num_states, 3, padding=1, groups=num_states),
            nn.BatchNorm2d(num_states),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_states, num_states, 1),
        )

    def forward(self, state_logits_pixel, proposals=None, target_logits=None,
                validity_matrix=None):
        """
        Majority voting refinement: within each connected component proposal,
        replace all state predictions with the majority vote.
        
        No learnable parameters - pure post-processing.
        
        Args:
            state_logits_pixel: [B, num_states, H, W] pixel state predictions
            proposals: list of [N_i, 4] boxes (cx, cy, w, h normalized) - unused, kept for API
            target_logits: [B, num_targets, H, W] for proposal extraction
            validity_matrix: [num_targets, num_states] soft validity scores from GNN - unused
        
        Returns:
            refined_state_logits: [B, num_states, H, W] refined pixel predictions
            consistency_loss: scalar (always 0, no learning)
            proposals: proposals used
        """
        B, S, H, W = state_logits_pixel.shape
        device = state_logits_pixel.device
        
        # Get target predictions for connected components
        if target_logits is not None:
            target_pred = target_logits.argmax(dim=1)  # [B, H, W]
        else:
            # If no target logits, return unchanged
            return state_logits_pixel, torch.tensor(0.0, device=device, requires_grad=True), []
        
        # Get state predictions
        state_pred = state_logits_pixel.argmax(dim=1)  # [B, H, W]
        
        # Apply majority voting within each connected component
        refined_state = state_pred.clone()
        
        for b in range(B):
            target_np = target_pred[b].cpu().numpy()
            state_np = state_pred[b].cpu().numpy()
            
            # Process each target class (skip background=0)
            for cls_id in range(1, target_logits.shape[1]):
                mask = (target_np == cls_id).astype(np.uint8)
                if mask.sum() < 10:  # min area
                    continue
                
                # Get connected components
                labeled, nf = ndimage.label(mask)
                for comp_id in range(1, nf + 1):
                    comp_mask = (labeled == comp_id)
                    if comp_mask.sum() < 10:
                        continue
                    
                    # Get state predictions within this component
                    comp_states = state_np[comp_mask]
                    
                    # Majority vote (exclude no-change=0 if there are changed pixels)
                    unique, counts = np.unique(comp_states, return_counts=True)
                    
                    # If there are changed pixels, prefer changed states
                    changed_mask = (comp_states > 0) & (comp_states != 6)
                    if changed_mask.sum() > 0:
                        # Use majority of changed states
                        changed_states = comp_states[changed_mask]
                        unique_c, counts_c = np.unique(changed_states, return_counts=True)
                        majority_state = unique_c[np.argmax(counts_c)]
                    else:
                        # All no-change, keep as is
                        majority_state = unique[np.argmax(counts)]
                    
                    # Apply majority vote to entire component
                    refined_state_np = refined_state[b].cpu().numpy()
                    refined_state_np[comp_mask] = majority_state
                    refined_state[b] = torch.from_numpy(refined_state_np).to(device)
        
        # Convert back to one-hot logits (hard assignment)
        refined_state_logits = torch.zeros_like(state_logits_pixel)
        refined_state_logits.scatter_(1, refined_state.unsqueeze(1), 1.0)
        
        # No consistency loss (no learning)
        consistency_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return refined_state_logits, consistency_loss, []

    def _extract_proposals(self, target_logits, min_area=50):
        """Extract proposals from target predictions (connected components per class)."""
        B, T, H, W = target_logits.shape
        pred = target_logits.argmax(dim=1).cpu().numpy()
        proposals = []

        for b in range(B):
            boxes = []
            for cls_id in range(T):
                if cls_id == 0:  # skip background/farmland
                    continue
                mask = (pred[b] == cls_id).astype(np.uint8)
                if mask.sum() < min_area:
                    continue
                labeled, nf = ndimage.label(mask)
                for i in range(1, nf + 1):
                    ys, xs = np.where(labeled == i)
                    if len(ys) < min_area:
                        continue
                    cx = (xs.min() + xs.max()) / 2 / W
                    cy = (ys.min() + ys.max()) / 2 / H
                    bw = (xs.max() - xs.min()) / W
                    bh = (ys.max() - ys.min()) / H
                    if 0.01 < bw < 0.8 and 0.01 < bh < 0.8:
                        boxes.append([cx, cy, bw, bh])

            if boxes:
                proposals.append(torch.tensor(boxes, dtype=torch.float32, device=target_logits.device))
            else:
                proposals.append(torch.zeros(0, 4, device=target_logits.device))

        return proposals

    def _compute_consistency(self, refined, original, proposals):
        """
        Consistency loss: penalize variance of refined predictions within each proposal.
        """
        if not proposals or all(len(p) == 0 for p in proposals):
            return torch.tensor(0.0, device=refined.device, requires_grad=True)

        B, S, H, W = refined.shape
        probs = F.softmax(refined, dim=1)
        total_loss = torch.tensor(0.0, device=refined.device, requires_grad=True)
        count = 0

        for b_idx, boxes in enumerate(proposals):
            for box in boxes:
                cx, cy, bw, bh = box.tolist()
                x1 = max(0, int((cx - bw/2) * W))
                x2 = min(W, int((cx + bw/2) * W))
                y1 = max(0, int((cy - bh/2) * H))
                y2 = min(H, int((cy + bh/2) * H))
                if x2 <= x1 or y2 <= y1:
                    continue
                roi_probs = probs[b_idx, :, y1:y2, x1:x2]  # [S, h, w]
                
                var = roi_probs.var(dim=[1, 2]).mean()
                
                total_loss = total_loss + var
                count += 1

        if count > 0:
            total_loss = total_loss / count
        return total_loss


def build_instance_head(config):
    """Factory for backward compatibility."""
    num_states = config.get("num_states", 7)
    return VotingRefinementModule(num_states=num_states)
