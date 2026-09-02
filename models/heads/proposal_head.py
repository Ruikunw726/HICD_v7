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

    def forward(self, state_logits_pixel, proposals=None, target_logits=None):
        """
        Args:
            state_logits_pixel: [B, num_states, H, W] pixel state predictions
            proposals: list of [N_i, 4] boxes (cx, cy, w, h normalized)
                      If None, extract from target_logits
            target_logits: [B, num_targets, H, W] for proposal extraction

        Returns:
            refined_state_logits: [B, num_states, H, W] refined pixel predictions
            consistency_loss: scalar, encourages refinement consistency
            proposals: proposals used
        """
        B, S, H, W = state_logits_pixel.shape

        # Extract proposals if not provided
        if proposals is None and target_logits is not None:
            proposals = self._extract_proposals(target_logits)

        # Apply attention and refinement
        attention_weights = self.attention(state_logits_pixel)
        refined = self.refine_conv(state_logits_pixel * attention_weights)
        refined_state_logits = state_logits_pixel + refined  # Residual connection

        # Consistency loss: refined predictions within each proposal should be uniform
        consistency_loss = self._compute_consistency(
            refined_state_logits, state_logits_pixel, proposals
        )

        return refined_state_logits, consistency_loss, proposals or []

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
        Encourages the module to produce uniform predictions for each object.
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
                # Variance of predictions within proposal (should be low)
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
