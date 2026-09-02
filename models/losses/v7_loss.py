"""
V7 Loss - Adaptive Weights + Clean Architecture
Losses:
  1. Pixel Target (Focal + Dice) - what class is each pixel
  2. Pixel State (CE + Dice) - what change state is each pixel
  3. Refinement Consistency - instance branch voting matches pixel predictions
  4. Contrastive Label Structure - graph embedding supervision
Removed:
  - Change loss (redundant with state)
  - Instance state CE (no GT available)
  - Box consistency (no GT boxes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        num_classes = pred.shape[1]
        valid_mask = (target.squeeze(1) != self.ignore_index).float()
        safe_target = target.squeeze(1).clone()
        safe_target[safe_target == self.ignore_index] = 0
        target_clamped = safe_target.clamp(0, num_classes - 1)
        target_onehot = F.one_hot(target_clamped.long(), num_classes).permute(0, 3, 1, 2).float()
        valid_mask_exp = valid_mask.unsqueeze(1)
        pred_flat = (pred * valid_mask_exp).view(pred.shape[0], pred.shape[1], -1)
        target_flat = (target_onehot * valid_mask_exp).view(target_onehot.shape[0], target_onehot.shape[1], -1)
        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction="none", ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            safe_targets = targets.clone()
            safe_targets[safe_targets == self.ignore_index] = 0
            alpha_t = self.alpha.to(inputs.device)[safe_targets]
            valid = (targets != self.ignore_index).float()
            alpha_t = alpha_t * valid
            focal_loss = focal_loss * alpha_t
        if self.reduction == "mean":
            valid = (targets != self.ignore_index).float()
            return focal_loss.sum() / valid.sum().clamp(min=1)
        return focal_loss


class AdaptiveClassWeighter:
    """
    Adaptive class weights based on running IoU.
    Classes with low IoU get higher weight automatically.
    """
    def __init__(self, num_classes, base_weights=None, momentum=0.9, min_weight=0.1, max_weight=10.0):
        self.num_classes = num_classes
        self.momentum = momentum
        self.min_weight = min_weight
        self.max_weight = max_weight
        if base_weights is not None:
            self.weights = torch.tensor(base_weights, dtype=torch.float32)
        else:
            self.weights = torch.ones(num_classes, dtype=torch.float32)
        self.running_iou = torch.zeros(num_classes)

    def update(self, per_class_iou):
        """Update weights based on per-class IoU from validation."""
        new_iou = torch.tensor(per_class_iou, dtype=torch.float32)
        self.running_iou = self.momentum * self.running_iou + (1 - self.momentum) * new_iou
        # Inverse IoU weighting: low IoU -> high weight
        inv_iou = 1.0 - self.running_iou + 0.01
        # Normalize so mean=1
        inv_iou = inv_iou / inv_iou.mean()
        # Clip to reasonable range
        inv_iou = inv_iou.clamp(self.min_weight, self.max_weight)
        self.weights = inv_iou

    def get_weights(self):
        return self.weights.clone()


def _zero(device):
    return torch.tensor(0.0, device=device, requires_grad=True)


class V7Loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        ignore_idx = config.get("ignore_index", -100)
        self.ignore_idx = ignore_idx
        self.config = config
        num_classes = config.get("num_pixel_targets", 15)

        # Adaptive class weighter
        base_weights = config.get("pixel_class_weights", None)
        self.adaptive_weighter = AdaptiveClassWeighter(
            num_classes, base_weights=base_weights,
            momentum=0.9, min_weight=0.1, max_weight=10.0
        )

        # Losses
        self.pixel_ce = FocalLoss(gamma=2.0, ignore_index=ignore_idx)
        self.pixel_dice = DiceLoss(ignore_index=ignore_idx)
        self.state_ce = nn.CrossEntropyLoss(ignore_index=ignore_idx, label_smoothing=0.05)
        self.state_dice = DiceLoss(ignore_index=ignore_idx)

        # State adaptive weighter
        num_states = config.get("num_states", 7)
        self.state_weighter = AdaptiveClassWeighter(
            num_states, momentum=0.9, min_weight=0.1, max_weight=10.0
        )

        # Weights
        self.w_pixel = config.get("w_pixel", 1.0)
        self.w_state = config.get("w_state_pixel", 2.0)
        self.w_refine = config.get("w_refine", 0.5)
        self.w_contrastive = config.get("w_contrastive", 0.5)

    def update_adaptive_weights(self, per_class_iou, per_state_iou=None):
        """Call this from training loop after each validation epoch."""
        self.adaptive_weighter.update(per_class_iou)
        if per_state_iou is not None:
            self.state_weighter.update(per_state_iou)

    def get_class_weights(self):
        """Get current adaptive weights for logging."""
        return self.adaptive_weighter.get_weights()

    def forward(self, predictions, targets):
        loss_dict = {}
        device = predictions["target_logits"].device

        # Update FocalLoss alpha with adaptive weights
        weights = self.adaptive_weighter.get_weights().to(device)
        self.pixel_ce.alpha = weights

        # === 1. Pixel Target Loss ===
        loss_pixel = _zero(device)
        if "target_logits" in predictions and "target_gt" in targets:
            target_logits = predictions["target_logits"]
            if target_logits.shape[0] > 0:
                target_for_loss = targets["target_gt"].clamp(0, target_logits.shape[1] - 1).clone()
                target_for_loss[targets["target_gt"] == 255] = 255
                ce = self.pixel_ce(target_logits, target_for_loss)
                dice = self.pixel_dice(target_logits, target_for_loss)
                loss_pixel = ce + 2.0 * dice
                loss_dict["pixel_ce"] = ce.item()
                loss_dict["pixel_dice"] = dice.item()
        loss_dict["loss_pixel"] = loss_pixel.item()

        # === 2. Pixel State Loss ===
        loss_state = _zero(device)
        if "state_logits" in predictions and "state_gt" in targets:
            state_logits = predictions["state_logits"]
            if state_logits.shape[0] > 0:
                non_bg = (targets.get("target_gt", torch.zeros_like(targets["state_gt"])) != 255)
                state_gt = targets["state_gt"].clamp(0, state_logits.shape[1] - 1).clone()
                state_gt[~non_bg] = 255
                # Adaptive state class weights (safe: only if valid targets exist)
                valid_count = (state_gt != 255).sum().item()
                if valid_count > 0:
                    st_weights = self.state_weighter.get_weights().to(device)
                    self.state_ce.weight = st_weights
                ce = self.state_ce(state_logits, state_gt)
                ce = torch.nan_to_num(ce, nan=0.0)
                dice = self.state_dice(state_logits, state_gt)
                dice = torch.nan_to_num(dice, nan=0.0)
                loss_state = ce + dice
                loss_dict["pixel_state_ce"] = ce.item()
                loss_dict["pixel_state_dice"] = dice.item()
        loss_dict["loss_state"] = loss_state.item()

        # === 3. Refinement Consistency Loss ===
        loss_refine = _zero(device)
        refine_loss = predictions.get("refine_consistency_loss")
        if refine_loss is not None and isinstance(refine_loss, torch.Tensor) and refine_loss.requires_grad:
            loss_refine = refine_loss
            loss_dict["refine_consistency"] = refine_loss.item()
        else:
            loss_dict["refine_consistency"] = 0.0

        # === 4. Contrastive Loss ===
        loss_contrastive = _zero(device)
        cl = predictions.get("contrastive_losses", {})
        if cl:
            for key in ["contrastive_validity", "contrastive_hierarchy", "contrastive_state_group", "validity_supervision"]:
                if key in cl and isinstance(cl[key], torch.Tensor) and cl[key].requires_grad:
                    loss_contrastive = loss_contrastive + cl[key]
                    loss_dict[key] = cl[key].item()
        loss_dict["loss_contrastive"] = loss_contrastive.item()

        # === Total ===
        total = (
            self.w_pixel * loss_pixel +
            self.w_state * loss_state +
            self.w_refine * loss_refine +
            self.w_contrastive * loss_contrastive
        )
        total = torch.nan_to_num(total, nan=0.0)
        loss_dict["total"] = total.item()
        return total, loss_dict
