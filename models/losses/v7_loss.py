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


class BackgroundSuppressionLoss(nn.Module):
    """
    Penalize predicting foreground on background pixels.
    Directly addresses the false positive problem.
    """
    def __init__(self, bg_class=15, fg_threshold=0.5):
        super().__init__()
        self.bg_class = bg_class
        self.fg_threshold = fg_threshold

    def forward(self, target_logits, target_gt):
        """
        target_logits: [B, num_classes, H, W]
        target_gt: [B, H, W] with bg_class for background
        """
        # Background mask
        bg_mask = (target_gt == self.bg_class)
        if bg_mask.sum() == 0:
            return torch.tensor(0.0, device=target_logits.device)

        # Get foreground probability for background pixels
        # fg_prob = 1 - P(background)
        probs = F.softmax(target_logits, dim=1)
        bg_prob = probs[:, self.bg_class]  # [B, H, W]
        fg_prob = 1.0 - bg_prob

        # Only penalize background pixels predicted as foreground
        bg_fg_prob = fg_prob[bg_mask]  # [N_bg]

        # Binary cross entropy: encourage bg_prob → 1
        # Loss = -log(bg_prob) for background pixels
        bg_loss = -torch.log(bg_prob[bg_mask] + 1e-8)

        # Focus on hard negatives (high fg_prob)
        hard_mask = bg_fg_prob > self.fg_threshold
        if hard_mask.sum() > 0:
            # Extra weight for hard negatives
            hard_weight = 2.0
            bg_loss = bg_loss * (1.0 + hard_weight * hard_mask.float())

        return bg_loss.mean()


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
        # Save zero-weight mask before update
        zero_mask = (self.weights == 0)
        new_iou = torch.tensor(per_class_iou, dtype=torch.float32)
        self.running_iou = self.momentum * self.running_iou + (1 - self.momentum) * new_iou
        # Inverse IoU weighting: low IoU -> high weight
        inv_iou = 1.0 - self.running_iou + 0.01
        # Normalize so mean=1
        inv_iou = inv_iou / inv_iou.mean()
        # Clip to reasonable range
        inv_iou = inv_iou.clamp(self.min_weight, self.max_weight)
        # Preserve zero-weight classes (e.g., Tower, Farmland)
        inv_iou[zero_mask] = 0.0
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
        self.bg_suppression = BackgroundSuppressionLoss(bg_class=15, fg_threshold=0.5)
        self.w_bg_suppress = config.get("w_bg_suppress", 0.5)

        # State adaptive weighter (with initial weights from config)
        num_states = config.get("num_states", 7)
        state_weights = config.get("state_pixel_weights", None)
        self.state_weighter = AdaptiveClassWeighter(
            num_states, base_weights=state_weights, momentum=0.9, min_weight=0.1, max_weight=10.0
        )

        # Weights
        self.w_pixel = config.get("w_pixel", 1.0)
        self.w_state = config.get("w_state", 2.0)
        self.w_refine = config.get("w_refine", 0.5)
        self.refine_warmup_miou = config.get("refine_warmup_miou", 0.5)
        self.current_target_miou = 0.0  # Updated each epoch
        self.w_contrastive = config.get("w_contrastive", 0.5)
        self._state_ignore_targets = config.get("state_ignore_targets", [])


    def update_adaptive_weights(self, per_class_iou, per_state_iou=None):
        """Call this from training loop after each validation epoch."""
        self.adaptive_weighter.update(per_class_iou)
        if per_state_iou is not None:
            self.state_weighter.update(per_state_iou)

    def get_class_weights(self):
        """Get current adaptive weights for logging."""
        return self.adaptive_weighter.get_weights()

    def set_phase(self, phase, config=None):
        """Switch between training phases with different class weights."""
        if phase == 1:
            pw = [0.0, 10.0, 10.0, 5.0, 50.0, 5.0, 0.0, 50.0, 50.0, 50.0, 10.0, 10.0, 50.0, 20.0, 10.0]
            sw = [0.0, 5.0, 8.0, 8.0, 8.0, 8.0, 0.0]
            print("[Phase 1] Rare-class focused weights activated")
        else:
            if config:
                pw = config.get("pixel_class_weights", None)
                sw = config.get("state_pixel_weights", None)
            else:
                pw = None
                sw = None
            # Reset running_iou so adaptive weighter starts fresh in Phase 2
            # Phase 1 locked nochange/none at 0; their running_iou stayed at 0
            # which causes momentum-based adaptive weights to remain near 0
            self.adaptive_weighter.running_iou = torch.zeros(self.adaptive_weighter.num_classes)
            self.state_weighter.running_iou = torch.zeros(self.state_weighter.num_classes)
            print("[Phase 2] Normal class weights restored, running_iou reset")
        if pw is not None:
            self.adaptive_weighter.weights = torch.tensor(pw, dtype=torch.float32)
        if sw is not None:
            self.state_weighter.weights = torch.tensor(sw, dtype=torch.float32)


    def set_target_miou(self, miou):
        """Update current target mIoU for refine warmup scheduling."""
        self.current_target_miou = miou

    def forward(self, predictions, targets):
        loss_dict = {}
        device = predictions["target_logits"].device

        # Update FocalLoss alpha with adaptive weights
        weights = self.adaptive_weighter.get_weights().to(device)
        self.pixel_ce.alpha = weights

        # === 1. Pixel Target Loss ===
        loss_pixel = _zero(device)
        loss_bg_suppress = _zero(device)
        if "target_logits" in predictions and "target_gt" in targets:
            target_logits = predictions["target_logits"]
            if target_logits.shape[0] > 0:
                target_for_loss = targets["target_gt"].clamp(0, target_logits.shape[1] - 1).clone()
                target_for_loss[targets["target_gt"] == 255] = 255
                ce = self.pixel_ce(target_logits, target_for_loss)
                dice = self.pixel_dice(target_logits, target_for_loss)
                # Background suppression loss
                bg_sup = self.bg_suppression(target_logits, target_for_loss)
                loss_pixel = ce + 2.0 * dice + self.w_bg_suppress * bg_sup
                loss_dict["pixel_ce"] = ce.item()
                loss_dict["pixel_dice"] = dice.item()
                loss_dict["bg_suppress"] = bg_sup.item()
        loss_dict["loss_pixel"] = loss_pixel.item()

        # === 2. Dynamic State Loss (target-conditioned validity) ===
        loss_state = _zero(device)
        if "state_logits" in predictions and "state_gt" in targets:
            state_logits = predictions["state_logits"]
            if state_logits.shape[0] > 0:
                target_gt = targets.get("target_gt", torch.zeros_like(targets["state_gt"]))
                # Exclude background
                valid_mask = (target_gt != 255)
                # Ignore state 6 (none) - only background uses this after crater relabeling
                state_gt_raw_tmp = targets["state_gt"].clamp(0, state_logits.shape[1] - 1)
                valid_mask = valid_mask & (state_gt_raw_tmp != 6)
                # Ignore specific targets from state loss (e.g., Farmland, Tower)
                for ign_cls in self._state_ignore_targets:
                    valid_mask = valid_mask & (target_gt != ign_cls)
                # Dynamic validity: per-pixel mask based on (target, state) graph
                # Get validity matrix from label_structure via predictions
                validity = predictions.get("validity_matrix")
                if validity is not None:
                    # validity: [num_targets, num_states] soft scores
                    # For each pixel, look up valid states for its target
                    # target_gt: [B, H, W] -> one-hot -> [B, T, H, W]
                    num_t = validity.shape[0]
                    target_gt_clamped = target_gt.clamp(0, num_t - 1)
                    target_onehot = F.one_hot(target_gt_clamped, num_t).permute(0, 3, 1, 2).float()
                    # Per-pixel validity: [B, S, H, W]
                    pixel_validity = torch.einsum('bthw,ts->bshw', target_onehot, validity)
                    # Invalid states for this target: set state GT to 255
                    state_gt_raw = targets["state_gt"].clamp(0, state_logits.shape[1] - 1).clone()
                    # Gather validity for the GT state of each pixel
                    gt_validity = pixel_validity.gather(1, state_gt_raw.unsqueeze(1)).squeeze(1)
                    # Pixels where GT state is invalid for the predicted target
                    invalid_state = (gt_validity < 0.05) & valid_mask
                    valid_mask = valid_mask & ~invalid_state
                valid_count = valid_mask.sum().item()
                if valid_count > 0:
                    state_gt = targets["state_gt"].clamp(0, state_logits.shape[1] - 1).clone()
                    state_gt[~valid_mask] = 255
                    st_weights = self.state_weighter.get_weights().to(device)
                    self.state_ce.weight = st_weights
                    ce = self.state_ce(state_logits, state_gt)
                    ce = torch.nan_to_num(ce, nan=0.0)
                    dice = self.state_dice(state_logits, state_gt)
                    dice = torch.nan_to_num(dice, nan=0.0)
                    loss_state = ce + dice
                    loss_dict["pixel_state_ce"] = ce.item()
                    loss_dict["pixel_state_dice"] = dice.item()
                else:
                    loss_dict["pixel_state_ce"] = 0.0
                    loss_dict["pixel_state_dice"] = 0.0
        loss_dict["loss_state"] = loss_state.item()

        # === 4. Refinement Consistency Loss (DISABLED: refinement is inference-only post-processing) ===
        loss_refine = _zero(device)
        loss_dict["refine_consistency"] = 0.0

        # === 5. Contrastive Loss ===
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
