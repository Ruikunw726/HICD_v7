"""
V7 Dual Branch Loss - Graph-Structured Label Space
Includes: pixel CE, state CE, change detection, consistency,
          contrastive validity loss, category hierarchy loss, state group loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255, reduction='mean'):
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
            inputs, targets, reduction='none', ignore_index=self.ignore_index
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

        if self.reduction == 'mean':
            valid = (targets != self.ignore_index).float()
            return focal_loss.sum() / valid.sum().clamp(min=1)
        return focal_loss


def _zero(device):
    return torch.tensor(0.0, device=device, requires_grad=True)


class V7Loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.state_ce = nn.CrossEntropyLoss()
        ignore_idx = config.get('ignore_index', -100)
        pixel_weights = config.get('pixel_class_weights', None)
        if pixel_weights is not None:
            self.register_buffer('pixel_weight_tensor', torch.tensor(pixel_weights, dtype=torch.float32))
        else:
            self.pixel_weight_tensor = None
        self.pixel_target_ce = FocalLoss(alpha=pixel_weights, gamma=2.0, ignore_index=ignore_idx)
        self.pixel_dice = DiceLoss(ignore_index=ignore_idx)
        
        inst_state_weights = config.get('inst_state_weights', None)
        if inst_state_weights is not None:
            self.register_buffer('inst_state_w', torch.tensor(inst_state_weights, dtype=torch.float32))
        else:
            self.inst_state_w = None
        self.state_ce_weighted = nn.CrossEntropyLoss(weight=self.inst_state_w)
        
        state_pixel_weights = config.get('state_pixel_weights', None)
        if state_pixel_weights is not None:
            self.register_buffer('state_pixel_w', torch.tensor(state_pixel_weights, dtype=torch.float32))
        else:
            self.state_pixel_w = None
        self.state_pixel_ce = nn.CrossEntropyLoss(weight=self.state_pixel_w, ignore_index=ignore_idx, label_smoothing=0.05)
        self.state_pixel_dice = DiceLoss(ignore_index=ignore_idx)
        self.change_ce = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.change_dice = DiceLoss()
        
        self.w_pixel = config.get('w_pixel', 1.0)
        self.w_state = config.get('w_state', 1.0)
        self.w_bbox = config.get('w_bbox', 1.0)
        self.w_giou = config.get('w_giou', 1.0)
        self.w_consistency = config.get('w_consistency', 0.3)
        self.w_state_pixel = config.get('w_state_pixel', 1.0)
        self.w_change = config.get('w_change', 1.0)
        self.w_contrastive = config.get('w_contrastive', 0.5)
        
        self.active_heads = config.get('active_heads', {})
        from ..modules.interaction import BoxConsistencyLoss
        self.consistency_fn = BoxConsistencyLoss()

    def forward(self, predictions, targets):
        loss_dict = {}
        device = predictions['change_logits'].device

        # === 1. Pixel target loss (target classification, excl background) ===
        loss_pixel = _zero(device)
        if 'target_logits' in predictions and 'target_gt' in targets:
            target_logits = predictions['target_logits']
            if target_logits.shape[0] > 0:
                # Mask out background pixels from loss
                non_bg_mask = (targets['target_gt'] != 255)
                target_clamped = targets['target_gt'].clamp(0, target_logits.shape[1] - 1)
                # Replace background with ignore_index so loss skips them
                target_for_loss = target_clamped.clone()
                target_for_loss[~non_bg_mask] = 255
                ce = self.pixel_target_ce(target_logits, target_for_loss)
                dice = self.pixel_dice(target_logits, target_for_loss)
                loss_pixel = ce + 2.0 * dice
                loss_dict['pixel_ce'] = ce.item()
                loss_dict['pixel_dice'] = dice.item()
        loss_dict['loss_pixel'] = loss_pixel.item()

        # === 2. Pixel state loss (state classification, excl background) ===
        loss_state_pixel = _zero(device)
        if 'state_logits' in predictions and 'state_gt' in targets:
            state_logits = predictions['state_logits']
            if state_logits.shape[0] > 0:
                # Only compute state loss on non-background pixels
                non_bg_mask = (targets.get('target_gt', torch.zeros_like(targets['state_gt'])) != 255)
                state_gt_c = targets['state_gt'].clamp(0, state_logits.shape[1] - 1)
                state_gt_for_loss = state_gt_c.clone()
                state_gt_for_loss[~non_bg_mask] = 255  # ignore background
                ce = self.state_pixel_ce(state_logits, state_gt_for_loss)
                dice = self.state_pixel_dice(state_logits, state_gt_for_loss)
                loss_state_pixel = ce + dice
                loss_dict['pixel_state_ce'] = ce.item()
                loss_dict['pixel_state_dice'] = dice.item()
        loss_dict['loss_state_pixel'] = loss_state_pixel.item()

        # === 3. Change detection loss ===
        loss_change = _zero(device)
        if 'change_logits' in predictions and 'change_state' in targets:
            change_logits = predictions['change_logits']
            change_gt = targets['change_state']
            if change_logits.shape[0] > 0:
                change_gt_c = change_gt.clamp(0, change_logits.shape[1] - 1)
                loss_change = self.change_ce(change_logits, change_gt_c) + self.change_dice(change_logits, change_gt_c)
                loss_dict['change_ce'] = loss_change.item()
        loss_dict['loss_change'] = loss_change.item()

        # === 4. Instance consistency loss ===
        consistency_loss = predictions.get('consistency_loss', _zero(device))
        if isinstance(consistency_loss, torch.Tensor) and consistency_loss.requires_grad:
            loss_dict['inst_consistency'] = consistency_loss.item()
        else:
            loss_dict['inst_consistency'] = 0.0

        # === 5. Instance state loss (GT-based when available) ===
        loss_state = _zero(device)
        inst_state_logits = predictions.get('inst_state_logits')
        if inst_state_logits is not None and inst_state_logits.shape[0] > 0 and 'instance_labels' in targets:
            gt_states_list = []
            for inst in targets['instance_labels']:
                if 'states' in inst and len(inst['states']) > 0:
                    gt_states_list.append(inst['states'])
            if gt_states_list:
                gt_states = torch.cat(gt_states_list)
                if len(gt_states) == inst_state_logits.shape[0]:
                    loss_state = self.state_ce_weighted(inst_state_logits, gt_states)
                    loss_dict['inst_state'] = loss_state.item()
        loss_dict['loss_instance'] = (consistency_loss + loss_state).item()

        # === 6. Box consistency loss ===
        loss_consistency = _zero(device)
        if 'target_logits' in predictions and 'instance_labels' in targets:
            gt_boxes_list = [t['boxes'] for t in targets['instance_labels']]
            gt_labels_list = [t['labels'] for t in targets['instance_labels']]
            if any(len(b) > 0 for b in gt_boxes_list):
                state_map = predictions.get('state_logits', predictions.get('target_logits'))
                if state_map is not None:
                    loss_consistency = self.consistency_fn(state_map, gt_boxes_list, gt_labels_list)
        loss_dict['loss_consistency'] = loss_consistency.item()

        # === 7. Contrastive label structure losses (NEW) ===
        loss_contrastive = _zero(device)
        contrastive_losses = predictions.get('contrastive_losses', {})
        if contrastive_losses:
            cl_keys = ['contrastive_validity', 'contrastive_hierarchy', 'contrastive_state_group', 'validity_supervision']
            for key in cl_keys:
                if key in contrastive_losses:
                    val = contrastive_losses[key]
                    if isinstance(val, torch.Tensor) and val.requires_grad:
                        loss_contrastive = loss_contrastive + val
                        loss_dict[key] = val.item()
                    else:
                        loss_dict[key] = 0.0
        loss_dict['loss_contrastive'] = loss_contrastive.item()

        # === Total ===
        total_loss = (
            self.w_pixel * loss_pixel +
            self.w_state_pixel * loss_state_pixel +
            self.w_change * loss_change +
            self.w_state * loss_state +
            self.w_consistency * consistency_loss +
            self.w_consistency * loss_consistency +
            self.w_contrastive * loss_contrastive
        )

        total_loss = torch.nan_to_num(total_loss, nan=0.0)
        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


