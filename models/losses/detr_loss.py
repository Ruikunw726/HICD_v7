"""
DETR Loss with Hungarian Matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher for DETR
    
    Matches predictions to ground truth using Hungarian algorithm
    """
    
    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs['pred_logits'].shape[:2]
        
        # Handle empty targets
        sizes = [len(v['boxes']) for v in targets]
        if sum(sizes) == 0:
            return [(torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)) for _ in range(bs)]
        
        # Flatten to compute cost matrix
        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)
        out_bbox = outputs['pred_boxes'].flatten(0, 1)
        
        tgt_ids = torch.cat([v['labels'] for v in targets])
        tgt_bbox = torch.cat([v['boxes'] for v in targets])
        
        # Compute class cost
        cost_class = -out_prob[:, tgt_ids]  # [bs*num_queries, num_targets]
        
        # Compute bbox cost (L1)
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # [bs*num_queries, num_targets]
        
        # Compute GIoU cost
        cost_giou = -self._generalized_box_iou(
            self._box_cxcywh_to_xyxy(out_bbox),
            self._box_cxcywh_to_xyxy(tgt_bbox)
        )
        
        # Final cost matrix
        C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        
        # Split by batch
        sizes = [len(v['boxes']) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    
    def _box_cxcywh_to_xyxy(self, x):
        """Convert [cx, cy, w, h] to [x1, y1, x2, y2]"""
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)
    
    def _generalized_box_iou(self, boxes1, boxes2):
        """Compute GIoU"""
        # Compute IoU
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2[None, :] - inter
        iou = inter / union
        
        # Compute enclosing box
        lt_enc = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
        rb_enc = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh_enc = (rb_enc - lt_enc).clamp(min=0)
        area_enc = wh_enc[:, :, 0] * wh_enc[:, :, 1]
        
        giou = iou - (area_enc - union) / area_enc
        return giou


class DETRLoss(nn.Module):
    """
    DETR Loss with Hungarian Matching
    
    Args:
        num_classes: Number of target classes
        cost_class: Weight for class cost
        cost_bbox: Weight for bbox cost
        cost_giou: Weight for GIoU cost
        loss_class: Weight for class loss
        loss_bbox: Weight for bbox loss
        loss_giou: Weight for GIoU loss
    """
    
    def __init__(self, num_classes, num_states=0, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0,
                 loss_class=1.0, loss_bbox=5.0, loss_giou=2.0, loss_state=2.0):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_states = num_states
        self.matcher = HungarianMatcher(cost_class, cost_bbox, cost_giou)
        self.loss_class = loss_class
        self.loss_bbox = loss_bbox
        self.loss_giou = loss_giou
        self.loss_state = loss_state
        if num_states > 0:
            self.state_ce = nn.CrossEntropyLoss()
        
        # Empty weight for class loss (no-object class has weight 1)
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = 0.1  # Lower weight for no-object
        self.register_buffer('empty_weight', empty_weight)
    
    def forward(self, pred_logits, pred_boxes, targets, pred_state=None):
        """
        Compute DETR loss
        
        Args:
            pred_logits: [B, num_queries, num_classes + 1]
            pred_boxes: [B, num_queries, 4]
            targets: list of dicts with 'labels', 'boxes', and optionally 'states'
            pred_state: [B, num_queries, num_states] optional state predictions
            
        Returns:
            Total loss, loss_dict
        """
        # Match predictions to targets
        outputs = {'pred_logits': pred_logits, 'pred_boxes': pred_boxes}
        indices = self.matcher(outputs, targets)
        
        # Compute classification loss
        loss_class = self._loss_classification(pred_logits, targets, indices)
        
        # Compute bbox loss
        loss_bbox, loss_giou = self._loss_boxes(pred_boxes, targets, indices)
        
        # Compute state loss
        loss_state = torch.tensor(0.0, device=pred_logits.device)
        if pred_state is not None and self.num_states > 0:
            loss_state = self._loss_state(pred_state, targets, indices)
        
        # Weighted sum
        total_loss = (self.loss_class * loss_class + 
                      self.loss_bbox * loss_bbox + 
                      self.loss_giou * loss_giou +
                      self.loss_state * loss_state)
        
        return total_loss, {
            'loss_class': loss_class.item(),
            'loss_bbox': loss_bbox.item(),
            'loss_giou': loss_giou.item(),
            'loss_state': loss_state.item()
        }, indices
    
    def _loss_state(self, pred_state, targets, indices):
        """Compute state classification loss for matched queries."""
        idx = self._get_src_permutation_idx(indices)
        # Collect target states for matched queries
        target_states = []
        for t, (_, J) in zip(targets, indices):
            if 'states' in t and len(J) > 0:
                target_states.append(t['states'][J])
            elif len(J) > 0:
                target_states.append(torch.zeros(len(J), dtype=torch.long, device=pred_state.device))
        if not target_states:
            return torch.tensor(0.0, device=pred_state.device)
        target_states = torch.cat(target_states)
        src_states = pred_state[idx]
        return self.state_ce(src_states, target_states)

    def _loss_classification(self, pred_logits, targets, indices):
        """Compute classification loss"""
        idx = self._get_src_permutation_idx(indices)
        
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            pred_logits.shape[:2], self.num_classes,
            dtype=torch.int64, device=pred_logits.device
        )
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2),
            target_classes,
            self.empty_weight
        )
        
        return loss_ce
    
    def _loss_boxes(self, pred_boxes, targets, indices):
        """Compute bbox losses"""
        idx = self._get_src_permutation_idx(indices)
        
        src_boxes = pred_boxes[idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / max(len(target_boxes), 1)
        
        # GIoU loss
        loss_giou = 1 - torch.diag(self.matcher._generalized_box_iou(
            self.matcher._box_cxcywh_to_xyxy(src_boxes),
            self.matcher._box_cxcywh_to_xyxy(target_boxes)
        ))
        loss_giou = loss_giou.sum() / max(len(target_boxes), 1)
        
        return loss_bbox, loss_giou
    
    def _get_src_permutation_idx(self, indices):
        """Get source indices for permutation"""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


if __name__ == '__main__':
    # Test
    loss_fn = DETRLoss(num_classes=2)
    
    pred_logits = torch.randn(2, 50, 3)  # 2 classes + no-object
    pred_boxes = torch.rand(2, 50, 4) * 0.5 + 0.25
    
    targets = [
        {'labels': torch.tensor([0, 1]), 'boxes': torch.tensor([[0.5, 0.5, 0.1, 0.1], [0.3, 0.3, 0.2, 0.2]])},
        {'labels': torch.tensor([0]), 'boxes': torch.tensor([[0.4, 0.4, 0.15, 0.15]])}
    ]
    
    total_loss, loss_dict = loss_fn(pred_logits, pred_boxes, targets)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")
