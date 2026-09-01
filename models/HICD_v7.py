"""
HICD v7: Hierarchical Instance Change Detection v7
Dual-branch architecture with configurable backbone.
Graph-structured label space with validity constraints and contrastive embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .backbone import LWGANetBackbone
from .backbone.dinov3 import DINOv3Backbone
from .modules import (
    NeighborFeatureAggregation,
    MultiScaleTemporalFusion,
    MultiScaleAdapter,
    ChangeGuidedROI,
    SpatialModulation,
    BoxConsistencyLoss,
    LearnableStateEmbeddings,
    LabelStructure,
    PixelStateHead,
)
from .heads.proposal_head import InstanceRefinementModule
from .losses import V7Loss


def build_backbone(config):
    backbone_cfg = config.get('backbone', {})
    backbone_type = backbone_cfg.get('type', 'lwganet')
    if backbone_type == 'dinov3':
        return DINOv3Backbone(
            variant=backbone_cfg.get('variant', 'dinov3_base'),
            pretrained=backbone_cfg.get('pretrained', True),
            out_channels=backbone_cfg.get('out_channels', [64, 128, 256, 512]),
            freeze_stages=backbone_cfg.get('freeze_stages', 2),
            pretrained_path=backbone_cfg.get('pretrained_path', None),
        )
    else:
        return LWGANetBackbone(
            variant=backbone_cfg.get('variant', 'L1'),
            pretrained=backbone_cfg.get('pretrained', True),
            pretrained_path=backbone_cfg.get('pretrained_path', None),
        )


class HICD_v7(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        num_pixel_targets = config.get('num_pixel_targets', 6)
        num_states = config.get('num_states', 3)
        hidden_dim = config.get('hidden_dim', 128)
        embed_dim = config.get('embed_dim', 64)

        self.backbone = build_backbone(config)
        backbone_channels = self.backbone.get_out_channels()

        self.nfa_t1 = NeighborFeatureAggregation(in_channels=backbone_channels, out_channels=hidden_dim)
        self.nfa_t2 = NeighborFeatureAggregation(in_channels=backbone_channels, out_channels=hidden_dim)
        self.tfm = MultiScaleTemporalFusion(in_channels=[hidden_dim]*4, out_channels=[hidden_dim]*4)
        self.instance_adapter = MultiScaleAdapter(in_channels=[hidden_dim]*4, adapter_channels=hidden_dim//4)
        self.semantic_adapter = MultiScaleAdapter(in_channels=[hidden_dim]*4, adapter_channels=hidden_dim//4)

        self.label_structure = LabelStructure(
            num_targets=num_pixel_targets, num_states=num_states, embed_dim=embed_dim,
        )
        self.shared_state_embeddings = LearnableStateEmbeddings(num_states=num_states, embed_dim=hidden_dim)
        self.instance_head = InstanceRefinementModule(hidden_dim=hidden_dim, num_states=num_states, roi_size=7, spatial_scale=1.0)
        self.semantic_head = PixelStateHead(
            num_targets=num_pixel_targets, num_states=num_states,
            in_channels=hidden_dim, embed_dim=embed_dim, hidden_dim=hidden_dim,
        )
        self.change_guided_roi = ChangeGuidedROI(in_channels=hidden_dim, change_channels=1)
        self.spatial_modulation = SpatialModulation(channels=hidden_dim)
        self.box_consistency_loss = BoxConsistencyLoss()

        config['num_pixel_targets'] = num_pixel_targets
        self.loss_fn = V7Loss(config)

    def forward(self, img_t1, img_t2, targets=None):
        B = img_t1.shape[0]

        feat_t1_list = self.backbone(img_t1)
        feat_t2_list = self.backbone(img_t2)
        nfa_t1 = self.nfa_t1(*feat_t1_list)
        nfa_t2 = self.nfa_t2(*feat_t2_list)
        fused_features, t1_features, t2_features = self.tfm(nfa_t1, nfa_t2)

        inst_features = self.instance_adapter(fused_features)
        sem_features = self.semantic_adapter(fused_features)

        state_emb_weight = self.shared_state_embeddings.get_weight()

        target_logits, state_logits, change_logits = self.semantic_head(
            sem_features, t1_features, t2_features, label_structure=self.label_structure,
        )

        inst_feat_single = inst_features[0] if isinstance(inst_features, list) else inst_features
        inst_state_logits, inst_roi_features, proposals = self.instance_head(
            inst_feat_single, state_logits_pixel=state_logits,
        )

        # Save state logits before interaction (interaction modifies roi_features)
        inst_state_logits_for_loss = inst_state_logits
        # Cross-branch interaction
        if proposals and any(len(p) > 0 for p in proposals):
            inst_state_logits, inst_features = self._instance_interaction(
                inst_features, inst_roi_features, proposals, change_logits,
            )

        # Contrastive losses
        contrastive_losses = {}
        if targets is not None:
            target_gt = targets.get('target_gt')
            state_gt = targets.get('state_gt')
            if target_gt is not None and state_gt is not None:
                contrastive_losses = self.label_structure.compute_contrastive_losses(
                    target_logits, state_logits, target_gt, state_gt,
                )

        consistency_loss = torch.tensor(0.0, device=img_t1.device)
        if inst_state_logits.shape[0] > 0 and proposals and any(len(p) > 0 for p in proposals):
            consistency_loss = self.instance_head.compute_consistency_loss(
                inst_state_logits_for_loss, proposals, state_logits,
            )

        predictions = {
            'target_logits': target_logits,
            'state_logits': state_logits,
            'change_logits': change_logits,
            'inst_state_logits': inst_state_logits_for_loss,
            'inst_roi_features': inst_roi_features,
            'proposals': proposals,
            'consistency_loss': consistency_loss,
            'contrastive_losses': contrastive_losses,
        }
        if targets is not None:
            loss, loss_dict = self.loss_fn(predictions, targets)
        else:
            loss = torch.tensor(0.0, device=img_t1.device, requires_grad=True)
            loss_dict = {}
        return predictions, loss, loss_dict

    def _instance_interaction(self, features, roi_features, proposals, change_logits):
        B, C, H, W = features[0].shape if isinstance(features, list) else features.shape
        flat_features = features[0] if isinstance(features, list) else features

        change_for_roi = change_logits[:, 1:2] if change_logits.shape[1] > 1 else change_logits[:, 0:1]
        change_context = self.change_guided_roi(roi_features, proposals, change_for_roi)
        if isinstance(change_context, torch.Tensor) and change_context.numel() > 0:
            roi_features = roi_features + change_context

        inst_heatmap = torch.zeros(B, 1, H, W, device=flat_features.device)
        for b_idx, boxes in enumerate(proposals):
            if len(boxes) == 0:
                continue
            for box in boxes:
                cx, cy, bw, bh = box.tolist()
                x1 = max(0, int((cx - bw/2) * W))
                x2 = min(W, int((cx + bw/2) * W))
                y1 = max(0, int((cy - bh/2) * H))
                y2 = min(H, int((cy + bh/2) * H))
                inst_heatmap[b_idx, 0, y1:y2, x1:x2] = 1.0
        if isinstance(features, list):
            for i in range(len(features)):
                features[i] = self.spatial_modulation(features[i], inst_heatmap)
        else:
            features = self.spatial_modulation(features, inst_heatmap)

        return roi_features, features

    def freeze_backbone(self):
        self.backbone.freeze()

    def unfreeze_backbone(self, ratio=1.0):
        self.backbone.unfreeze(ratio)

    def get_param_groups(self, lr=1e-4, weight_decay=1e-4):
        param_groups = []
        backbone_trainable = [p for p in self.backbone.parameters() if p.requires_grad]
        if backbone_trainable:
            param_groups.append({'params': backbone_trainable, 'lr': lr*0.1, 'weight_decay': weight_decay, 'name': 'backbone'})
        ntfm_params = list(self.nfa_t1.parameters()) + list(self.nfa_t2.parameters()) + list(self.tfm.parameters())
        param_groups.append({'params': ntfm_params, 'lr': lr, 'weight_decay': weight_decay, 'name': 'nfa_tfm'})
        adapter_params = list(self.instance_adapter.parameters()) + list(self.semantic_adapter.parameters())
        param_groups.append({'params': adapter_params, 'lr': lr, 'weight_decay': weight_decay, 'name': 'adapters'})
        interaction_params = list(self.change_guided_roi.parameters()) + list(self.spatial_modulation.parameters()) + list(self.box_consistency_loss.parameters())
        param_groups.append({'params': interaction_params, 'lr': lr, 'weight_decay': weight_decay, 'name': 'interaction'})
        param_groups.append({'params': list(self.instance_head.parameters()), 'lr': lr, 'weight_decay': weight_decay, 'name': 'instance_head'})
        label_params = list(self.label_structure.parameters())
        param_groups.append({'params': label_params, 'lr': lr*0.5, 'weight_decay': weight_decay, 'name': 'label_structure'})
        param_groups.append({'params': list(self.semantic_head.parameters()), 'lr': lr, 'weight_decay': weight_decay, 'name': 'semantic_head'})
        return param_groups


def build_hicd_v7(config):
    return HICD_v7(config)
