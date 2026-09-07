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
    LearnableStateEmbeddings,
    GNNLabelStructure as LabelStructure,
    PixelStateHead,
)
from .heads.proposal_head import VotingRefinementModule
from .modules.frequency_branch import FrequencyBranch
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

        self.label_structure = LabelStructure(
            num_targets=num_pixel_targets, num_states=num_states, embed_dim=embed_dim,
        )
        self.instance_head = VotingRefinementModule(num_states=num_states, roi_size=7)
        self.semantic_head = PixelStateHead(
            num_targets=num_pixel_targets, num_states=num_states,
            in_channels=hidden_dim, embed_dim=embed_dim, hidden_dim=hidden_dim,
        )
        
        # Frequency branch: independent frequency domain features for small class enhancement
        freq_cfg = config.get('frequency_branch', {})
        self.use_freq_branch = freq_cfg.get('enabled', False)
        if self.use_freq_branch:
            self.freq_branch = FrequencyBranch(
                in_channels=3,
                out_dim=freq_cfg.get('out_dim', 64),
                levels=freq_cfg.get('levels', 2),
            )
            # Fusion layer: combine backbone features with frequency features
            self.freq_fusion = nn.Sequential(
                nn.Conv2d(hidden_dim + freq_cfg.get('out_dim', 64), hidden_dim, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )
            print(f"FrequencyBranch enabled: out_dim={freq_cfg.get('out_dim', 64)}, levels={freq_cfg.get('levels', 2)}")

        config['num_pixel_targets'] = num_pixel_targets
        self.loss_fn = V7Loss(config)

    def forward(self, img_t1, img_t2, targets=None):
        B = img_t1.shape[0]

        # Backbone + temporal fusion
        feat_t1_list = self.backbone(img_t1)
        feat_t2_list = self.backbone(img_t2)
        nfa_t1 = self.nfa_t1(*feat_t1_list)
        nfa_t2 = self.nfa_t2(*feat_t2_list)
        fused_features, _, _ = self.tfm(nfa_t1, nfa_t2)

        # Frequency branch: enhance features with frequency domain information
        if self.use_freq_branch:
            # fused_features is a list of multi-scale features [P2, P3, P4, P5]
            # Enhance the highest resolution feature (P2) with frequency features
            p2_size = fused_features[0].shape[2:]  # (H/4, W/4)
            freq_features = self.freq_branch(img_t1, img_t2, target_size=p2_size)
            # Fuse with P2 features
            enhanced_p2 = self.freq_fusion(torch.cat([fused_features[0], freq_features], dim=1))
            fused_features = [enhanced_p2] + fused_features[1:]

        # Branch B+C: Target + state classification
        target_logits, state_logits = self.semantic_head(
            fused_features, label_structure=self.label_structure,
        )

        # Instance branch: voting refinement (inference-only post-processing)
        validity_matrix = self.label_structure.get_validity_score()
        if not self.training:
            # Only run refinement during inference
            refined_state_logits, refine_consistency, proposals = self.instance_head(
                state_logits_pixel=state_logits,
                target_logits=target_logits,
                validity_matrix=validity_matrix.detach(),
            )
        else:
            # During training, skip refinement to save memory
            refined_state_logits = state_logits
            refine_consistency = torch.tensor(0.0, device=state_logits.device, requires_grad=True)
            proposals = []

        # Contrastive losses from label structure
        contrastive_losses = {}
        if targets is not None:
            target_gt = targets.get('target_gt')
            state_gt = targets.get('state_gt')
            if target_gt is not None and state_gt is not None:
                contrastive_losses = self.label_structure.compute_contrastive_losses(
                    target_logits, state_logits, target_gt, state_gt,
                )

        predictions = {
            'target_logits': target_logits,
            'state_logits': state_logits,  # Raw: for main state loss
            'refined_state_logits': refined_state_logits,  # Refined: for eval
            'proposals': proposals,
            'refine_consistency_loss': refine_consistency,
            'contrastive_losses': contrastive_losses,
            'validity_matrix': validity_matrix,
        }
        if targets is not None:
            loss, loss_dict = self.loss_fn(predictions, targets)
        else:
            loss = torch.tensor(0.0, device=img_t1.device, requires_grad=True)
            loss_dict = {}
        return predictions, loss, loss_dict


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
        param_groups.append({'params': list(self.instance_head.parameters()), 'lr': lr, 'weight_decay': weight_decay, 'name': 'instance_head'})
        label_params = list(self.label_structure.parameters())
        param_groups.append({'params': label_params, 'lr': lr*0.5, 'weight_decay': weight_decay, 'name': 'label_structure'})
        param_groups.append({'params': list(self.semantic_head.parameters()), 'lr': lr, 'weight_decay': weight_decay, 'name': 'semantic_head'})
        return param_groups


def build_hicd_v7(config):
    return HICD_v7(config)
