"""
Graph-Structured Label Space for HICD v7

Core insight: the 69-class war damage label space forms a sparse bipartite graph
between 16 targets and 7 states, where each target has only 2-5 valid states.
This module learns this graph structure as embeddings and uses it to:
1. Condition state prediction on target prediction (constrained decoder)
2. Learn contrastive embeddings that separate valid/invalid combinations
3. Encode category-level hierarchy (Surface Base, Fixed, Mobile, Stateless)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product


# === 69-class -> (target_id, state_id) mapping ===
# Target classes (16 targets):
#   0=farmland, 1=runway, 2=taxiway, 3=apron,
#   4=bridge, 5=highway, 6=building, 7=shelter, 8=tower,
#   9=pier, 10=dock, 11=tank, 12=aircraft, 13=vessel, 14=crater
# States (7 states):
#   0=no-change, 1=damaged, 2=reduced, 3=added, 4=extended, 5=replaced, 6=none(stateless)
# Category hierarchy:
#   Surface Base: [1,2,3,4,5,6]  Fixed Targets: [7,8,9,10,11,12]
#   Mobile Targets: [13,14]  Stateless Targets: [15,16]

TARGET_NAMES = [
    'background', 'farmland', 'runway', 'taxiway', 'apron',
    'bridge', 'highway', 'building', 'shelter', 'tower',
    'pier', 'dock', 'tank', 'aircraft', 'vessel', 'crater',
]

TARGET_NAMES_ZH = [
    '背景', '农田', '跑道', '滑行道', '停机坪',
    '桥梁', '主要公路', '建筑物', '飞机掩体', '塔台',
    '栈桥', '船坞', '大型罐体', '飞机', '舰船', '弹坑',
]

STATE_NAMES = ['no-change', 'damaged', 'reduced', 'added', 'extended', 'replaced', 'none']

CATEGORY_HIERARCHY = {
    'Surface Base':    [0, 1, 2, 3, 4, 5],
    'Fixed Targets':   [6, 7, 8, 9, 10, 11],
    'Mobile Targets':  [12, 13],
    'Stateless Targets': [14]
}

# Valid (target, state) pairs derived from classes.csv
# Each tuple is (target_id, state_id)
VALID_COMBINATIONS = [
        (0, 0),   # farmland: nochange (damaged merged into no-change, visible damage indistinguishable)
    (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),  # runway: nochange, damaged, reduced, added, extended
    (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),  # taxiway
    (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),  # apron
    (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),  # bridge
    (5, 0), (5, 1), (5, 2), (5, 3), (5, 4),  # highway
    (6, 0), (6, 1), (6, 2), (6, 3), (6, 4),  # building
    (7, 0), (7, 1), (7, 2), (7, 3), (7, 4),  # shelter
    (8, 0), (8, 1), (8, 2), (8, 3), (8, 4),  # tower
    (9, 0), (9, 1), (9, 2), (9, 3), (9, 4),  # pier
    (10, 0), (10, 1), (10, 2), (10, 3), (10, 4),  # dock
    (11, 0), (11, 1), (11, 2), (11, 3),  # tank (no extended)
    (12, 0), (12, 1), (12, 2), (12, 3), (12, 5),  # aircraft: no extended, has replaced
    (13, 0), (13, 1), (13, 2), (13, 3), (13, 5),  # vessel: no extended, has replaced
    (14, 6),  # crater: stateless (none only)
]

# Which states are valid for each target (sorted)
TARGET_VALID_STATES = {}
for tid, sid in VALID_COMBINATIONS:
    if tid not in TARGET_VALID_STATES:
        TARGET_VALID_STATES[tid] = []
    TARGET_VALID_STATES[tid].append(sid)

NUM_TARGETS = 15
NUM_STATES = 7


def build_validity_matrix(num_targets=NUM_TARGETS, num_states=NUM_STATES):
    """Build binary validity matrix [num_targets, num_states]."""
    mat = torch.zeros(num_targets, num_states)
    for tid, sid in VALID_COMBINATIONS:
        mat[tid, sid] = 1.0
    return mat


def build_state_mask_from_targets(target_logits):
    """From target logits [B, num_targets, H, W], return per-pixel state mask [B, num_states, H, W]."""
    # Get most likely target
    target_pred = target_logits.argmax(dim=1)  # [B, H, W]
    B, H, W = target_pred.shape
    num_states = len(STATE_NAMES)
    mask = torch.zeros(B, num_states, H, W, device=target_logits.device)
    # Gather valid states per target
    for tid in range(len(TARGET_NAMES)):
        valid_sids = TARGET_VALID_STATES.get(tid, [0])
        mask[:, valid_sids] = 1.0
    # Index per pixel
    mask_per_pixel = mask[target_pred]  # [B, H, W, num_states]
    return mask_per_pixel.permute(0, 3, 1, 2)  # [B, num_states, H, W]


class LabelStructure(nn.Module):
    """
    Graph-structured label space module.
    
    Maintains learnable embeddings for targets and states that encode:
    - Validity: valid pairs have high similarity, invalid pairs low
    - Category hierarchy: same-category targets cluster together
    - State semantics: damaged/reduced (decrease) vs added/extended (increase)
    
    Args:
        num_targets: number of target classes (including background)
        num_states: number of state classes
        embed_dim: embedding dimension
        category_ids: list of lists, each inner list is target IDs in one category
    """
    
    def __init__(self, num_targets=NUM_TARGETS, num_states=NUM_STATES, 
                 embed_dim=64, category_ids=None):
        super().__init__()
        self.num_targets = num_targets
        self.num_states = num_states
        self.embed_dim = embed_dim
        
        # Learnable target and state embeddings
        self.target_embeddings = nn.Embedding(num_targets, embed_dim)
        self.state_embeddings = nn.Embedding(num_states, embed_dim)
        
        # Validity mask (buffer, not learnable)
        validity = build_validity_matrix(num_targets, num_states)
        self.register_buffer('validity_mask', validity)
        
        # Learnable validity refinement (small perturbation on top of binary mask)
        self.validity_logits = nn.Parameter(torch.zeros(num_targets, num_states))
        
        # Category hierarchy: which targets belong to same category
        if category_ids is None:
            category_ids = list(CATEGORY_HIERARCHY.values())
        # Build adjacency: [num_targets, num_targets], 1 if same category
        cat_adj = torch.zeros(num_targets, num_targets)
        for cat_targets in category_ids:
            for i in cat_targets:
                for j in cat_targets:
                    if i != j:
                        cat_adj[i, j] = 1.0
        self.register_buffer('category_adjacency', cat_adj)
        
        # State group embedding (decrease/increase/stateless)
        self.state_group_proj = nn.Linear(embed_dim, 3)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.target_embeddings.weight)
        nn.init.xavier_uniform_(self.state_embeddings.weight)
        
        # Projection for combined target-state pair embedding
        self.pair_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def get_validity_score(self):
        """Soft validity score [num_targets, num_states] in [0,1]."""
        return torch.sigmoid(self.validity_logits)
    
    def mask_invalid_logits(self, target_logits, state_logits):
        """
        Zero out invalid (target, state) combinations.
        
        Args:
            target_logits: [B, num_targets, H, W]
            state_logits: [B, num_states, H, W]
        Returns:
            state_logits_masked: [B, num_states, H, W] with invalid logits set to -inf
            validity: [num_targets, num_states] soft validity
        """
        # Get per-pixel validity from target prediction
        target_probs = F.softmax(target_logits, dim=1)  # [B, T, H, W]
        validity = self.get_validity_score()  # [T, S]
        
        # Weighted validity: sum over targets of (target_prob * validity[target, :])
        # This gives per-pixel valid state distribution [B, S, H, W]
        weighted_validity = torch.einsum('bthw,ts->bshw', target_probs, validity)
        
        # Mask: states with very low validity get penalized
        mask = (weighted_validity < 0.01).float()
        state_logits_masked = state_logits.masked_fill(mask.bool(), float('-inf'))
        
        return state_logits_masked, validity
    
    def compute_contrastive_losses(self, target_logits, state_logits, target_gt, state_gt):
        """
        Compute contrastive losses that train the embeddings.
        
        Args:
            target_logits: [B, num_targets, H, W]
            state_logits: [B, num_states, H, W]  
            target_gt: [B, H, W] ground truth target IDs
            state_gt: [B, H, W] ground truth state IDs
        Returns:
            dict of loss values
        """
        B, H, W = target_gt.shape
        device = target_logits.device
        losses = {}
        
        # 0. Validity logits supervision (data-driven)
        # Push validity_logits toward the true valid/invalid structure
        data_validity = torch.zeros(self.num_targets, self.num_states, device=device)
        for t, s in VALID_COMBINATIONS:
            data_validity[t, s] = 1.0
        loss_validity_bce = F.binary_cross_entropy_with_logits(
            self.validity_logits, data_validity,
            pos_weight=torch.tensor([5.0], device=device)
        )
        losses['validity_supervision'] = loss_validity_bce
        
        # 1. Contrastive target-state validity loss
        # Sample pixels and encourage valid pairs, discourage invalid pairs
        valid_pairs = torch.tensor(
            VALID_COMBINATIONS,
            device=device, dtype=torch.long
        )
        invalid_pairs = []
        all_pairs = set((t, s) for t in range(self.num_targets) for s in range(self.num_states))
        valid_set = set((t, s) for t, s in VALID_COMBINATIONS)
        invalid_list = list(all_pairs - valid_set)
        # Subsample invalid pairs to match valid count
        if len(invalid_list) > len(valid_pairs):
            indices = torch.randperm(len(invalid_list), device=device)[:len(valid_pairs)]
            invalid_pairs = [invalid_list[i] for i in indices]
        else:
            invalid_pairs = invalid_list
        invalid_pairs = torch.tensor(invalid_pairs, device=device, dtype=torch.long)
        
        if len(valid_pairs) > 0 and len(invalid_pairs) > 0:
            # Get embeddings
            t_emb_valid = self.target_embeddings(valid_pairs[:, 0])  # [N_valid, D]
            s_emb_valid = self.state_embeddings(valid_pairs[:, 1])
            pair_emb_valid = self.pair_proj(torch.cat([t_emb_valid, s_emb_valid], dim=1))
            
            t_emb_invalid = self.target_embeddings(invalid_pairs[:, 0])
            s_emb_invalid = self.state_embeddings(invalid_pairs[:, 1])
            pair_emb_invalid = self.pair_proj(torch.cat([t_emb_invalid, s_emb_invalid], dim=1))
            
            # Pull valid pairs together, push invalid pairs apart
            valid_sim = F.cosine_similarity(pair_emb_valid, pair_emb_valid.mean(0, keepdim=True), dim=1)
            invalid_sim = F.cosine_similarity(pair_emb_invalid, pair_emb_valid.mean(0, keepdim=True), dim=1)
            
            losses['contrastive_validity'] = F.relu(invalid_sim.mean() - valid_sim.mean() + 0.5)
        
        # 2. Category hierarchy loss
        # Target embeddings of same category should be closer than different categories
        cat_targets = []
        for cat_list in CATEGORY_HIERARCHY.values():
            non_bg = cat_list
            if len(non_bg) > 1:
                cat_targets.append(non_bg)
        
        if cat_targets:
            cat_loss = torch.tensor(0.0, device=device)
            for targets_in_cat in cat_targets:
                t_embs = self.target_embeddings.weight[targets_in_cat]  # [K, D]
                # Intra-category distance (should be small)
                intra_dist = torch.cdist(t_embs.unsqueeze(0), t_embs.unsqueeze(0)).mean()
                # Inter-category distance (should be large)
                other_targets = [t for cat in cat_targets if cat != targets_in_cat for t in cat]
                if other_targets:
                    other_embs = self.target_embeddings.weight[other_targets]
                    inter_dist = torch.cdist(t_embs.unsqueeze(0), other_embs.unsqueeze(0)).mean()
                    cat_loss = cat_loss + F.relu(intra_dist - inter_dist + 0.5)
            losses['contrastive_hierarchy'] = cat_loss / len(cat_targets)
        
        # 3. State group loss
        # Damaged/reduced (decrease=1) should be closer; added/extended (increase=2) closer
        state_groups = {
            0: [0],       # no-change
            1: [1, 2],    # decrease
            2: [3, 4],    # increase  
            3: [5],       # replaced
            4: [6],       # none
        }
        group_loss = torch.tensor(0.0, device=device)
        group_count = 0
        for gid, sids in state_groups.items():
            if len(sids) > 1:
                s_embs = self.state_embeddings.weight[sids]
                intra = torch.cdist(s_embs.unsqueeze(0), s_embs.unsqueeze(0)).mean()
                other_sids = [s for g, ss in state_groups.items() if g != gid for s in ss]
                if other_sids:
                    other_embs = self.state_embeddings.weight[other_sids]
                    inter = torch.cdist(s_embs.unsqueeze(0), other_embs.unsqueeze(0)).mean()
                    group_loss = group_loss + F.relu(intra - inter + 0.5)
                    group_count += 1
        if group_count > 0:
            losses['contrastive_state_group'] = group_loss / group_count
        
        return losses
    
    def get_target_state_embeddings(self):
        """Return current target and state embeddings for visualization/analysis."""
        return {
            'target_embeddings': self.target_embeddings.weight.detach().cpu(),
            'state_embeddings': self.state_embeddings.weight.detach().cpu(),
            'validity_matrix': self.validity_mask.clone(),
        }


class HierarchicalConditionDecoder(nn.Module):
    """
    Condition decoder that predicts state conditioned on target.
    
    Instead of independent target + state heads, this module:
    1. Takes target logits as conditioning signal
    2. Modulates state logits using learnable target-state interactions
    3. Applies validity masking
    
    This implements the "state prediction conditioned on target" idea.
    """
    
    def __init__(self, in_channels, num_targets, num_states, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.num_targets = num_targets
        self.num_states = num_states
        
        # Target-conditioned state projection
        # Takes fused features + target prediction context -> state logits
        self.target_condition_proj = nn.Sequential(
            nn.Conv2d(in_channels + num_targets, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_states, 1)
        )
        
        # Learnable target-to-state bias: how target identity biases state prediction
        # [num_targets, num_states] -> per-target state prior
        self.target_state_bias = nn.Parameter(torch.zeros(num_targets, num_states))
        
        # Validity mask temperature
        self.mask_temperature = nn.Parameter(torch.tensor(10.0))
    
    def forward(self, features, target_logits, label_structure=None):
        """
        Args:
            features: [B, C, H, W] fused features from FPN
            target_logits: [B, num_targets, H, W] target predictions
            label_structure: LabelStructure module (for validity masking)
        Returns:
            state_logits: [B, num_states, H, W]
            target_context: [B, C, H, W] target-conditioned features
        """
        B, C, H, W = features.shape
        
        # Concatenate features with target prediction context
        target_context = F.softmax(target_logits, dim=1)  # [B, T, H, W]
        state_input = torch.cat([features, target_context], dim=1)
        
        # Target-conditioned state logits
        state_logits_raw = self.target_condition_proj(state_input)
        
        # Add per-target state bias
        # target_probs: [B, T, H, W] -> [B, T, H, W] * [T, S] -> [B, S, H, W]
        target_probs = F.softmax(target_logits.detach(), dim=1)
        state_bias = torch.einsum('bthw,ts->bshw', target_probs, self.target_state_bias)
        state_logits = state_logits_raw + state_bias
        
        # Apply validity masking if label_structure is provided
        if label_structure is not None:
            state_logits = label_structure.mask_invalid_logits(target_logits, state_logits)[0]
        
        return state_logits, features


class PixelStateHead(nn.Module):
    """
    Complete pixel branch head with graph-structured label space.
    
    Replaces the old SemanticHead with:
    1. FPN + multi-scale fusion (unchanged)
    2. Target prediction head (unchanged)
    3. Hierarchical condition decoder for state (NEW)
    4. Validity masking via LabelStructure (NEW)
    5. Change detection via cosine similarity (unchanged)
    """
    
    def __init__(self, num_targets, num_states, in_channels=128, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.num_targets = num_targets
        self.num_states = num_states
        self.in_channels = in_channels
        
        # FPN (reuse from old SemanticHead)
        from ..heads.semantic_head import FPN
        self.fpn = FPN(
            in_channels_list=[in_channels] * 4,
            out_channels=in_channels
        )
        
        # Multi-scale fusion
        self.ms_fuse = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        
        # Target head (unchanged)
        self.target_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, num_targets, 1)
        )
        
        # State head: target-conditioned decoder (NEW)
        self.state_decoder = HierarchicalConditionDecoder(
            in_channels=in_channels,
            num_targets=num_targets,
            num_states=num_states,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
        )
    
    def forward(self, features, t1_features=None, t2_features=None, label_structure=None):
        """
        Args:
            features: list of [B, C, H_i, W_i] fused features (P2-P5)
            t1_features: T1-specific features for change detection
            t2_features: T2-specific features for change detection
            label_structure: LabelStructure module for validity masking
        Returns:
            target_logits: [B, num_targets, H, W]
            state_logits: [B, num_states, H, W]
            change_logits: [B, 2, H, W]
        """
        fpn_features = self.fpn(features)
        p2 = fpn_features[0]
        p3 = fpn_features[1]
        p4 = fpn_features[2]
        p5 = fpn_features[3]
        B, C, H, W = p2.shape
        
        # Multi-scale fusion
        p3_up = F.interpolate(p3, size=(H, W), mode='bilinear', align_corners=True)
        p4_up = F.interpolate(p4, size=(H, W), mode='bilinear', align_corners=True)
        p5_up = F.interpolate(p5, size=(H, W), mode='bilinear', align_corners=True)
        fused = self.ms_fuse(torch.cat([p2, p3_up, p4_up, p5_up], dim=1))
        
        # Target prediction
        target_logits = self.target_head(fused)
        
        # State prediction conditioned on target
        state_logits, _ = self.state_decoder(fused, target_logits, label_structure)
        
        # Change detection (unchanged)
        change_logits = self._compute_change(t1_features, t2_features, p2.shape)
        
        return target_logits, state_logits, change_logits
    
    def _compute_change(self, t1_features, t2_features, ref_shape):
        """Cosine similarity change detection (unchanged)."""
        B, C, H, W = ref_shape
        if t1_features is not None and t2_features is not None:
            t1_fpn = self.fpn(t1_features)
            t2_fpn = self.fpn(t2_features)
            t1_p3_up = F.interpolate(t1_fpn[1], size=(H, W), mode='bilinear', align_corners=True)
            t1_p4_up = F.interpolate(t1_fpn[2], size=(H, W), mode='bilinear', align_corners=True)
            t1_p5_up = F.interpolate(t1_fpn[3], size=(H, W), mode='bilinear', align_corners=True)
            t1_fused = self.ms_fuse(torch.cat([t1_fpn[0], t1_p3_up, t1_p4_up, t1_p5_up], dim=1))
            t2_p3_up = F.interpolate(t2_fpn[1], size=(H, W), mode='bilinear', align_corners=True)
            t2_p4_up = F.interpolate(t2_fpn[2], size=(H, W), mode='bilinear', align_corners=True)
            t2_p5_up = F.interpolate(t2_fpn[3], size=(H, W), mode='bilinear', align_corners=True)
            t2_fused = self.ms_fuse(torch.cat([t2_fpn[0], t2_p3_up, t2_p4_up, t2_p5_up], dim=1))
            cos_sim = F.cosine_similarity(t1_fused, t2_fused, dim=1)
            cos_sim = cos_sim * 0.5 + 0.5
            change_logits = torch.stack([1 - cos_sim, cos_sim], dim=1)
        else:
            change_logits = torch.zeros(B, 2, H, W, device=ref_shape[0].device 
                                         if torch.is_tensor(ref_shape[0]) else 'cpu')
        return change_logits




