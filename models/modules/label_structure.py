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
    'Farmland', 'Runway', 'Taxiway', 'Apron', 'Bridge', 'Highway',
    'Building', 'Shelter', 'Tower', 'Pier', 'Dock', 'Tank',
    'Aircraft', 'Vessel', 'Crater', 'Background',
]

TARGET_NAMES_ZH = [
    '农田', '跑道', '滑行道', '停机坪', '桥梁', '主要公路',
    '建筑物', '飞机掩体', '塔台', '栈桥', '船坞', '大型罐体',
    '飞机', '舰船', '弹坑', '背景',
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
        (0, 0), (0, 2),  # farmland: no-change, reduced
    (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),  # runway: nochange, damaged, reduced, added, extended
    (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),  # taxiway
    (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),  # apron
    (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),  # bridge
    (5, 0), (5, 1), (5, 2), (5, 3), (5, 4),  # highway
    (6, 0), (6, 1), (6, 2), (6, 3), (6, 4),  # building
    (7, 0), (7, 1), (7, 2), (7, 3), (7, 4),  # shelter
    # Tower (8) removed - mapped to ignore
    (9, 0), (9, 1), (9, 2), (9, 3), (9, 4),  # pier
    (10, 0), (10, 1), (10, 2), (10, 3), (10, 4),  # dock
    (11, 0), (11, 1), (11, 2), (11, 3),  # tank (no extended)
    (12, 0), (12, 1), (12, 2), (12, 3), (12, 5),  # aircraft: no extended, has replaced
    (13, 0), (13, 1), (13, 2), (13, 3), (13, 5),  # vessel: no extended, has replaced
    (14, 3),  # crater: added (originally stateless, relabeled to added)
    (15, 6),  # background: only 'none' state
]

# Which states are valid for each target (sorted)
TARGET_VALID_STATES = {}
for tid, sid in VALID_COMBINATIONS:
    if tid not in TARGET_VALID_STATES:
        TARGET_VALID_STATES[tid] = []
    TARGET_VALID_STATES[tid].append(sid)

NUM_TARGETS = 16  # 14 real targets + Tower(ignored) + Background
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
        Soft validity weighting (replaces hard -inf masking).

        Instead of setting invalid state logits to -inf (which kills gradients
        when target predictions are wrong early in training), apply a soft
        probability prior with a floor of 0.1, added as log(weight) to logits.
        """
        target_probs = F.softmax(target_logits, dim=1)  # [B, T, H, W]
        validity = self.get_validity_score()  # [T, S] in [0, 1]
        weighted_validity = torch.einsum('bthw,ts->bshw', target_probs, validity)
        # Soft floor: min 0.1 so even invalid states get gradient signal
        soft_weight = 0.1 + 0.9 * weighted_validity
        state_logits_weighted = state_logits + soft_weight.clamp_min(1e-6).log()
        return state_logits_weighted, validity
    
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
        
        # GNN soft validity mask: suppress invalid states softly.
        # Valid combos get weight ~1.0, invalid get weight ~0.1 (preserves gradient).
        if label_structure is not None:
            validity = label_structure.get_validity_score()  # [T, S] soft scores
            target_probs = F.softmax(target_logits.detach(), dim=1)  # [B, T, H, W]
            soft_weight = torch.einsum('bthw,ts->bshw', target_probs, validity)
            soft_weight = 0.1 + 0.9 * soft_weight  # min 0.1 keeps gradient alive
            # A probability prior is additive in logit space. Multiplication
            # would increase a negative logit when the weight is small.
            state_logits = state_logits + soft_weight.clamp_min(1e-6).log()
        
        return state_logits, features


class PixelStateHead(nn.Module):
    """
    Complete pixel branch head with graph-structured label space.
    
    Replaces the old SemanticHead with:
    1. FPN + multi-scale fusion (unchanged)
    2. Target prediction head (unchanged)
    3. Hierarchical condition decoder for state (NEW)
    4. Validity masking via LabelStructure (NEW)
    5. Change detection via cosine similarity (removed (non-learnable, redundant with state))
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
    
    def forward(self, features, label_structure=None):
        """
        Args:
            features: list of [B, C, H_i, W_i] fused features (P2-P5)
            label_structure: LabelStructure module for validity masking
        Returns:
            target_logits: [B, num_targets, H, W]
            state_logits: [B, num_states, H, W]
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
        
        
        return target_logits, state_logits
    







class GNNLabelStructure(nn.Module):
    """
    GNN-based label space structure module.

    Replaces static embeddings + manual contrastive losses with graph
    convolutions that naturally encode target-state validity, category
    hierarchy, and state grouping in a single differentiable pass.

    Graph topology (22 nodes total):
      - 15 target nodes (indices 0..14)
      -  7 state  nodes (indices 15..21)
      - Edges: target-state bipartite (valid combos),
               target-target (same category),
               state-state  (same group)

    Args:
        num_targets: 15
        num_states:  7
        embed_dim:   node embedding dimension (default 64)
        num_gnn_layers: message-passing depth (default 2)
    """

    def __init__(self, num_targets=NUM_TARGETS, num_states=NUM_STATES,
                 embed_dim=64, num_gnn_layers=2):
        super().__init__()
        self.num_targets = num_targets
        self.num_states = num_states
        self.embed_dim = embed_dim
        self.num_nodes = num_targets + num_states  # 22

        # --- Build adjacency (dense, registered as buffer) ---
        adj = self._build_adjacency(num_targets, num_states)
        self.register_buffer('adj', adj)  # [22, 22]

        # --- Node feature initialised from learnable embeddings ---
        self.node_embeddings = nn.Embedding(self.num_nodes, embed_dim)
        nn.init.xavier_uniform_(self.node_embeddings.weight)

        # --- GNN layers: Graph Attention with residual ---
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(
                GATLayer(embed_dim, num_heads=4, residual=True)
            )
            self.gnn_norms.append(nn.LayerNorm(embed_dim))

        # --- Validity predictor on (target, state) pair embeddings ---
        self.validity_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1),
        )

        # --- HierarchicalConditionDecoder compatibility ---
        # Provide the same interface as the old LabelStructure
        self.target_state_bias = nn.Parameter(torch.zeros(num_targets, num_states))

        # Validity mask buffer (hard, for masking)
        validity = build_validity_matrix(num_targets, num_states)
        self.register_buffer('validity_mask', validity)

    # ---- graph construction ----

    @staticmethod
    def _build_adjacency(num_targets, num_states):
        """Build [N, N] symmetric adjacency for the heterogeneous label graph."""
        N = num_targets + num_states
        adj = torch.zeros(N, N)

        # 1) target-state bipartite edges (valid combinations)
        for tid, sid in VALID_COMBINATIONS:
            adj[tid, num_targets + sid] = 1.0
            adj[num_targets + sid, tid] = 1.0

        # 2) target-target: same category
        for cat_list in CATEGORY_HIERARCHY.values():
            for i in cat_list:
                for j in cat_list:
                    if i != j:
                        adj[i, j] = 1.0

        # 3) state-state: same group
        state_groups = [[0], [1, 2], [3, 4], [5], [6]]  # nochange, decrease, increase, replaced, none
        for grp in state_groups:
            for si in grp:
                for sj in grp:
                    if si != sj:
                        adj[num_targets + si, num_targets + sj] = 1.0

        # Add self-loops
        adj += torch.eye(N)
        return adj

    # ---- forward ----

    def forward(self):
        """
        Run GNN message passing and return refined embeddings.

        Returns:
            target_emb: [num_targets, embed_dim]
            state_emb:  [num_states,  embed_dim]
        """
        x = self.node_embeddings.weight  # [22, D]

        for gnn, norm in zip(self.gnn_layers, self.gnn_norms):
            x_new = gnn(x, self.adj)
            x = norm(x + x_new)  # residual + layer norm

        target_emb = x[:self.num_targets]
        state_emb = x[self.num_targets:]
        return target_emb, state_emb

    def get_validity_score(self):
        """
        Predict soft validity [num_targets, num_states] from GNN embeddings.
        Falls back to hard mask * sigmoid(logits) for compatibility.
        """
        target_emb, state_emb = self.forward()
        # Pairwise MLP
        t = target_emb.unsqueeze(1).expand(-1, self.num_states, -1)  # [T, S, D]
        s = state_emb.unsqueeze(0).expand(self.num_targets, -1, -1)  # [T, S, D]
        pair = torch.cat([t, s], dim=-1)  # [T, S, 2D]
        validity = torch.sigmoid(self.validity_mlp(pair).squeeze(-1))  # [T, S]
        return validity

    def mask_invalid_logits(self, target_logits, state_logits):
        """Soft validity weighting (same interface as old LabelStructure)."""
        target_probs = F.softmax(target_logits, dim=1)
        validity = self.get_validity_score()
        weighted_validity = torch.einsum('bthw,ts->bshw', target_probs, validity)
        soft_weight = 0.1 + 0.9 * weighted_validity
        state_logits_weighted = state_logits + soft_weight.clamp_min(1e-6).log()
        return state_logits_weighted, validity

    def compute_contrastive_losses(self, target_logits, state_logits, target_gt, state_gt):
        """
        GNN version: the graph structure is learned through message passing,
        so we only need a lightweight supervision signal on the validity predictor.
        No more manual contrastive losses.
        """
        losses = {}
        # Supervise validity: BCE with hard GT mask
        validity = self.get_validity_score()
        # BCE with logits: use raw MLP output before sigmoid
        target_emb, state_emb = self.forward()
        t = target_emb.unsqueeze(1).expand(-1, self.num_states, -1)
        s = state_emb.unsqueeze(0).expand(self.num_targets, -1, -1)
        pair = torch.cat([t, s], dim=-1)
        raw_logits = self.validity_mlp(pair).squeeze(-1)
        losses['validity_supervision'] = F.binary_cross_entropy_with_logits(
            raw_logits, self.validity_mask,
            pos_weight=torch.tensor([5.0], device=raw_logits.device)
        )
        return losses

    def get_target_state_embeddings(self):
        target_emb, state_emb = self.forward()
        return {
            'target_embeddings': target_emb.detach().cpu(),
            'state_embeddings': state_emb.detach().cpu(),
            'validity_matrix': self.get_validity_score().detach().cpu(),
        }


class GATLayer(nn.Module):
    """
    Graph Attention Layer (Velic'kovic et al. 2018) with residual support.

    Operates on a dense adjacency matrix (small graph, N=22).
    """

    def __init__(self, embed_dim, num_heads=4, residual=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        self.residual = residual

        self.W = nn.Linear(embed_dim, embed_dim, bias=False)
        self.a_src = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, adj):
        """
        Args:
            x:   [N, D] node features
            adj: [N, N] adjacency (0/1, with self-loops)
        Returns:
            [N, D] updated node features
        """
        N, D = x.shape
        h = self.W(x).view(N, self.num_heads, self.head_dim)  # [N, H, d]

        # Attention scores: e_ij = LeakyReLU(a_src . h_i + a_dst . h_j)
        e_src = (h * self.a_src).sum(dim=-1)  # [N, H]
        e_dst = (h * self.a_dst).sum(dim=-1)  # [N, H]
        e = e_src.unsqueeze(1) + e_dst.unsqueeze(0)  # [N, N, H]
        e = self.leaky_relu(e)

        # Mask non-edges
        mask = (adj == 0).unsqueeze(-1)  # [N, N, 1]
        e = e.masked_fill(mask, float('-inf'))

        # Softmax over neighbours
        alpha = F.softmax(e, dim=1)  # [N, N, H]
        alpha = alpha.masked_fill(mask, 0.0)

        # Aggregate
        h_expanded = h.unsqueeze(0).expand(N, -1, -1, -1)  # [N_src, N_dst, H, d]
        out = (alpha.unsqueeze(-1) * h_expanded).sum(dim=1)  # [N, H, d]
        out = out.reshape(N, D)
        out = self.out_proj(out)
        return out
