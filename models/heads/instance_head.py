"""
Instance Branch Head (DETR-based) - V7
Object detection for instance-level change detection.
Returns instance_features for interaction layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class DETRInstanceHead(nn.Module):
    """
    DETR-based Instance Head for object detection (V7)
    
    Same as V6 but returns instance_features (decoder output) for the
    interaction layer's ChangeTypeClassifier.
    """
    
    def __init__(self, num_classes, num_states=0, hidden_dim=128, num_queries=100, 
                 nhead=8, num_decoder_layers=6):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_states = num_states
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        
        # Object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Positional encoding
        self.pos_embed = nn.Embedding(4, hidden_dim)
        
        # DETR Decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # Prediction heads
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        )
        
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
        
        # State classification head
        if num_states > 0:
            self.state_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_states)
            )
        else:
            self.state_head = None
        
        # Feature projection
        self.input_proj = nn.Conv2d(hidden_dim, hidden_dim, 1)
    
    def forward(self, features, change_heatmap=None):
        """
        Args:
            features: [B, C, H, W] �� instance adapter output
            change_heatmap: [B, 1, H/4, W/4] �� optional change heatmap for guided attention
            
        Returns:
            class_logits: [B, num_queries, num_classes + 1]
            bbox_pred: [B, num_queries, 4]
            state_logits: [B, num_queries, num_states] or None
            instance_features: [B, num_queries, D] �� decoder output for interaction layer
        """
        B = features.shape[0]
        
        # Project features
        src = self.input_proj(features)  # [B, C, H, W]
        H, W = src.shape[2], src.shape[3]
        
        # Flatten spatial dimensions
        src = src.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        
        # Positional encoding
        pos = self._get_pos_embed(H, W, src.device)
        pos = pos.expand(B, -1, -1)
        
        # Object queries
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        
        # Decode
        decoder_out = self.decoder(
            tgt=queries,
            memory=src + pos
        )  # [B, num_queries, C]
        
        # Predict
        class_logits = self.class_head(decoder_out)
        bbox_pred = self.bbox_head(decoder_out).sigmoid()
        
        state_logits = None
        if self.state_head is not None:
            state_logits = self.state_head(decoder_out)
        
        return class_logits, bbox_pred, state_logits, decoder_out
    
    def _get_pos_embed(self, H, W, device):
        """Generate 2D sinusoidal positional encoding"""
        pos_embed = torch.zeros(H * W, self.hidden_dim, device=device)
        
        y_pos = torch.arange(H, device=device).unsqueeze(1).expand(H, W).float()
        x_pos = torch.arange(W, device=device).unsqueeze(0).expand(H, W).float()
        
        y_pos = y_pos / max(H - 1, 1)
        x_pos = x_pos / max(W - 1, 1)
        
        y_pos = y_pos.reshape(-1)
        x_pos = x_pos.reshape(-1)
        
        dim_t = torch.arange(self.hidden_dim // 4, device=device).float()
        dim_t = 10000 ** (2 * dim_t / (self.hidden_dim // 2))
        
        pos_x = x_pos.unsqueeze(1) / dim_t.unsqueeze(0)
        pos_y = y_pos.unsqueeze(1) / dim_t.unsqueeze(0)
        
        pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=-1).flatten(-2)
        
        pos_embed = torch.cat([pos_y, pos_x], dim=-1)
        return pos_embed.unsqueeze(0)
