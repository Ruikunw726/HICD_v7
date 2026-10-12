"""
DINOv3 Backbone Wrapper for HICD v7
Replaces LWGANet with DINOv3 pretrained ViT backbone.
Outputs multi-scale CNN-compatible features for NFA integration.

Key design:
- DINOv3 uses RoPE (rotary position embedding), not absolute pos_embed
- Has 5 prefix tokens (1 CLS + 4 register) before spatial tokens
- Extract features from intermediate blocks and project to CNN pyramid
- Stage freezing: freeze early blocks for domain transfer (你的域差距分析的核心发现)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DINOv3Backbone(nn.Module):
    """
    DINOv3 ViT backbone producing multi-scale CNN-compatible features.
    
    Output format matches LWGANet L1:
        C2: [B, 64, H/4, W/4]   (stride 4)
        C3: [B, 128, H/8, W/8]  (stride 8)
        C4: [B, 256, H/16, W/16] (stride 16)
        C5: [B, 512, H/32, W/32] (stride 32)
    """
    
    VARIANT_CONFIGS = {
        'dinov3_small': {
            'model_name': 'vit_small_patch16_dinov3',
            'embed_dim': 384,
            'num_blocks': 12,
            'extract_blocks': [2, 5, 8, 11],
        },
        'dinov3_base': {
            'model_name': 'vit_base_patch16_dinov3',
            'embed_dim': 768,
            'num_blocks': 12,
            'extract_blocks': [2, 5, 8, 11],
        },
        'dinov3_large': {
            'model_name': 'vit_large_patch16_dinov3',
            'embed_dim': 1024,
            'num_blocks': 24,
            'extract_blocks': [5, 11, 17, 23],
        },
    }
    
    def __init__(self, variant='dinov3_base', pretrained=True, 
                 out_channels=None, freeze_stages=2, pretrained_path=None):
        super().__init__()
        
        if variant not in self.VARIANT_CONFIGS:
            raise ValueError(f"Unknown variant: {variant}")
        
        cfg = self.VARIANT_CONFIGS[variant]
        self.variant = variant
        self.embed_dim = cfg['embed_dim']
        self.extract_blocks = cfg['extract_blocks']
        self.out_channels = out_channels or [64, 128, 256, 512]
        self.freeze_stages = freeze_stages
        self.patch_size = 16
        
        # Create DINOv3 model
        self.backbone = timm.create_model(
            cfg['model_name'],
            pretrained=False,
            num_classes=0,
            global_pool='',
        )

        # Load weights: local path first, then fall back to pretrained download
        if pretrained_path is not None and os.path.exists(pretrained_path):
            print(f'Loading DINOv3 weights from local: {pretrained_path}')
            from safetensors.torch import load_file
            state_dict = load_file(pretrained_path)
            filtered = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
            msg = self.backbone.load_state_dict(filtered, strict=False)
            print(f'  Loaded {len(filtered)} keys, missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}')
        elif pretrained:
            print('No local weights found, downloading from HuggingFace...')
            self.backbone = timm.create_model(
                cfg['model_name'],
                pretrained=True,
                num_classes=0,
                global_pool='',
            )
        
        # Cache key attributes
        self.patch_embed = self.backbone.patch_embed
        self.blocks = self.backbone.blocks
        self.norm_pre = self.backbone.norm_pre
        self.norm = self.backbone.norm
        self.cls_token = self.backbone.cls_token
        self.reg_token = self.backbone.reg_token
        self.rope = self.backbone.rope
        self.num_prefix_tokens = self.backbone.num_prefix_tokens
        
        # Projection heads: [B, N, D] -> [B, C_out, H, W]
        self.proj_heads = nn.ModuleList()
        for out_ch in self.out_channels:
            self.proj_heads.append(nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, out_ch),
            ))
        
        self._freeze_stages()
        self._print_info()
    
    def _freeze_stages(self):
        """Freeze early transformer blocks for domain transfer."""
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        
        num_frozen = min(self.freeze_stages * 3, len(self.blocks))
        for i, block in enumerate(self.blocks):
            if i < num_frozen:
                for param in block.parameters():
                    param.requires_grad = False
        
        frozen_count = sum(1 for p in self.parameters() if not p.requires_grad)
        total_count = sum(1 for p in self.parameters())
        print(f"DINOv3: froze {num_frozen}/{len(self.blocks)} blocks, "
              f"{frozen_count}/{total_count} param groups frozen")
    
    def _print_info(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"DINOv3Backbone ({self.variant}):")
        print(f"  embed_dim={self.embed_dim}, blocks={len(self.blocks)}, "
              f"extract_at={self.extract_blocks}")
        print(f"  out_channels={self.out_channels}")
        print(f"  trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M")
    
    def forward(self, x):
        """
        Forward pass with proper RoPE handling.
        
        Args:
            x: [B, 3, H, W]
        Returns:
            [C2, C3, C4, C5] matching LWGANet L1 format
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, H', W', D]
        H_p, W_p = x.shape[1], x.shape[2]
        
        # Position embedding (handles CLS + register tokens + RoPE)
        x, rot_pos_embed = self._pos_embed(x)
        
        # Pre-norm
        x = self.norm_pre(x)
        
        # Forward through blocks, extracting at specified indices
        extracted = {}
        for i, blk in enumerate(self.blocks):
            if i in self.extract_blocks:
                extracted[i] = x
            # Pass RoPE to block
            if getattr(self.backbone, 'rope_mixed', False) and rot_pos_embed is not None:
                x = blk(x, rope=rot_pos_embed[i])
            else:
                x = blk(x, rope=rot_pos_embed)
        
        # Final norm
        x = self.norm(x)
        
        # Extract spatial tokens (skip prefix: CLS + register)
        prefix = self.num_prefix_tokens
        spatial_tokens = x[:, prefix:, :]  # [B, H_p*W_p, D]
        
        # Project and reshape each extracted level
        multi_scale = []
        for level, block_idx in enumerate(self.extract_blocks):
            feat = extracted[block_idx]
            # Skip prefix tokens
            feat_spatial = feat[:, prefix:, :]  # [B, H_p*W_p, D]
            
            # Project: [B, N, D] -> [B, N, C_out]
            feat_proj = self.proj_heads[level](feat_spatial)
            
            # Reshape to 2D: [B, N, C_out] -> [B, C_out, H_p, W_p]
            feat_2d = feat_proj.transpose(1, 2).reshape(
                B, self.out_channels[level], H_p, W_p
            )
            multi_scale.append(feat_2d)
        
        # C2 (stride 4): upsample from patch resolution (stride 16) to stride 4
        c2 = F.interpolate(multi_scale[0], scale_factor=4, mode='bilinear', align_corners=True)
        
        # C3 (stride 8): upsample from patch resolution to stride 8
        c3 = F.interpolate(multi_scale[1], scale_factor=2, mode='bilinear', align_corners=True)
        
        # C4 (stride 16): native patch resolution
        c4 = multi_scale[2]
        
        # C5 (stride 32): downsample from patch resolution to stride 32
        c5 = F.max_pool2d(multi_scale[3], kernel_size=2, stride=2)
        
        return [c2, c3, c4, c5]
    
    def _pos_embed(self, x):
        """
        Handle position embedding including CLS token, register tokens, and RoPE.
        Replicates timm's _pos_embed logic.
        """
        B, H, W, C = x.shape
        
        # Compute RoPE embeddings
        rot_pos_embed = self.rope.get_embed(shape=(H, W)) if self.rope is not None else None
        
        # Absolute position embedding (DINOv3 doesn't use this, but handle for completeness)
        pos_embed = None
        if hasattr(self.backbone, 'pos_embed') and self.backbone.pos_embed is not None:
            pos_embed = self.backbone.pos_embed
        
        # Concatenate prefix tokens
        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(B, -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(B, -1, -1))
        
        # Flatten spatial dims
        x = x.view(B, -1, C)
        
        # Add position embeddings
        if self.backbone.no_embed_class:
            if pos_embed is not None:
                x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            if pos_embed is not None:
                x = x + pos_embed
        
        
        return x, rot_pos_embed
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        print("DINOv3 backbone fully frozen")
    
    def unfreeze(self, ratio=1.0):
        if not 0.0 <= ratio <= 1.0:
            raise ValueError('unfreeze ratio must be between 0 and 1')
        all_params = list(self.parameters())
        num_to_unfreeze = int(len(all_params) * ratio)
        selected = all_params[-num_to_unfreeze:] if num_to_unfreeze else []
        for param in selected:
            param.requires_grad = True
        print(f"Unfroze {num_to_unfreeze}/{len(all_params)} parameter groups")
    
    def get_out_channels(self):
        """Return 5 values: [C1, C2, C3, C4, C5] for NFA compatibility."""
        return [self.out_channels[0]] + self.out_channels
    
    def train(self, mode=True):
        super().train(mode)
        num_frozen = min(self.freeze_stages * 3, len(self.blocks))
        for i, block in enumerate(self.blocks):
            if i < num_frozen:
                block.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        return self


def build_dinov3_backbone(config):
    backbone_cfg = config.get('backbone', {})
    return DINOv3Backbone(
        variant=backbone_cfg.get('variant', 'dinov3_base'),
        pretrained=backbone_cfg.get('pretrained', True),
        out_channels=backbone_cfg.get('out_channels', [64, 128, 256, 512]),
        freeze_stages=backbone_cfg.get('freeze_stages', 2),
        pretrained_path=backbone_cfg.get('pretrained_path', None),
    )


