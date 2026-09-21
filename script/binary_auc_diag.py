"""
Binary-AUC Diagnostic: 评估 DINOv3 backbone 特征对每个类别的前景/背景分离质量。

原理 (CVPR 2026, arXiv:2511.18380):
  对每个类别 c，计算 backbone 特征激活图与该类 GT mask 之间的 AUC。
  - AUC ≈ 1.0: 特征能清晰分离该类前景/背景
  - AUC ≈ 0.5: 特征无法区分该类（随机水平）
  - AUC < 0.5: 特征对该类有反向响应（异常）

用途:
  - 死类 (crater/tower) 的 AUC 低 → 特征层面分不开，混合 backbone 可能有效
  - 死类的 AUC 不低 → 特征没问题，是分类器/数据的问题

两种激活图:
  1. CLS-attention: CLS token 与空间 token 的余弦相似度（DINO 风格，backbone 原始质量）
  2. Target-logits: 模型 target head 的预测置信度（端到端质量）

用法:
  python script/binary_auc_diag.py --ckpt checkpoints/v7_0617_freq_branch/best_model.pth
  python script/binary_auc_diag.py --ckpt checkpoints/v7_0617_no_tower/best_model.pth --split val
  python script/binary_auc_diag.py --backbone_only  # 只测原始 backbone，不需要 checkpoint
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='Binary-AUC Diagnostic')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Model checkpoint path. If None, only backbone AUC is computed.')
    parser.add_argument('--data_dir', type=str, default='D:/CD/0617final_256')
    parser.add_argument('--config', type=str, default='datasets/configs/0617final.yaml')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_batches', type=int, default=50,
                        help='Max number of batches to process (for speed)')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='auto',
                        help='cuda / cpu / auto')
    parser.add_argument('--backbone_only', action='store_true',
                        help='Only test backbone features, skip model head')
    parser.add_argument('--shuffle', action='store_true', default=True,
                        help='Shuffle data for better class coverage')
    parser.add_argument('--output', type=str, default=None,
                        help='Output text file path')
    return parser.parse_args()


def compute_binary_auc(activation, gt_mask):
    """
    Compute AUC between a continuous activation map and a binary mask.

    Args:
        activation: numpy array, any shape, continuous values
        gt_mask: numpy array, same shape, binary (0/1)
    Returns:
        auc: float, area under ROC curve (normalized to [0.5, 1.0])
    """
    activation = activation.flatten().astype(np.float64)
    gt_mask = gt_mask.flatten().astype(np.float64)

    # Need both positive and negative samples
    n_pos = gt_mask.sum()
    n_neg = len(gt_mask) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None  # Cannot compute AUC

    # Sort by activation (descending)
    order = np.argsort(-activation)
    sorted_gt = gt_mask[order]

    # Compute TPR/FPR at each threshold
    tp = np.cumsum(sorted_gt)
    fp = np.cumsum(1 - sorted_gt)
    tpr = tp / n_pos
    fpr = fp / n_neg

    # AUC via trapezoidal rule
    try:
        auc = np.trapezoid(tpr, fpr)
    except AttributeError:
        auc = np.trapz(tpr, fpr)  # older NumPy

    # Normalize: AUC could be < 0.5 if activation is anti-correlated
    return max(auc, 1.0 - auc)


def cls_attention_map(backbone, img):
    """
    Compute CLS-token attention activation map from DINOv3 backbone.
    Similar to DINO's attention visualization.

    Args:
        backbone: DINOv3Backbone
        img: [B, 3, H, W]
    Returns:
        activation: [B, H_p, W_p] — CLS-to-patch cosine similarity
    """
    # Patch embedding
    x = backbone.patch_embed(img)  # [B, H', W', D]
    H_p, W_p = x.shape[1], x.shape[2]

    # Position embedding
    x, rot_pos_embed = backbone._pos_embed(x)
    x = backbone.norm_pre(x)

    # Forward through all blocks
    for i, blk in enumerate(backbone.blocks):
        if getattr(backbone.backbone, 'rope_mixed', False) and rot_pos_embed is not None:
            x = blk(x, rope=rot_pos_embed[i])
        else:
            x = blk(x, rope=rot_pos_embed)

    x = backbone.norm(x)

    # Extract CLS token and spatial tokens
    prefix = backbone.num_prefix_tokens
    cls_token = x[:, 0, :]  # [B, D]
    spatial = x[:, prefix:, :]  # [B, H_p*W_p, D]

    # Cosine similarity
    cls_norm = F.normalize(cls_token, dim=-1)  # [B, D]
    spatial_norm = F.normalize(spatial, dim=-1)  # [B, N, D]
    sim = torch.bmm(spatial_norm, cls_norm.unsqueeze(-1)).squeeze(-1)  # [B, N]

    # Reshape to 2D
    activation = sim.reshape(-1, H_p, W_p)  # [B, H_p, W_p]
    return activation


def feature_map_activation(backbone, img, level=-1):
    """
    Compute activation map from intermediate backbone features.
    Uses mean-pooled feature magnitude as activation.

    Args:
        backbone: DINOv3Backbone
        img: [B, 3, H, W]
        level: which extract level to use (-1 = last)
    Returns:
        activation: [B, H_p, W_p]
    """
    # Use the backbone's forward to get multi-scale features
    features = backbone(img)  # [C2, C3, C4, C5]
    feat = features[level]  # [B, C, H, W]

    # Use L2 norm of feature channels as activation
    activation = feat.norm(dim=1)  # [B, H, W]
    return activation


def run_diagnostic(args):
    # Device selection
    if args.device == 'auto':
        if torch.cuda.is_available():
            # Check GPU memory to avoid interfering with training
            try:
                free_mem = torch.cuda.mem_get_info(0)[0] / 1e9
                if free_mem > 3.0:
                    device = torch.device('cuda')
                    print(f"Using CUDA (free: {free_mem:.1f} GB)")
                else:
                    device = torch.device('cpu')
                    print(f"GPU memory low ({free_mem:.1f} GB free), using CPU")
            except Exception:
                device = torch.device('cpu')
                print("Cannot check GPU memory, using CPU")
        else:
            device = torch.device('cpu')
            print("CUDA not available, using CPU")
    else:
        device = torch.device(args.device)
        print(f"Using device: {device}")

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)

    target_names = dataset_config.get('target_names', [])
    num_targets = dataset_config.get('num_pixel_targets', 15)

    print(f"\nDataset: {dataset_config.get('dataset', '?')}")
    print(f"Classes: {num_targets}")
    print(f"Class names: {target_names}")
    print(f"Split: {args.split}")
    print(f"Max batches: {args.max_batches}")

    # Build backbone
    model_config = dict(dataset_config)
    model_config['backbone'] = {
        'type': 'dinov3',
        'variant': 'dinov3_base',
        'pretrained': True,
        'pretrained_path': 'weights/model.safetensors',
    }
    model_config['hidden_dim'] = 128
    model_config['embed_dim'] = 64
    model_config['num_pixel_targets'] = num_targets

    from models.backbone.dinov3 import DINOv3Backbone
    backbone = DINOv3Backbone(
        variant='dinov3_base',
        pretrained=True,
        out_channels=[64, 128, 256, 512],
        freeze_stages=2,
        pretrained_path='weights/model.safetensors',
    ).to(device)
    backbone.eval()

    # Optionally load full model
    model = None
    if args.ckpt and not args.backbone_only:
        print(f"\nLoading model from {args.ckpt}...")
        from models import build_hicd_v7
        model = build_hicd_v7(model_config).to(device)
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"  Loaded (epoch={ckpt.get('epoch', '?')})")

    # Load data
    from datasets import create_dataloader
    print(f"\nLoading {args.split} data from {args.data_dir}...")
    data_loader = create_dataloader(
        data_dir=args.data_dir,
        config=dataset_config,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=0,  # Avoid multiprocessing issues
        img_size=args.img_size,
        pin_memory=False,
    )
    # Shuffle for better class coverage
    if args.shuffle:
        data_loader = torch.utils.data.DataLoader(
            data_loader.dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
    print(f"  {len(data_loader)} batches available (shuffle={args.shuffle})")

    # Storage: per-class AUC lists
    # Method 1: CLS-attention AUC
    cls_auc_per_class = defaultdict(list)
    # Method 2: Feature-magnitude AUC
    feat_auc_per_class = defaultdict(list)
    # Method 3: Model target-logits AUC
    logits_auc_per_class = defaultdict(list)

    # Pixel counts per class (for reference)
    class_pixel_counts = defaultdict(int)
    total_pixels = 0

    print(f"\nRunning diagnostic ({args.max_batches} batches)...")
    print("-" * 70)

    with torch.no_grad():
        for bi, batch in enumerate(data_loader):
            if bi >= args.max_batches:
                break

            img_t1 = batch['img_t1'].to(device)
            img_t2 = batch['img_t2'].to(device)
            B = img_t1.shape[0]

            # Get label (target GT)
            if 'label' not in batch:
                continue
            label = batch['label']  # [B, H, W] at original resolution
            if label.dim() == 4:
                label = label.squeeze(1)

            # Use H/4 = 64x64 resolution for AUC (finer than patch grid 16x16)
            auc_res = args.img_size // 4  # 64
            label_s = label.unsqueeze(1).float()
            label_s = F.interpolate(label_s, size=(auc_res, auc_res),
                                     mode='nearest').squeeze(1).long()
            label_s = label_s.to(device)

            # === Method 1: CLS-attention AUC ===
            act_cls = cls_attention_map(backbone, img_t2)  # [B, 16, 16]
            # Upsample to AUC resolution for finer spatial matching
            act_cls = F.interpolate(act_cls.unsqueeze(1),
                                     size=(auc_res, auc_res),
                                     mode='bilinear', align_corners=True).squeeze(1)

            # === Method 2: Feature-magnitude AUC ===
            feat_act = feature_map_activation(backbone, img_t2, level=-1)  # [B, H, W]
            feat_act = F.interpolate(feat_act.unsqueeze(1),
                                      size=(auc_res, auc_res),
                                      mode='bilinear', align_corners=True).squeeze(1)

            # === Method 3: Model target-logits AUC ===
            act_logits = None
            if model is not None:
                targets = {}
                label_orig = batch['label'].to(device)
                if label_orig.dim() == 3:
                    label_ds = label_orig.unsqueeze(1).float()
                    label_ds = F.interpolate(label_ds, scale_factor=0.25, mode='nearest').squeeze(1).long()
                    targets['target_gt'] = label_ds
                if 'change_state' in batch:
                    cs = batch['change_state'].to(device)
                    if cs.dim() == 3:
                        cs_s = cs.unsqueeze(1).float()
                        cs_s = F.interpolate(cs_s, scale_factor=0.25, mode='nearest').squeeze(1).long()
                        targets['change_state'] = cs_s
                        targets['state_gt'] = cs_s

                predictions, _, _ = model(img_t1, img_t2, targets)
                if 'target_logits' in predictions:
                    # Use softmax probability as activation
                    probs = F.softmax(predictions['target_logits'], dim=1)  # [B, C, H/4, W/4]
                    # Already at H/4 = 64x64, matches auc_res
                    act_logits = probs  # [B, C, 64, 64]

            # Compute per-class AUC
            for c in range(num_targets):
                gt_mask = (label_s == c)  # [B, 16, 16]
                n_pixels = gt_mask.sum().item()
                class_pixel_counts[c] += n_pixels
                total_pixels += gt_mask.numel()

                if n_pixels == 0:
                    continue

                # CLS-attention AUC
                for b in range(B):
                    auc = compute_binary_auc(
                        act_cls[b].cpu().numpy(),
                        gt_mask[b].cpu().numpy()
                    )
                    if auc is not None:
                        cls_auc_per_class[c].append(auc)

                # Feature-magnitude AUC
                for b in range(B):
                    auc = compute_binary_auc(
                        feat_act[b].cpu().numpy(),
                        gt_mask[b].cpu().numpy()
                    )
                    if auc is not None:
                        feat_auc_per_class[c].append(auc)

                # Model logits AUC
                if act_logits is not None:
                    for b in range(B):
                        auc = compute_binary_auc(
                            act_logits[b, c].cpu().numpy(),
                            gt_mask[b].cpu().numpy()
                        )
                        if auc is not None:
                            logits_auc_per_class[c].append(auc)

            if (bi + 1) % 10 == 0:
                print(f"  Batch {bi + 1}/{args.max_batches} done")

    # === Results ===
    print("\n" + "=" * 80)
    print("Binary-AUC Diagnostic Results")
    print("=" * 80)
    print(f"Checkpoint: {args.ckpt or 'None (backbone only)'}")
    print(f"Split: {args.split}, Batches: {min(args.max_batches, bi + 1)}")
    print(f"Total pixels analyzed: {total_pixels:,}")
    print()

    # Header
    header = f"{'Class':<12} {'ID':>3} {'Pixels':>10} {'Pixel%':>7} {'CLS-AUC':>8} {'Feat-AUC':>9} {'Logit-AUC':>10} {'Assessment'}"
    print(header)
    print("-" * len(header))

    results = []
    for c in range(num_targets):
        name = target_names[c] if c < len(target_names) else f"class_{c}"
        px = class_pixel_counts[c]
        px_pct = 100.0 * px / max(total_pixels, 1)

        cls_mean = np.mean(cls_auc_per_class[c]) if cls_auc_per_class[c] else None
        feat_mean = np.mean(feat_auc_per_class[c]) if feat_auc_per_class[c] else None
        logit_mean = np.mean(logits_auc_per_class[c]) if logits_auc_per_class[c] else None

        # Assessment
        if cls_mean is None:
            assessment = "NO DATA"
        elif cls_mean >= 0.85:
            assessment = "EXCELLENT"
        elif cls_mean >= 0.75:
            assessment = "GOOD"
        elif cls_mean >= 0.65:
            assessment = "MODERATE"
        elif cls_mean >= 0.55:
            assessment = "WEAK"
        else:
            assessment = "POOR (dead class?)"

        cls_str = f"{cls_mean:.3f}" if cls_mean is not None else "  N/A"
        feat_str = f"{feat_mean:.3f}" if feat_mean is not None else "  N/A"
        logit_str = f"{logit_mean:.3f}" if logit_mean is not None else "  N/A"

        print(f"{name:<12} {c:>3} {px:>10,} {px_pct:>6.2f}% {cls_str:>8} {feat_str:>9} {logit_str:>10} {assessment}")

        results.append({
            'class_id': c,
            'name': name,
            'pixels': px,
            'pixel_pct': px_pct,
            'cls_auc': cls_mean,
            'feat_auc': feat_mean,
            'logit_auc': logit_mean,
            'assessment': assessment,
        })

    # Summary
    print("\n" + "=" * 80)
    print("Interpretation Guide:")
    print("  CLS-AUC  ≥ 0.75: backbone 特征能清晰分离该类 → 混合 backbone 帮助有限")
    print("  CLS-AUC  < 0.65: backbone 特征分离度不足 → 混合 backbone 可能有效")
    print("  CLS-AUC  < 0.55: 特征基本无法区分 → 强烈建议尝试混合 backbone 或数据增强")
    print("  Logit-AUC >> CLS-AUC: 分类头补偿了特征不足（端到端有效但特征弱）")
    print("  Logit-AUC ≈ CLS-AUC: 瓶颈在特征层")
    print("  Logit-AUC << CLS-AUC: 分类头反而拖累了（罕见，检查训练）")
    print("=" * 80)

    # Key findings
    poor_classes = [r for r in results if r['cls_auc'] is not None and r['cls_auc'] < 0.65]
    if poor_classes:
        print(f"\n⚠ Classes with weak backbone separation (CLS-AUC < 0.65):")
        for r in poor_classes:
            gap = ""
            if r['logit_auc'] is not None and r['cls_auc'] is not None:
                diff = r['logit_auc'] - r['cls_auc']
                if diff > 0.1:
                    gap = f" (head compensates: +{diff:.3f})"
                elif diff < -0.1:
                    gap = f" (head hurts: {diff:.3f})"
            print(f"  - {r['name']}: CLS-AUC={r['cls_auc']:.3f}{gap}")

    # Save to file
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write("# Binary-AUC Diagnostic Report\n\n")
            f.write(f"- Checkpoint: {args.ckpt or 'None (backbone only)'}\n")
            f.write(f"- Split: {args.split}\n")
            f.write(f"- Batches: {min(args.max_batches, bi + 1)}\n")
            f.write(f"- Total pixels: {total_pixels:,}\n\n")
            f.write("| Class | ID | Pixels | Pixel% | CLS-AUC | Feat-AUC | Logit-AUC | Assessment |\n")
            f.write("|-------|----|--------|--------|---------|----------|-----------|------------|\n")
            for r in results:
                cls_s = f"{r['cls_auc']:.3f}" if r['cls_auc'] else "N/A"
                feat_s = f"{r['feat_auc']:.3f}" if r['feat_auc'] else "N/A"
                logit_s = f"{r['logit_auc']:.3f}" if r['logit_auc'] else "N/A"
                f.write(f"| {r['name']} | {r['class_id']} | {r['pixels']:,} | {r['pixel_pct']:.2f}% | "
                        f"{cls_s} | {feat_s} | {logit_s} | {r['assessment']} |\n")
        print(f"\nReport saved to: {args.output}")

    return results


if __name__ == '__main__':
    args = parse_args()
    run_diagnostic(args)
