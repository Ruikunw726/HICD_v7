"""
State Transferability Analysis v2: 用像素级特征做 t-SNE 聚类。

改进:
  1. 存储原始像素特征（不只是均值），t-SNE 用几千个点而不是 22 个
  2. 按 (target, state) 着色，观察聚类结构
  3. 两组图：按目标着色 vs 按状态着色
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='checkpoints/v7_0617_freq_branch/best_model.pth')
    parser.add_argument('--data_dir', type=str, default='D:/CD/0617final_256')
    parser.add_argument('--config', type=str, default='datasets/configs/0617final.yaml')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_batches', type=int, default=300)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--min_pixels', type=int, default=20)
    parser.add_argument('--max_pixels_per_combo', type=int, default=300,
                        help='Max pixels to keep per combo for t-SNE')
    parser.add_argument('--output_dir', type=str, default='F:/mambacd/working_notes/state_transfer_v2')
    return parser.parse_args()


def extract_fused_features(model, img_t1, img_t2):
    with torch.no_grad():
        feat_t1_list = model.backbone(img_t1)
        feat_t2_list = model.backbone(img_t2)
        nfa_t1 = model.nfa_t1(*feat_t1_list)
        nfa_t2 = model.nfa_t2(*feat_t2_list)
        fused_features, _, _ = model.tfm(nfa_t1, nfa_t2)
        if model.use_freq_branch:
            p2_size = fused_features[0].shape[2:]
            freq_features = model.freq_branch(img_t1, img_t2, target_size=p2_size)
            enhanced_p2 = model.freq_fusion(torch.cat([fused_features[0], freq_features], dim=1))
            fused_features = [enhanced_p2] + fused_features[1:]
        feat = fused_features[0]  # [B, C, H/4, W/4]
    return feat


def run_analysis(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    with open(args.config, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)

    target_names = dataset_config.get('target_names', [])
    state_names = dataset_config.get('state_names', [])
    num_targets = dataset_config.get('num_pixel_targets', 15)
    num_states = dataset_config.get('num_states', 7)

    from models.modules.label_structure import VALID_COMBINATIONS
    valid_combos = set(VALID_COMBINATIONS)

    # Build model
    model_config = dict(dataset_config)
    model_config['backbone'] = {
        'type': 'dinov3', 'variant': 'dinov3_base',
        'pretrained': True, 'pretrained_path': 'weights/model.safetensors',
    }
    model_config['hidden_dim'] = 128
    model_config['embed_dim'] = 64

    from models import build_hicd_v7
    model = build_hicd_v7(model_config).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    model.eval()
    print(f"Loaded: {args.ckpt}")

    from datasets import create_dataloader
    data_loader = create_dataloader(
        data_dir=args.data_dir, config=dataset_config,
        split=args.split, batch_size=args.batch_size,
        num_workers=0, img_size=args.img_size, pin_memory=False,
    )
    data_loader = torch.utils.data.DataLoader(
        data_loader.dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=False,
    )

    # Storage for pixel-level features
    # Each entry: (feature_vector [C], target_id, state_id)
    all_features = []
    all_targets = []
    all_states = []
    pixel_counts = defaultdict(int)

    feat_dim = None
    print(f"Extracting features from {args.max_batches} batches...")
    print(f"  (shuffled, so rare classes will appear)")

    with torch.no_grad():
        for bi, batch in enumerate(data_loader):
            if bi >= args.max_batches:
                break

            img_t1 = batch['img_t1'].to(device)
            img_t2 = batch['img_t2'].to(device)

            if 'label' not in batch or 'change_state' not in batch:
                continue

            target_gt = batch['label']
            state_gt = batch['change_state']
            if target_gt.dim() == 4:
                target_gt = target_gt.squeeze(1)
            if state_gt.dim() == 4:
                state_gt = state_gt.squeeze(1)

            feat = extract_fused_features(model, img_t1, img_t2)
            B, C, Hf, Wf = feat.shape
            feat_dim = C

            target_gt_s = F.interpolate(
                target_gt.unsqueeze(1).float(), size=(Hf, Wf), mode='nearest'
            ).squeeze(1).long()
            state_gt_s = F.interpolate(
                state_gt.unsqueeze(1).float(), size=(Hf, Wf), mode='nearest'
            ).squeeze(1).long()

            feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, C).cpu()
            tgt_flat = target_gt_s.reshape(-1)
            st_flat = state_gt_s.reshape(-1)

            for t in range(num_targets):
                for s in range(num_states):
                    if (t, s) not in valid_combos:
                        continue
                    mask = (tgt_flat == t) & (st_flat == s)
                    n = mask.sum().item()
                    if n == 0:
                        continue
                    pixel_counts[(t, s)] += n

                    # Keep up to max_pixels_per_combo
                    remaining = args.max_pixels_per_combo - len([1 for ti, si in zip(all_targets, all_states) if ti == t and si == s])
                    if remaining <= 0:
                        continue
                    feats = feat_flat[mask]
                    if feats.shape[0] > remaining:
                        idx = torch.randperm(feats.shape[0])[:remaining]
                        feats = feats[idx]
                    all_features.append(feats)
                    all_targets.extend([t] * feats.shape[0])
                    all_states.extend([s] * feats.shape[0])

            if (bi + 1) % 50 == 0:
                print(f"  Batch {bi+1}/{args.max_batches}, "
                      f"combos≥{args.min_pixels}px: {sum(1 for k,v in pixel_counts.items() if v >= args.min_pixels)}, "
                      f"total pixels stored: {len(all_targets)}")

    if not all_features:
        print("ERROR: No features extracted")
        return

    all_features = torch.cat(all_features, dim=0).numpy()
    all_targets = np.array(all_targets)
    all_states = np.array(all_states)

    # Filter by min_pixels
    valid_mask = np.zeros(len(all_targets), dtype=bool)
    for t in range(num_targets):
        for s in range(num_states):
            if (t, s) not in valid_combos:
                continue
            if pixel_counts[(t, s)] < args.min_pixels:
                continue
            mask = (all_targets == t) & (all_states == s)
            valid_mask |= mask

    all_features = all_features[valid_mask]
    all_targets = all_targets[valid_mask]
    all_states = all_states[valid_mask]

    print(f"\nTotal pixels for t-SNE: {len(all_targets)}")
    print(f"Feature dim: {feat_dim}")

    # Per-combo stats
    print("\nCombos with ≥{} pixels:".format(args.min_pixels))
    for t in range(num_targets):
        for s in range(num_states):
            if (t, s) not in valid_combos:
                continue
            n = pixel_counts[(t, s)]
            if n >= args.min_pixels:
                stored = np.sum((all_targets == t) & (all_states == s))
                print(f"  {target_names[t]:12s}_{state_names[s]:12s}: {n:>8,} px (stored {stored})")

    # === t-SNE ===
    print("\nRunning t-SNE...")
    from sklearn.manifold import TSNE

    # Normalize features
    feat_norm = all_features / (np.linalg.norm(all_features, axis=1, keepdims=True) + 1e-8)

    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=1000)
    except TypeError:
        tsne = TSNE(n_components=2, random_state=42, perplexity=50, n_iter=1000)
    coords = tsne.fit_transform(feat_norm)
    print(f"  Done. Shape: {coords.shape}")

    # === Figure 1: t-SNE colored by STATE ===
    fig, ax = plt.subplots(1, 1, figsize=(14, 11))
    state_colors = plt.cm.Set1(np.linspace(0, 0.9, num_states))

    for s in range(num_states):
        mask = all_states == s
        if mask.sum() == 0:
            continue
        n = mask.sum()
        ax.scatter(coords[mask, 0], coords[mask, 1],
                  c=[state_colors[s]], label=f"{state_names[s]} (n={n})",
                  s=8, alpha=0.5, edgecolors='none')

    ax.set_title(f't-SNE of Pixel Features — Colored by STATE\n'
                 f'If states cluster separately → transferable; '
                 f'if mixed by target → target-specific',
                 fontsize=12)
    ax.legend(loc='best', fontsize=10, markerscale=3)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    path1 = os.path.join(args.output_dir, 'tsne_by_state.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path1}")

    # === Figure 2: t-SNE colored by TARGET ===
    fig, ax = plt.subplots(1, 1, figsize=(14, 11))
    target_colors = plt.cm.tab20(np.linspace(0, 1, num_targets))

    for t in range(num_targets):
        mask = all_targets == t
        if mask.sum() == 0:
            continue
        n = mask.sum()
        ax.scatter(coords[mask, 0], coords[mask, 1],
                  c=[target_colors[t]], label=f"{target_names[t]} (n={n})",
                  s=8, alpha=0.5, edgecolors='none')

    ax.set_title(f't-SNE of Pixel Features — Colored by TARGET\n'
                 f'Clear target clusters = target dominates feature space',
                 fontsize=12)
    ax.legend(loc='best', fontsize=9, markerscale=3, ncol=2)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    path2 = os.path.join(args.output_dir, 'tsne_by_target.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path2}")

    # === Figure 3: t-SNE colored by (target, state) for key combos ===
    # Focus on combos that appear in multiple targets
    state_target_counts = defaultdict(set)
    for t, s in zip(all_targets, all_states):
        state_target_counts[s].add(t)

    # Pick states that appear in ≥2 targets
    multi_target_states = [s for s, ts in state_target_counts.items() if len(ts) >= 2]
    print(f"\nStates with ≥2 targets: {[state_names[s] for s in multi_target_states]}")

    fig, axes = plt.subplots(1, len(multi_target_states), figsize=(6 * len(multi_target_states), 5))
    if len(multi_target_states) == 1:
        axes = [axes]

    for ax_i, s in enumerate(multi_target_states):
        ax = axes[ax_i]
        targets_for_state = sorted(state_target_counts[s])
        colors = plt.cm.Set2(np.linspace(0, 0.9, len(targets_for_state)))

        # Plot background (other states, light gray)
        bg_mask = all_states != s
        if bg_mask.sum() > 0:
            # Subsample background for speed
            bg_idx = np.where(bg_mask)[0]
            if len(bg_idx) > 2000:
                bg_idx = np.random.choice(bg_idx, 2000, replace=False)
            ax.scatter(coords[bg_idx, 0], coords[bg_idx, 1],
                      c='lightgray', s=3, alpha=0.2, edgecolors='none')

        # Plot this state's targets
        for ci, t in enumerate(targets_for_state):
            mask = (all_states == s) & (all_targets == t)
            n = mask.sum()
            if n == 0:
                continue
            ax.scatter(coords[mask, 0], coords[mask, 1],
                      c=[colors[ci]], label=f"{target_names[t]} ({n})",
                      s=15, alpha=0.7, edgecolors='black', linewidth=0.3)

        ax.set_title(f'State: {state_names[s]}\n(should cluster together if transferable)',
                     fontsize=11)
        ax.legend(loc='best', fontsize=8, markerscale=2)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    path3 = os.path.join(args.output_dir, 'tsne_per_state.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path3}")

    # === Quantitative: silhouette score ===
    print("\n" + "=" * 60)
    print("Silhouette Scores (higher = better clustering)")
    print("=" * 60)

    from sklearn.metrics import silhouette_score

    # Subsample for speed
    n_sample = min(5000, len(all_targets))
    idx = np.random.choice(len(all_targets), n_sample, replace=False)
    feat_sample = feat_norm[idx]
    state_sample = all_states[idx]
    target_sample = all_targets[idx]

    # By state
    unique_states = np.unique(state_sample)
    if len(unique_states) >= 2:
        score_state = silhouette_score(feat_sample, state_sample)
        print(f"  By STATE:  {score_state:.4f}")

    # By target
    unique_targets = np.unique(target_sample)
    if len(unique_targets) >= 2:
        score_target = silhouette_score(feat_sample, target_sample)
        print(f"  By TARGET: {score_target:.4f}")

    # By (target, state) combo
    combo_labels = target_sample * 100 + state_sample
    unique_combos = np.unique(combo_labels)
    if len(unique_combos) >= 2:
        score_combo = silhouette_score(feat_sample, combo_labels)
        print(f"  By COMBO:  {score_combo:.4f}")

    print(f"\n  Interpretation:")
    print(f"    If TARGET >> STATE: features encode target identity more than state")
    print(f"    If STATE >> TARGET: features encode state more than target")
    print(f"    If COMBO is highest: features encode joint (target, state)")

    # Save summary
    summary_path = os.path.join(args.output_dir, 'summary.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# State Transferability Analysis v2\n\n")
        f.write(f"- Checkpoint: {args.ckpt}\n")
        f.write(f"- Split: {args.split}, Batches: {args.max_batches}\n")
        f.write(f"- Total pixels: {len(all_targets)}\n")
        f.write(f"- Feature dim: {feat_dim}\n\n")

        f.write("## Silhouette Scores\n\n")
        f.write("| Grouping | Score | Interpretation |\n")
        f.write("|----------|-------|----------------|\n")
        f.write(f"| By State | {score_state:.4f} | Higher = states form clusters |\n")
        f.write(f"| By Target | {score_target:.4f} | Higher = targets form clusters |\n")
        f.write(f"| By Combo | {score_combo:.4f} | Higher = joint clusters |\n\n")

        if score_target > score_state:
            f.write("**Conclusion**: Features are more target-dominated than state-dominated. ")
            f.write("State semantics are entangled with target identity. ")
            f.write("Cross-target state transfer is unlikely to work without explicit disentanglement.\n\n")
        elif score_state > score_target:
            f.write("**Conclusion**: Features are more state-dominated. ")
            f.write("Cross-target state transfer may be viable.\n\n")
        else:
            f.write("**Conclusion**: Features encode both target and state similarly.\n\n")

        f.write("## Per-Combo Pixel Counts\n\n")
        f.write("| Combo | Pixels | Stored |\n")
        f.write("|-------|--------|--------|\n")
        for t in range(num_targets):
            for s in range(num_states):
                if (t, s) not in valid_combos:
                    continue
                n = pixel_counts[(t, s)]
                if n >= args.min_pixels:
                    stored = int(np.sum((all_targets == t) & (all_states == s)))
                    f.write(f"| {target_names[t]}_{state_names[s]} | {n:,} | {stored} |\n")

    print(f"\nSummary: {summary_path}")
    print(f"All outputs: {args.output_dir}")


if __name__ == '__main__':
    args = parse_args()
    run_analysis(args)
