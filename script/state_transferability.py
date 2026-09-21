"""
State Transferability Analysis: 验证状态语义是否跨目标可迁移。

核心问题: 「损毁」在 building 上和在 bridge 上的特征表征是否相似？
  - 如果相似 → 稀有目标的状态可以从常见目标迁移学习
  - 如果不相似 → 状态语义是目标特定的，迁移困难

方法:
  1. 用训练好的模型提取 fused_features (语义头之前的特征)
  2. 按 GT (target, state) 组合分组，计算每组的平均特征向量
  3. 计算所有组合之间的余弦相似度矩阵
  4. 可视化: 相似度热力图 + 按状态分组的跨目标对比

用法:
  python script/state_transferability.py --ckpt checkpoints/v7_0617_freq_branch/best_model.pth
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
import matplotlib.colors as mcolors


def parse_args():
    parser = argparse.ArgumentParser(description='State Transferability Analysis')
    parser.add_argument('--ckpt', type=str, default='checkpoints/v7_0617_freq_branch/best_model.pth')
    parser.add_argument('--data_dir', type=str, default='D:/CD/0617final_256')
    parser.add_argument('--config', type=str, default='datasets/configs/0617final.yaml')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_batches', type=int, default=200)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--min_pixels', type=int, default=50,
                        help='Minimum pixels per (target,state) combo to include')
    parser.add_argument('--output_dir', type=str, default='F:/mambacd/working_notes/state_transfer')
    return parser.parse_args()


def extract_fused_features(model, img_t1, img_t2):
    """
    Extract fused features (before semantic head) from the model.
    Returns features at H/4 resolution.
    """
    with torch.no_grad():
        # Backbone
        feat_t1_list = model.backbone(img_t1)
        feat_t2_list = model.backbone(img_t2)

        # NFA + TFM
        nfa_t1 = model.nfa_t1(*feat_t1_list)
        nfa_t2 = model.nfa_t2(*feat_t2_list)
        fused_features, _, _ = model.tfm(nfa_t1, nfa_t2)

        # Frequency branch enhancement (if enabled)
        if model.use_freq_branch:
            p2_size = fused_features[0].shape[2:]
            freq_features = model.freq_branch(img_t1, img_t2, target_size=p2_size)
            enhanced_p2 = model.freq_fusion(torch.cat([fused_features[0], freq_features], dim=1))
            fused_features = [enhanced_p2] + fused_features[1:]

        # Use P2 (highest resolution, H/4) as the feature
        feat = fused_features[0]  # [B, C, H/4, W/4]

    return feat


def run_analysis(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)

    target_names = dataset_config.get('target_names', [])
    state_names = dataset_config.get('state_names', [])
    num_targets = dataset_config.get('num_pixel_targets', 15)
    num_states = dataset_config.get('num_states', 7)

    # Valid combinations from label_structure
    from models.modules.label_structure import VALID_COMBINATIONS
    valid_combos = set(VALID_COMBINATIONS)

    print(f"Targets: {target_names}")
    print(f"States: {state_names}")
    print(f"Valid combinations: {len(valid_combos)}")

    # Build model
    model_config = dict(dataset_config)
    model_config['backbone'] = {
        'type': 'dinov3',
        'variant': 'dinov3_base',
        'pretrained': True,
        'pretrained_path': 'weights/model.safetensors',
    }
    model_config['hidden_dim'] = 128
    model_config['embed_dim'] = 64

    from models import build_hicd_v7
    model = build_hicd_v7(model_config).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}")

    # Load data
    from datasets import create_dataloader
    data_loader = create_dataloader(
        data_dir=args.data_dir, config=dataset_config,
        split=args.split, batch_size=args.batch_size,
        num_workers=0, img_size=args.img_size, pin_memory=False,
    )
    # Shuffle
    data_loader = torch.utils.data.DataLoader(
        data_loader.dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=False,
    )
    print(f"Data: {len(data_loader)} batches, analyzing {args.max_batches}")

    # Storage: features per (target, state) combination
    # Key: (target_id, state_id), Value: list of feature vectors [C]
    feature_store = defaultdict(list)
    pixel_counts = defaultdict(int)

    feat_dim = None

    print("\nExtracting features...")
    with torch.no_grad():
        for bi, batch in enumerate(data_loader):
            if bi >= args.max_batches:
                break

            img_t1 = batch['img_t1'].to(device)
            img_t2 = batch['img_t2'].to(device)

            if 'label' not in batch or 'change_state' not in batch:
                continue

            # Get GT labels at original resolution
            target_gt = batch['label']  # [B, H, W]
            state_gt = batch['change_state']  # [B, H, W]
            if target_gt.dim() == 4:
                target_gt = target_gt.squeeze(1)
            if state_gt.dim() == 4:
                state_gt = state_gt.squeeze(1)

            # Extract fused features at H/4
            feat = extract_fused_features(model, img_t1, img_t2)  # [B, C, H/4, W/4]
            B, C, Hf, Wf = feat.shape
            feat_dim = C

            # Resize GT to feature resolution
            target_gt_s = F.interpolate(
                target_gt.unsqueeze(1).float(), size=(Hf, Wf), mode='nearest'
            ).squeeze(1).long().to(device)
            state_gt_s = F.interpolate(
                state_gt.unsqueeze(1).float(), size=(Hf, Wf), mode='nearest'
            ).squeeze(1).long().to(device)

            # Extract features per (target, state) combination
            feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, C)  # [B*Hf*Wf, C]
            tgt_flat = target_gt_s.reshape(-1)
            st_flat = state_gt_s.reshape(-1)

            for t in range(num_targets):
                for s in range(num_states):
                    if (t, s) not in valid_combos:
                        continue
                    mask = (tgt_flat == t) & (st_flat == s)
                    n = mask.sum().item()
                    if n > 0:
                        pixel_counts[(t, s)] += n
                        # Subsample to avoid memory issues
                        feats = feat_flat[mask]
                        if feats.shape[0] > 200:
                            idx = torch.randperm(feats.shape[0])[:200]
                            feats = feats[idx]
                        feature_store[(t, s)].append(feats.cpu())

            if (bi + 1) % 20 == 0:
                total_combos = sum(1 for k in feature_store if pixel_counts[k] >= args.min_pixels)
                print(f"  Batch {bi+1}/{args.max_batches}, "
                      f"combos with ≥{args.min_pixels}px: {total_combos}")

    # Compute mean feature per combination
    print(f"\nFeature dim: {feat_dim}")
    print(f"Total combinations found: {len(feature_store)}")

    combo_features = {}
    combo_labels = {}
    for key, feat_list in feature_store.items():
        if pixel_counts[key] < args.min_pixels:
            continue
        all_feats = torch.cat(feat_list, dim=0)  # [N, C]
        mean_feat = all_feats.mean(dim=0)  # [C]
        # L2 normalize
        mean_feat = F.normalize(mean_feat, dim=0)
        combo_features[key] = mean_feat
        t, s = key
        combo_labels[key] = f"{target_names[t]}_{state_names[s]}"

    print(f"Combinations with ≥{args.min_pixels} pixels: {len(combo_features)}")

    if len(combo_features) < 2:
        print("ERROR: Not enough combinations for analysis")
        return

    # Compute cosine similarity matrix
    keys = sorted(combo_features.keys())
    labels = [combo_labels[k] for k in keys]
    feat_matrix = torch.stack([combo_features[k] for k in keys])  # [N, C]
    sim_matrix = feat_matrix @ feat_matrix.T  # [N, N] cosine similarity

    sim_np = sim_matrix.numpy()

    # === Analysis 1: Cross-target similarity for same state ===
    print("\n" + "=" * 70)
    print("Cross-Target State Similarity (same state, different targets)")
    print("=" * 70)

    cross_target_results = {}
    for s in range(num_states):
        state_keys = [k for k in keys if k[1] == s]
        if len(state_keys) < 2:
            continue

        # Compute pairwise similarities within this state
        state_indices = [keys.index(k) for k in state_keys]
        state_sim = sim_np[np.ix_(state_indices, state_indices)]

        # Get upper triangle (excluding diagonal)
        n = len(state_keys)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                t1, _ = state_keys[i]
                t2, _ = state_keys[j]
                sim = state_sim[i, j]
                pairs.append((target_names[t1], target_names[t2], sim))

        if pairs:
            sims = [p[2] for p in pairs]
            mean_sim = np.mean(sims)
            std_sim = np.std(sims)
            cross_target_results[state_names[s]] = {
                'mean': mean_sim, 'std': std_sim,
                'n_targets': len(state_keys), 'pairs': pairs
            }
            print(f"\n{state_names[s]} ({len(state_keys)} targets):")
            print(f"  Mean cross-target similarity: {mean_sim:.3f} ± {std_sim:.3f}")
            # Show top-3 most similar pairs
            sorted_pairs = sorted(pairs, key=lambda x: -x[2])
            for p in sorted_pairs[:3]:
                print(f"    {p[0]:12s} ↔ {p[1]:12s}: {p[2]:.3f}")
            if len(sorted_pairs) > 3:
                for p in sorted_pairs[-2:]:
                    print(f"    {p[0]:12s} ↔ {p[1]:12s}: {p[2]:.3f} (least similar)")

    # === Analysis 2: Within-target vs cross-target comparison ===
    print("\n" + "=" * 70)
    print("Within-Target vs Cross-Target State Similarity")
    print("=" * 70)

    for s in range(num_states):
        state_keys = [k for k in keys if k[1] == s]
        if len(state_keys) < 2:
            continue

        # Cross-target: same state, different targets
        cross_sims = []
        for i, ki in enumerate(state_keys):
            for j, kj in enumerate(state_keys):
                if i >= j:
                    continue
                if ki[0] != kj[0]:  # different targets
                    idx_i, idx_j = keys.index(ki), keys.index(kj)
                    cross_sims.append(sim_np[idx_i, idx_j])

        # Within-target: different states, same target (baseline)
        within_sims = []
        for t in range(num_targets):
            target_keys = [k for k in keys if k[0] == t and k[1] != s]
            this_key = (t, s)
            if this_key not in combo_features:
                continue
            idx_this = keys.index(this_key)
            for tk in target_keys:
                idx_tk = keys.index(tk)
                within_sims.append(sim_np[idx_this, idx_tk])

        if cross_sims and within_sims:
            cross_mean = np.mean(cross_sims)
            within_mean = np.mean(within_sims)
            print(f"\n{state_names[s]}:")
            print(f"  Cross-target (same state): {cross_mean:.3f}")
            print(f"  Within-target (diff state): {within_mean:.3f}")
            print(f"  Ratio (cross/within): {cross_mean/within_mean:.3f}")
            if cross_mean > within_mean:
                print(f"  ✓ State semantics transfer across targets!")
            else:
                print(f"  ✗ State is more target-specific than state-specific")

    # === Visualization ===
    print("\nGenerating visualizations...")

    # Figure 1: Full similarity heatmap
    fig, ax = plt.subplots(1, 1, figsize=(max(12, len(keys) * 0.5), max(10, len(keys) * 0.45)))
    im = ax.imshow(sim_np, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f'(Target, State) Feature Cosine Similarity\n({args.ckpt.split("/")[-1]}, {args.split})')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Draw grid lines between states
    for i, k in enumerate(keys):
        if i > 0 and k[1] != keys[i-1][1]:
            ax.axhline(y=i - 0.5, color='gray', linewidth=0.5)
            ax.axvline(x=i - 0.5, color='gray', linewidth=0.5)

    plt.tight_layout()
    path1 = os.path.join(args.output_dir, 'similarity_heatmap.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path1}")

    # Figure 2: Cross-target state similarity bar chart
    if cross_target_results:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        states = list(cross_target_results.keys())
        means = [cross_target_results[s]['mean'] for s in states]
        stds = [cross_target_results[s]['std'] for s in states]
        n_targets = [cross_target_results[s]['n_targets'] for s in states]

        colors = plt.cm.Set2(np.linspace(0, 1, len(states)))
        bars = ax.bar(range(len(states)), means, yerr=stds, capsize=5,
                      color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(states)))
        ax.set_xticklabels([f"{s}\n({n} targets)" for s, n in zip(states, n_targets)],
                           fontsize=9)
        ax.set_ylabel('Mean Cross-Target Cosine Similarity')
        ax.set_title('State Transferability: Cross-Target Feature Similarity\n'
                     '(higher = more transferable)')
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
        ax.legend()

        # Annotate values
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        path2 = os.path.join(args.output_dir, 'cross_target_similarity.png')
        plt.savefig(path2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path2}")

    # Figure 3: Within-target vs cross-target comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x_pos = 0
    bar_width = 0.35
    state_labels = []
    cross_vals = []
    within_vals = []

    for s in range(num_states):
        state_keys = [k for k in keys if k[1] == s]
        if len(state_keys) < 2:
            continue

        cross_sims = []
        for i, ki in enumerate(state_keys):
            for j, kj in enumerate(state_keys):
                if i >= j or ki[0] == kj[0]:
                    continue
                idx_i, idx_j = keys.index(ki), keys.index(kj)
                cross_sims.append(sim_np[idx_i, idx_j])

        within_sims = []
        for t in range(num_targets):
            target_keys = [k for k in keys if k[0] == t and k[1] != s]
            this_key = (t, s)
            if this_key not in combo_features:
                continue
            idx_this = keys.index(this_key)
            for tk in target_keys:
                within_sims.append(sim_np[idx_this, keys.index(tk)])

        if cross_sims and within_sims:
            state_labels.append(state_names[s])
            cross_vals.append(np.mean(cross_sims))
            within_vals.append(np.mean(within_sims))

    x = np.arange(len(state_labels))
    bars1 = ax.bar(x - bar_width/2, cross_vals, bar_width, label='Cross-target (same state)',
                   color='steelblue', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + bar_width/2, within_vals, bar_width, label='Within-target (diff state)',
                   color='coral', edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(state_labels, fontsize=9)
    ax.set_ylabel('Mean Cosine Similarity')
    ax.set_title('State Transferability Check\n'
                 'If blue > red: state semantics transfer across targets')
    ax.set_ylim(0, 1)
    ax.legend()

    for bar, val in zip(bars1, cross_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, within_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path3 = os.path.join(args.output_dir, 'within_vs_cross.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path3}")

    # Figure 4: t-SNE of all combinations
    try:
        from sklearn.manifold import TSNE
        if len(keys) >= 5:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(keys)-1))
            coords = tsne.fit_transform(sim_np)  # Use similarity as input

            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            # Color by state
            state_colors = plt.cm.Set1(np.linspace(0, 1, num_states))
            for s in range(num_states):
                mask = [k[1] == s for k in keys]
                if not any(mask):
                    continue
                idx = [i for i, m in enumerate(mask) if m]
                ax.scatter(coords[idx, 0], coords[idx, 1],
                          c=[state_colors[s]], label=state_names[s],
                          s=100, edgecolors='black', linewidth=0.5)
                for i in idx:
                    ax.annotate(combo_labels[keys[i]], (coords[i, 0], coords[i, 1]),
                               fontsize=6, alpha=0.7,
                               xytext=(5, 5), textcoords='offset points')

            ax.set_title('t-SNE of (Target, State) Feature Embeddings\n'
                         'Same color = same state (should cluster if transferable)')
            ax.legend(loc='best', fontsize=8)
            plt.tight_layout()
            path4 = os.path.join(args.output_dir, 'tsne_embeddings.png')
            plt.savefig(path4, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {path4}")
    except ImportError:
        print("  sklearn not available, skipping t-SNE")

    # Save summary
    summary_path = os.path.join(args.output_dir, 'summary.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# State Transferability Analysis\n\n")
        f.write(f"- Checkpoint: {args.ckpt}\n")
        f.write(f"- Split: {args.split}, Batches: {args.max_batches}\n")
        f.write(f"- Combinations analyzed: {len(combo_features)}\n")
        f.write(f"- Min pixels per combo: {args.min_pixels}\n\n")

        f.write("## Cross-Target State Similarity\n\n")
        f.write("| State | N Targets | Mean Similarity | Std | Assessment |\n")
        f.write("|-------|-----------|-----------------|-----|------------|\n")
        for s_name, data in cross_target_results.items():
            mean = data['mean']
            if mean >= 0.7:
                assess = "Highly transferable"
            elif mean >= 0.5:
                assess = "Moderately transferable"
            elif mean >= 0.3:
                assess = "Weakly transferable"
            else:
                assess = "Not transferable"
            f.write(f"| {s_name} | {data['n_targets']} | {mean:.3f} | {data['std']:.3f} | {assess} |\n")

        f.write("\n## Within vs Cross Target Comparison\n\n")
        f.write("| State | Cross-target | Within-target | Ratio | Verdict |\n")
        f.write("|-------|-------------|---------------|-------|--------|\n")
        for i, s_name in enumerate(state_labels):
            ratio = cross_vals[i] / max(within_vals[i], 0.001)
            verdict = "✓ Transferable" if cross_vals[i] > within_vals[i] else "✗ Target-specific"
            f.write(f"| {s_name} | {cross_vals[i]:.3f} | {within_vals[i]:.3f} | {ratio:.3f} | {verdict} |\n")

        f.write("\n## Conclusion\n\n")
        transferable_states = [s for i, s in enumerate(state_labels) if cross_vals[i] > within_vals[i]]
        non_transferable = [s for i, s in enumerate(state_labels) if cross_vals[i] <= within_vals[i]]
        if transferable_states:
            f.write(f"**Transferable states** (cross-target > within-target): {', '.join(transferable_states)}\n\n")
        if non_transferable:
            f.write(f"**Target-specific states** (cross-target ≤ within-target): {', '.join(non_transferable)}\n\n")

        if transferable_states:
            f.write("**Implication**: States with high cross-target similarity can benefit from "
                    "shared state encoding across targets. Rare targets' state predictions can "
                    "borrow strength from common targets.\n")
        else:
            f.write("**Implication**: State semantics appear to be target-specific. "
                    "Cross-target state transfer may not be effective. "
                    "Consider target-conditioned state heads or data augmentation instead.\n")

    print(f"\nSummary saved: {summary_path}")
    print(f"\nAll outputs in: {args.output_dir}")


if __name__ == '__main__':
    args = parse_args()
    run_analysis(args)
