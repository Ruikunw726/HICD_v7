"""
Cross-Dataset Domain Gap Analysis (Layer 1)

系统量化四个数据集之间的域差距，使用当前 DINOv3 backbone。

实验:
  1.1 特征分布距离: MMD, A-distance, 余弦相似度（各层）
  1.2 数据集可分性: 数据集分类器准确率（各层）
  1.4 标签空间兼容性: 输出空间差异分析

数据集:
  - 0617final: 战争损毁 (15类目标 × 7类状态)
  - XBD: 灾害损毁 (建筑, 4级严重度)
  - SECOND: 城市变化 (6类, from-to)
  - LEVIR-CD: 建筑变化 (二值)

用法:
  python script/domain_gap_analysis.py --max_per_dataset 500
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def parse_args():
    parser = argparse.ArgumentParser(description='Cross-Dataset Domain Gap Analysis')
    parser.add_argument('--output_dir', type=str, default='F:/mambacd/working_notes/domain_gap_v2')
    parser.add_argument('--max_per_dataset', type=int, default=500,
                        help='Max samples per dataset')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--backbone', type=str, default='dinov3_base')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


# ============================================================
# Dataset Loaders
# ============================================================

def load_0617final(data_dir, split, max_samples):
    """Load 0617final war damage dataset."""
    img_dir = os.path.join(data_dir, split, 'image')
    files = sorted([f for f in os.listdir(img_dir) if f.endswith('_pre_war.tif')])
    if len(files) > max_samples:
        np.random.seed(42)
        idx = np.random.choice(len(files), max_samples, replace=False)
        files = [files[i] for i in sorted(idx)]

    samples = []
    for f in files:
        pre_path = os.path.join(img_dir, f)
        post_path = pre_path.replace('_pre_war.tif', '_post_war.tif')
        if os.path.exists(post_path):
            samples.append({'pre': pre_path, 'post': post_path, 'dataset': '0617final'})
    return samples


def load_xbd(data_dir, split, max_samples):
    """Load XBD disaster dataset."""
    img_dir = os.path.join(data_dir, split, 'image')
    files = sorted([f for f in os.listdir(img_dir) if f.endswith('_pre_war.png')])
    if len(files) > max_samples:
        np.random.seed(42)
        idx = np.random.choice(len(files), max_samples, replace=False)
        files = [files[i] for i in sorted(idx)]

    samples = []
    for f in files:
        pre_path = os.path.join(img_dir, f)
        post_path = pre_path.replace('_pre_war.png', '_post_war.png')
        if os.path.exists(post_path):
            samples.append({'pre': pre_path, 'post': post_path, 'dataset': 'XBD'})
    return samples


def load_second(data_dir, split, max_samples):
    """Load SECOND urban change dataset."""
    img_dir = os.path.join(data_dir, split, 'image')
    files = sorted([f for f in os.listdir(img_dir) if f.endswith('_pre_war.tif')])
    if len(files) > max_samples:
        np.random.seed(42)
        idx = np.random.choice(len(files), max_samples, replace=False)
        files = [files[i] for i in sorted(idx)]

    samples = []
    for f in files:
        pre_path = os.path.join(img_dir, f)
        post_path = pre_path.replace('_pre_war.tif', '_post_war.tif')
        if os.path.exists(post_path):
            samples.append({'pre': pre_path, 'post': post_path, 'dataset': 'SECOND'})
    return samples


def load_levir(data_dir, split, max_samples):
    """Load LEVIR-CD building change dataset."""
    a_dir = os.path.join(data_dir, split, 'A')
    files = sorted([f for f in os.listdir(a_dir) if f.endswith('.png')])
    if len(files) > max_samples:
        np.random.seed(42)
        idx = np.random.choice(len(files), max_samples, replace=False)
        files = [files[i] for i in sorted(idx)]

    samples = []
    for f in files:
        pre_path = os.path.join(a_dir, f)
        post_path = os.path.join(data_dir, split, 'B', f)
        if os.path.exists(post_path):
            samples.append({'pre': pre_path, 'post': post_path, 'dataset': 'LEVIR-CD'})
    return samples


def read_image(path, img_size=256):
    """Read image and resize to img_size."""
    from PIL import Image
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.tif', '.tiff']:
        try:
            from osgeo import gdal
            gdal.UseExceptions()
            ds = gdal.Open(path)
            arr = ds.ReadAsArray()
            ds = None
            if arr.ndim == 3 and arr.shape[0] > 3:
                arr = arr[:3]
            img = np.transpose(arr, (1, 2, 0)).astype(np.float32) / 255.0
        except Exception:
            img = np.array(Image.open(path).convert('RGB')).astype(np.float32) / 255.0
    else:
        img = np.array(Image.open(path).convert('RGB')).astype(np.float32) / 255.0

    # Resize if needed
    if img.shape[0] != img_size or img.shape[1] != img_size:
        from PIL import Image as PILImage
        img_pil = PILImage.fromarray((img * 255).astype(np.uint8))
        img_pil = img_pil.resize((img_size, img_size), PILImage.BILINEAR)
        img = np.array(img_pil).astype(np.float32) / 255.0

    return img


# ============================================================
# Feature Extraction
# ============================================================

class FeatureExtractor:
    """Extract multi-stage features from DINOv3 backbone."""

    def __init__(self, backbone_name='dinov3_base', device='cpu'):
        self.device = torch.device(device)
        self.backbone_name = backbone_name

        from models.backbone.dinov3 import DINOv3Backbone
        self.backbone = DINOv3Backbone(
            variant=backbone_name,
            pretrained=True,
            out_channels=[64, 128, 256, 512],
            freeze_stages=2,
            pretrained_path='weights/model.safetensors',
        ).to(self.device)
        self.backbone.eval()

        # Stage names for analysis
        self.stage_names = ['S0_input', 'S1_early', 'S2_mid', 'S3_late', 'S4_deep', 'S5_final']

    def extract(self, img_np):
        """
        Extract features at multiple stages.
        img_np: numpy array [H, W, 3], float32, [0, 1]
        Returns: dict of stage_name -> feature_vector (1D, global pooled)
        """
        img = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get intermediate features by hooking into backbone
            features = {}

            # Stage 0: Input statistics (color histogram proxy)
            features['S0_input'] = self._input_stats(img)

            # Forward through backbone
            x = self.backbone.patch_embed(img)
            H_p, W_p = x.shape[1], x.shape[2]
            x, rot_pos_embed = self.backbone._pos_embed(x)
            x = self.backbone.norm_pre(x)

            # Extract at each block
            prefix = self.backbone.num_prefix_tokens
            extract_points = {
                'S1_early': 2,   # Block 2
                'S2_mid': 5,     # Block 5
                'S3_late': 8,    # Block 8
                'S4_deep': 11,   # Block 11 (last)
            }

            for i, blk in enumerate(self.backbone.blocks):
                if getattr(self.backbone.backbone, 'rope_mixed', False) and rot_pos_embed is not None:
                    x = blk(x, rope=rot_pos_embed[i])
                else:
                    x = blk(x, rope=rot_pos_embed)

                # Extract at specified blocks
                for stage_name, block_idx in extract_points.items():
                    if i == block_idx:
                        feat_spatial = x[:, prefix:, :]  # [B, N, D]
                        # Global average pool
                        feat_pooled = feat_spatial.mean(dim=1).squeeze(0)  # [D]
                        features[stage_name] = feat_pooled.cpu().numpy()

            # Final norm + projection (S5)
            x = self.backbone.norm(x)
            feat_spatial = x[:, prefix:, :]
            feat_pooled = feat_spatial.mean(dim=1).squeeze(0)
            features['S5_final'] = feat_pooled.cpu().numpy()

        return features

    def _input_stats(self, img):
        """Compute input-level statistics as S0 features."""
        img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        # Color statistics: mean, std per channel
        stats = []
        for c in range(3):
            channel = img_np[:, :, c]
            stats.extend([channel.mean(), channel.std()])
        # Gradient magnitude statistics
        gray = img_np.mean(axis=2)
        gy, gx = np.gradient(gray)
        grad_mag = np.sqrt(gx**2 + gy**2)
        stats.extend([grad_mag.mean(), grad_mag.std()])
        # Frequency domain energy
        fft = np.fft.fft2(gray)
        fft_mag = np.abs(fft)
        h, w = fft_mag.shape
        # Low freq (center) vs high freq (corners)
        low_freq = fft_mag[h//4:3*h//4, w//4:3*w//4].mean()
        high_freq = (fft_mag[:h//4, :].mean() + fft_mag[3*h//4:, :].mean() +
                     fft_mag[:, :w//4].mean() + fft_mag[:, 3*w//4:].mean()) / 4
        stats.extend([low_freq, high_freq, high_freq / (low_freq + 1e-8)])
        return np.array(stats, dtype=np.float32)


# ============================================================
# Distance Metrics
# ============================================================

def compute_mmd(features_a, features_b, kernel='rbf', gamma=None):
    """Compute Maximum Mean Discrepancy between two feature sets."""
    n_a, n_b = len(features_a), len(features_b)
    if n_a < 2 or n_b < 2:
        return 0.0

    # Concatenate for gamma estimation
    all_features = np.vstack([features_a, features_b])
    if gamma is None:
        # Median heuristic
        from scipy.spatial.distance import pdist
        dists = pdist(all_features, metric='euclidean')
        gamma = 1.0 / (2 * np.median(dists)**2 + 1e-8)

    def rbf_kernel(X, Y):
        XX = np.sum(X**2, axis=1, keepdims=True)
        YY = np.sum(Y**2, axis=1, keepdims=True)
        XY = X @ Y.T
        dist = XX + YY.T - 2 * XY
        return np.exp(-gamma * dist)

    K_aa = rbf_kernel(features_a, features_a)
    K_bb = rbf_kernel(features_b, features_b)
    K_ab = rbf_kernel(features_a, features_b)

    mmd = K_aa.mean() + K_bb.mean() - 2 * K_ab.mean()
    return max(mmd, 0.0)


def compute_a_distance(features_a, features_b):
    """Compute A-distance (proxy for domain divergence)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X = np.vstack([features_a, features_b])
    y = np.concatenate([np.zeros(len(features_a)), np.ones(len(features_b))])

    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X, y, cv=min(5, min(len(features_a), len(features_b))))
    acc = scores.mean()

    # A-distance = 2 * (1 - 2 * epsilon), where epsilon is the error rate
    epsilon = 1.0 - acc
    a_distance = 2 * (1 - 2 * epsilon)
    return max(a_distance, 0.0), acc


def compute_cosine_similarity(features_a, features_b):
    """Compute mean cosine similarity between two feature sets."""
    # Mean feature for each set
    mean_a = features_a.mean(axis=0)
    mean_b = features_b.mean(axis=0)

    # Cosine similarity between means
    cos_sim = np.dot(mean_a, mean_b) / (np.linalg.norm(mean_a) * np.linalg.norm(mean_b) + 1e-8)

    # Also compute pairwise average
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(features_a, features_b)
    pairwise_mean = sim_matrix.mean()

    return cos_sim, pairwise_mean


# ============================================================
# Dataset Classifier (Experiment 1.2)
# ============================================================

def train_dataset_classifier(features_dict, dataset_names):
    """Train a classifier to predict dataset identity from features."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder

    X_list = []
    y_list = []
    for name in dataset_names:
        feats = features_dict[name]
        X_list.append(feats)
        y_list.extend([name] * len(feats))

    X = np.vstack(X_list)
    y = np.array(y_list)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    n_splits = min(5, min([len(features_dict[n]) for n in dataset_names]))
    scores = cross_val_score(clf, X, y_encoded, cv=n_splits)

    return scores.mean(), scores.std()


# ============================================================
# Main Analysis
# ============================================================

def run_analysis(args):
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    # Dataset paths
    datasets = {
        '0617final': {
            'loader': load_0617final,
            'path': 'D:/CD/0617final_256',
            'type': 'war_damage',
        },
        'XBD': {
            'loader': load_xbd,
            'path': 'D:/CD/XBD_256',
            'type': 'disaster_damage',
        },
        'SECOND': {
            'loader': load_second,
            'path': 'D:/CD/SECOND_6class_v2',
            'type': 'urban_change',
        },
        'LEVIR-CD': {
            'loader': load_levir,
            'path': 'D:/CD/LEVIR-CD_256',
            'type': 'building_change',
        },
    }

    dataset_names = list(datasets.keys())
    print(f"Datasets: {dataset_names}")
    print(f"Max samples per dataset: {args.max_per_dataset}")

    # Initialize feature extractor
    print(f"\nLoading backbone: {args.backbone}")
    extractor = FeatureExtractor(args.backbone, args.device)

    # Load samples and extract features
    all_features = {name: defaultdict(list) for name in dataset_names}
    stage_names = extractor.stage_names

    for name in dataset_names:
        info = datasets[name]
        print(f"\n--- {name} ({info['type']}) ---")
        samples = info['loader'](info['path'], 'train', args.max_per_dataset)
        print(f"  Loaded {len(samples)} samples")

        for i, sample in enumerate(samples):
            if (i + 1) % 100 == 0:
                print(f"  Processing {i+1}/{len(samples)}...")

            try:
                img_pre = read_image(sample['pre'], args.img_size)
                img_post = read_image(sample['post'], args.img_size)

                # Extract features from both pre and post
                feats_pre = extractor.extract(img_pre)
                feats_post = extractor.extract(img_post)

                # Store both
                for stage in stage_names:
                    all_features[name][stage].append(feats_pre[stage])
                    all_features[name][stage].append(feats_post[stage])

            except Exception as e:
                if i < 5:
                    print(f"  Error processing {sample['pre']}: {e}")
                continue

    # Convert to numpy arrays
    for name in dataset_names:
        for stage in stage_names:
            if all_features[name][stage]:
                all_features[name][stage] = np.array(all_features[name][stage])
            else:
                all_features[name][stage] = np.zeros((1, 64))

    # Print feature dimensions
    print("\n--- Feature Dimensions ---")
    for stage in stage_names:
        dim = all_features[dataset_names[0]][stage].shape[1]
        n = all_features[dataset_names[0]][stage].shape[0]
        print(f"  {stage}: dim={dim}, n={n}")

    # ============================================================
    # Experiment 1.1: Distribution Distance
    # ============================================================
    print("\n" + "=" * 70)
    print("Experiment 1.1: Feature Distribution Distance")
    print("=" * 70)

    results_11 = {}
    for stage in stage_names:
        print(f"\n--- {stage} ---")
        results_11[stage] = {}

        for i, name_a in enumerate(dataset_names):
            for j, name_b in enumerate(dataset_names):
                if i >= j:
                    continue
                feats_a = all_features[name_a][stage]
                feats_b = all_features[name_b][stage]

                # MMD
                mmd = compute_mmd(feats_a, feats_b)

                # A-distance
                a_dist, a_acc = compute_a_distance(feats_a, feats_b)

                # Cosine similarity
                cos_sim, pairwise_cos = compute_cosine_similarity(feats_a, feats_b)

                key = f"{name_a} vs {name_b}"
                results_11[stage][key] = {
                    'mmd': mmd, 'a_distance': a_dist, 'a_accuracy': a_acc,
                    'cosine_mean': cos_sim, 'cosine_pairwise': pairwise_cos,
                }
                print(f"  {key}: MMD={mmd:.4f}, A-dist={a_dist:.4f}, "
                      f"Cos(mean)={cos_sim:.4f}, Cos(pair)={pairwise_cos:.4f}")

    # ============================================================
    # Experiment 1.2: Dataset Separability
    # ============================================================
    print("\n" + "=" * 70)
    print("Experiment 1.2: Dataset Separability (Classification Accuracy)")
    print("=" * 70)

    results_12 = {}
    for stage in stage_names:
        features_dict = {name: all_features[name][stage] for name in dataset_names}
        acc, std = train_dataset_classifier(features_dict, dataset_names)
        results_12[stage] = {'accuracy': acc, 'std': std}
        print(f"  {stage}: {acc:.4f} ± {std:.4f}")

    # ============================================================
    # Experiment 1.4: Label Space Compatibility
    # ============================================================
    print("\n" + "=" * 70)
    print("Experiment 1.4: Label Space Compatibility")
    print("=" * 70)

    label_spaces = {
        '0617final': {
            'type': 'target_state_joint',
            'num_targets': 15,
            'num_states': 7,
            'num_combinations': 48,  # valid combinations
            'description': '15 targets × 7 states, 48 valid combos',
        },
        'XBD': {
            'type': 'damage_severity',
            'num_classes': 5,  # background + 4 severity levels
            'description': 'Building damage severity (4 levels)',
        },
        'SECOND': {
            'type': 'from_to_transition',
            'num_classes': 6,
            'description': '6-class land cover from-to transitions',
        },
        'LEVIR-CD': {
            'type': 'binary_change',
            'num_classes': 2,
            'description': 'Binary changed/unchanged',
        },
    }

    print("\nLabel Space Summary:")
    for name, info in label_spaces.items():
        print(f"  {name}: {info['description']}")

    # Compute output space dimensions
    output_dims = {
        '0617final': 15 + 7,  # target + state (joint, but approximate)
        'XBD': 5,
        'SECOND': 6,
        'LEVIR-CD': 2,
    }

    print("\nOutput Space Dimensionality:")
    for name, dim in output_dims.items():
        print(f"  {name}: {dim}")

    # ============================================================
    # Visualization
    # ============================================================
    print("\nGenerating visualizations...")

    # Figure 1: Distance heatmap across stages
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, stage in enumerate(stage_names):
        ax = axes[idx]
        n = len(dataset_names)
        dist_matrix = np.zeros((n, n))

        for i, name_a in enumerate(dataset_names):
            for j, name_b in enumerate(dataset_names):
                if i == j:
                    dist_matrix[i, j] = 0
                else:
                    key = f"{name_a} vs {name_b}" if i < j else f"{name_b} vs {name_a}"
                    if key in results_11[stage]:
                        dist_matrix[i, j] = results_11[stage][key]['a_distance']

        im = ax.imshow(dist_matrix, cmap='RdYlBu_r', vmin=0, vmax=2, aspect='auto')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(dataset_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(dataset_names, fontsize=8)
        ax.set_title(f'{stage}\n(A-distance)', fontsize=10)

        # Annotate
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{dist_matrix[i,j]:.2f}',
                       ha='center', va='center', fontsize=8)

    plt.colorbar(im, ax=axes, shrink=0.6)
    plt.suptitle('Cross-Dataset Domain Gap (A-distance) Across Network Stages', fontsize=14)
    plt.tight_layout()
    path1 = os.path.join(args.output_dir, 'domain_gap_by_stage.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path1}")

    # Figure 2: Dataset separability across stages
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    stages = stage_names
    accs = [results_12[s]['accuracy'] for s in stages]
    stds = [results_12[s]['std'] for s in stages]

    ax.bar(range(len(stages)), accs, yerr=stds, capsize=5,
           color='steelblue', edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=45, ha='right')
    ax.set_ylabel('Dataset Classification Accuracy')
    ax.set_title('Dataset Separability Across Network Stages\n'
                 '(Higher = more dataset-specific features)')
    ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Random (4 classes)')
    ax.set_ylim(0, 1.05)
    ax.legend()

    for i, (acc, std) in enumerate(zip(accs, stds)):
        ax.text(i, acc + std + 0.02, f'{acc:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    path2 = os.path.join(args.output_dir, 'dataset_separability.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path2}")

    # Figure 3: Cosine similarity heatmap (final stage)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    n = len(dataset_names)
    cos_matrix = np.zeros((n, n))

    for i, name_a in enumerate(dataset_names):
        for j, name_b in enumerate(dataset_names):
            if i == j:
                cos_matrix[i, j] = 1.0
            else:
                key = f"{name_a} vs {name_b}" if i < j else f"{name_b} vs {name_a}"
                if key in results_11['S5_final']:
                    cos_matrix[i, j] = results_11['S5_final'][key]['cosine_mean']

    im = ax.imshow(cos_matrix, cmap='RdYlBu', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax.set_yticklabels(dataset_names)
    ax.set_title('Cross-Dataset Cosine Similarity (S5 Final Stage)')

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{cos_matrix[i,j]:.3f}',
                   ha='center', va='center', fontsize=9)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    path3 = os.path.join(args.output_dir, 'cosine_similarity_final.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path3}")

    # ============================================================
    # Summary Report
    # ============================================================
    summary_path = os.path.join(args.output_dir, 'summary.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Cross-Dataset Domain Gap Analysis v2\n\n")
        f.write(f"- Backbone: {args.backbone}\n")
        f.write(f"- Max samples per dataset: {args.max_per_dataset}\n")
        f.write(f"- Image size: {args.img_size}×{args.img_size}\n\n")

        f.write("## Datasets\n\n")
        f.write("| Dataset | Type | Description |\n")
        f.write("|---------|------|-------------|\n")
        for name in dataset_names:
            f.write(f"| {name} | {datasets[name]['type']} | {label_spaces[name]['description']} |\n")

        f.write("\n## Experiment 1.1: Feature Distribution Distance\n\n")
        for stage in stage_names:
            f.write(f"### {stage}\n\n")
            f.write("| Pair | MMD | A-distance | Cosine (mean) | Cosine (pairwise) |\n")
            f.write("|------|-----|------------|---------------|-------------------|\n")
            for key, vals in results_11[stage].items():
                f.write(f"| {key} | {vals['mmd']:.4f} | {vals['a_distance']:.4f} | "
                        f"{vals['cosine_mean']:.4f} | {vals['cosine_pairwise']:.4f} |\n")
            f.write("\n")

        f.write("## Experiment 1.2: Dataset Separability\n\n")
        f.write("| Stage | Accuracy | Std | Interpretation |\n")
        f.write("|-------|----------|-----|----------------|\n")
        for stage in stage_names:
            acc = results_12[stage]['accuracy']
            std = results_12[stage]['std']
            if acc > 0.8:
                interp = "Highly separable (strong domain gap)"
            elif acc > 0.5:
                interp = "Moderately separable"
            else:
                interp = "Weakly separable (features are domain-invariant)"
            f.write(f"| {stage} | {acc:.4f} | {std:.4f} | {interp} |\n")

        f.write("\n## Experiment 1.4: Label Space Compatibility\n\n")
        f.write("| Dataset | Output Type | Dimensionality | Notes |\n")
        f.write("|---------|-------------|----------------|-------|\n")
        for name in dataset_names:
            info = label_spaces[name]
            dim = output_dims[name]
            f.write(f"| {name} | {info['type']} | {dim} | {info['description']} |\n")

        f.write("\n## Key Findings\n\n")
        # Auto-generate findings
        s0_acc = results_12['S0_input']['accuracy']
        s5_acc = results_12['S5_final']['accuracy']
        if s5_acc > s0_acc:
            f.write(f"1. **Hierarchical pattern confirmed**: Dataset separability increases from "
                    f"S0 ({s0_acc:.3f}) to S5 ({s5_acc:.3f}), confirming that deeper features "
                    f"are more dataset-specific.\n\n")
        else:
            f.write(f"1. **Unexpected pattern**: Dataset separability does not increase with depth "
                    f"(S0: {s0_acc:.3f}, S5: {s5_acc:.3f}). This contradicts the previous finding "
                    f"and needs investigation.\n\n")

        # Check 0617final vs XBD (both damage)
        key_damage = "0617final vs XBD"
        key_change = "SECOND vs LEVIR-CD"
        if key_damage in results_11['S5_final'] and key_change in results_11['S5_final']:
            a_dmg = results_11['S5_final'][key_damage]['a_distance']
            a_chg = results_11['S5_final'][key_change]['a_distance']
            f.write(f"2. **Damage vs Change separation**: "
                    f"0617final-XBD (both damage) A-distance={a_dmg:.3f}, "
                    f"SECOND-LEVIR (both change) A-distance={a_chg:.3f}. "
                    f"{'Damage datasets are closer' if a_dmg < a_chg else 'Change datasets are closer'}.\n\n")

        f.write("## Implications\n\n")
        f.write("- If S5 separability is high (>0.8): Cross-dataset transfer will be challenging "
                "without adaptation.\n")
        f.write("- If 0617final-XBD similarity is high: Transfer between damage datasets is feasible.\n")
        f.write("- Label space incompatibility: Output heads need redesign for cross-dataset transfer.\n")

    print(f"\nSummary saved: {summary_path}")
    print(f"All outputs in: {args.output_dir}")


if __name__ == '__main__':
    args = parse_args()
    run_analysis(args)
