"""
Domain Gap Analysis: Layer 2 (Cross-Scene) + Layer 3 (Cross-City)

基于原始数据目录结构提取场景类型和城市信息。

Layer 2: 机场 vs 港口 vs 城乡 的特征距离
Layer 3: 城市之间的特征距离

用法:
  python script/domain_gap_scene_city.py --max_per_scene 200 --max_per_city 100
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# City -> Scene mapping from directory scan
CITY_SCENE_MAP = {
    # Airports
    'Donetsk': ['Airports', 'Urban-Rural Areas'],
    'Dyagilevo': ['Airports'],
    'Kyiv': ['Airports'],
    'Melitopol': ['Airports'],
    'Mykolaiv': ['Airports', 'Ports'],
    'Saky': ['Airports'],
    'Tripoli': ['Airports'],
    # Ports
    'Kherson': ['Ports', 'Urban-Rural Areas'],
    'Mariupol': ['Ports', 'Urban-Rural Areas'],
    'Novorossiysk': ['Ports'],
    'Sevastopol': ['Ports'],
    # Urban-Rural
    'Al-Fallujah': ['Urban-Rural Areas'],
    'Al-jazirah': ['Urban-Rural Areas'],
    'Avdiivka': ['Urban-Rural Areas'],
    'Bakhmut': ['Urban-Rural Areas'],
    'Buthidaung': ['Urban-Rural Areas'],
    'Idlib': ['Urban-Rural Areas'],
    'Khartoum': ['Urban-Rural Areas'],
    'Lysychansk': ['Urban-Rural Areas'],
    'Mosul': ['Urban-Rural Areas'],
    'Sievierodonetsk': ['Urban-Rural Areas'],
    'Soledar': ['Urban-Rural Areas'],
}

# Primary scene (for grouping when city appears in multiple)
CITY_PRIMARY_SCENE = {
    'Donetsk': 'Airports',
    'Dyagilevo': 'Airports',
    'Kyiv': 'Airports',
    'Melitopol': 'Airports',
    'Mykolaiv': 'Ports',
    'Saky': 'Airports',
    'Tripoli': 'Airports',
    'Kherson': 'Ports',
    'Mariupol': 'Ports',
    'Novorossiysk': 'Ports',
    'Sevastopol': 'Ports',
    'Al-Fallujah': 'Urban-Rural Areas',
    'Al-jazirah': 'Urban-Rural Areas',
    'Avdiivka': 'Urban-Rural Areas',
    'Bakhmut': 'Urban-Rural Areas',
    'Buthidaung': 'Urban-Rural Areas',
    'Idlib': 'Urban-Rural Areas',
    'Khartoum': 'Urban-Rural Areas',
    'Lysychansk': 'Urban-Rural Areas',
    'Mosul': 'Urban-Rural Areas',
    'Sievierodonetsk': 'Urban-Rural Areas',
    'Soledar': 'Urban-Rural Areas',
}

SCENE_NAMES = ['Airports', 'Ports', 'Urban-Rural Areas']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='D:/CD/0617final')
    parser.add_argument('--output_dir', type=str, default='F:/mambacd/working_notes/domain_gap_scene_city')
    parser.add_argument('--max_per_scene', type=int, default=200)
    parser.add_argument('--max_per_city', type=int, default=80)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_scene_samples(data_dir, scene, split, max_samples):
    """Load samples from a specific scene directory."""
    img_dir = os.path.join(data_dir, scene, split, 'image')
    if not os.path.exists(img_dir):
        return []

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
            city = f.split('_')[0]
            samples.append({
                'pre': pre_path, 'post': post_path,
                'city': city, 'scene': scene,
            })
    return samples


def read_image(path, img_size=256):
    from PIL import Image
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

    if img.shape[0] != img_size or img.shape[1] != img_size:
        from PIL import Image as PILImage
        img_pil = PILImage.fromarray((img * 255).astype(np.uint8))
        img_pil = img_pil.resize((img_size, img_size), PILImage.BILINEAR)
        img = np.array(img_pil).astype(np.float32) / 255.0
    return img


class FeatureExtractor:
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        from models.backbone.dinov3 import DINOv3Backbone
        self.backbone = DINOv3Backbone(
            variant='dinov3_base', pretrained=True,
            out_channels=[64, 128, 256, 512], freeze_stages=2,
            pretrained_path='weights/model.safetensors',
        ).to(self.device)
        self.backbone.eval()

    def extract(self, img_np):
        img = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            x = self.backbone.patch_embed(img)
            x, rot_pos_embed = self.backbone._pos_embed(x)
            x = self.backbone.norm_pre(x)
            prefix = self.backbone.num_prefix_tokens

            features = {}
            for i, blk in enumerate(self.backbone.blocks):
                if getattr(self.backbone.backbone, 'rope_mixed', False) and rot_pos_embed is not None:
                    x = blk(x, rope=rot_pos_embed[i])
                else:
                    x = blk(x, rope=rot_pos_embed)

                # Extract at blocks 2 (S1), 5 (S2), 8 (S3), 11 (S5)
                if i in [2, 5, 8, 11]:
                    feat_spatial = x[:, prefix:, :]
                    feat_pooled = feat_spatial.mean(dim=1).squeeze(0)
                    stage_name = {2: 'S1_early', 5: 'S2_mid', 8: 'S3_late', 11: 'S5_final'}[i]
                    features[stage_name] = feat_pooled.cpu().numpy()

            x = self.backbone.norm(x)
            feat_spatial = x[:, prefix:, :]
            feat_pooled = feat_spatial.mean(dim=1).squeeze(0)
            features['S5_final'] = feat_pooled.cpu().numpy()

        return features


def compute_a_distance(features_a, features_b):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    X = np.vstack([features_a, features_b])
    y = np.concatenate([np.zeros(len(features_a)), np.ones(len(features_b))])
    clf = LogisticRegression(max_iter=1000, random_state=42)
    n_splits = min(5, min(len(features_a), len(features_b)))
    scores = cross_val_score(clf, X, y, cv=n_splits)
    acc = scores.mean()
    epsilon = 1.0 - acc
    return max(2 * (1 - 2 * epsilon), 0.0), acc


def train_classifier(features_dict, labels):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    X_list, y_list = [], []
    for label in labels:
        if label in features_dict and len(features_dict[label]) > 0:
            X_list.append(features_dict[label])
            y_list.extend([label] * len(features_dict[label]))
    X = np.vstack(X_list)
    y = np.array(y_list)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    n_splits = min(5, min([len(features_dict[l]) for l in labels if l in features_dict]))
    scores = cross_val_score(clf, X, y_encoded, cv=n_splits)
    return scores.mean(), scores.std()


def run_analysis(args):
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    stages = ['S1_early', 'S2_mid', 'S3_late', 'S5_final']

    print(f"Data dir: {args.data_dir}")
    print(f"Scenes: {SCENE_NAMES}")
    print(f"Max per scene: {args.max_per_scene}, Max per city: {args.max_per_city}")

    # Load samples by scene
    print("\nLoading samples...")
    scene_samples = {}
    all_samples = []
    for scene in SCENE_NAMES:
        samples = load_scene_samples(args.data_dir, scene, 'train', args.max_per_scene)
        scene_samples[scene] = samples
        all_samples.extend(samples)
        cities = set(s['city'] for s in samples)
        print(f"  {scene}: {len(samples)} samples, {len(cities)} cities: {sorted(cities)}")

    # Limit per city
    city_counts = defaultdict(int)
    filtered_samples = []
    for s in all_samples:
        if city_counts[s['city']] < args.max_per_city:
            filtered_samples.append(s)
            city_counts[s['city']] += 1
    all_samples = filtered_samples
    print(f"\nTotal after city limit: {len(all_samples)} samples")

    # Extract features
    print("\nExtracting features...")
    extractor = FeatureExtractor(args.device)

    scene_features = {scene: defaultdict(list) for scene in SCENE_NAMES}
    city_features = defaultdict(lambda: defaultdict(list))

    for i, sample in enumerate(all_samples):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(all_samples)}...")

        try:
            img_pre = read_image(sample['pre'], args.img_size)
            feats = extractor.extract(img_pre)

            for stage in stages:
                scene_features[sample['scene']][stage].append(feats[stage])
                city_features[sample['city']][stage].append(feats[stage])
        except Exception as e:
            if i < 5:
                print(f"  Error: {e}")
            continue

    # Convert to numpy
    for scene in SCENE_NAMES:
        for stage in stages:
            if scene_features[scene][stage]:
                scene_features[scene][stage] = np.array(scene_features[scene][stage])
            else:
                scene_features[scene][stage] = np.zeros((1, 768))

    for city in city_features:
        for stage in stages:
            if city_features[city][stage]:
                city_features[city][stage] = np.array(city_features[city][stage])

    # Get cities with enough samples
    valid_cities = [c for c in city_features if len(city_features[c]['S5_final']) >= 20]
    print(f"\nCities with ≥20 samples: {len(valid_cities)}")
    for c in sorted(valid_cities):
        n = len(city_features[c]['S5_final'])
        scene = CITY_PRIMARY_SCENE.get(c, '?')
        print(f"  {c}: {n} samples ({scene})")

    # ============================================================
    # Layer 2: Cross-Scene Analysis
    # ============================================================
    print("\n" + "=" * 70)
    print("Layer 2: Cross-Scene Analysis")
    print("=" * 70)

    scene_results = {}
    for stage in stages:
        print(f"\n--- {stage} ---")
        scene_results[stage] = {}

        for i, scene_a in enumerate(SCENE_NAMES):
            for j, scene_b in enumerate(SCENE_NAMES):
                if i >= j:
                    continue
                feats_a = scene_features[scene_a][stage]
                feats_b = scene_features[scene_b][stage]
                if len(feats_a) < 2 or len(feats_b) < 2:
                    continue
                a_dist, acc = compute_a_distance(feats_a, feats_b)
                key = f"{scene_a} vs {scene_b}"
                scene_results[stage][key] = {'a_distance': a_dist, 'accuracy': acc}
                print(f"  {key}: A-dist={a_dist:.4f}, Acc={acc:.4f}")

    # Scene separability
    print("\nScene Separability:")
    scene_sep = {}
    for stage in stages:
        features_dict = {s: scene_features[s][stage] for s in SCENE_NAMES}
        acc, std = train_classifier(features_dict, SCENE_NAMES)
        scene_sep[stage] = {'acc': acc, 'std': std}
        print(f"  {stage}: {acc:.4f} ± {std:.4f}")

    # ============================================================
    # Layer 3: Cross-City Analysis
    # ============================================================
    print("\n" + "=" * 70)
    print("Layer 3: Cross-City Analysis")
    print("=" * 70)

    # City separability
    print("\nCity Separability:")
    city_sep = {}
    for stage in stages:
        features_dict = {c: city_features[c][stage] for c in valid_cities}
        acc, std = train_classifier(features_dict, valid_cities)
        city_sep[stage] = {'acc': acc, 'std': std}
        print(f"  {stage}: {acc:.4f} ± {std:.4f}")

    # Intra-scene vs Inter-scene city distances
    print("\nIntra-Scene vs Inter-Scene City Distances (S5_final):")
    intra_scene_dists = []
    inter_scene_dists = []

    for i, city_a in enumerate(valid_cities):
        for j, city_b in enumerate(valid_cities):
            if i >= j:
                continue
            feats_a = city_features[city_a]['S5_final']
            feats_b = city_features[city_b]['S5_final']
            if len(feats_a) < 5 or len(feats_b) < 5:
                continue

            a_dist, _ = compute_a_distance(feats_a, feats_b)

            scene_a = CITY_PRIMARY_SCENE.get(city_a, '?')
            scene_b = CITY_PRIMARY_SCENE.get(city_b, '?')

            if scene_a == scene_b:
                intra_scene_dists.append(a_dist)
            else:
                inter_scene_dists.append(a_dist)

    print(f"  Intra-scene (same scene, diff city): mean={np.mean(intra_scene_dists):.4f}, "
          f"n={len(intra_scene_dists)}")
    print(f"  Inter-scene (diff scene, diff city): mean={np.mean(inter_scene_dists):.4f}, "
          f"n={len(inter_scene_dists)}")
    print(f"  Ratio (inter/intra): {np.mean(inter_scene_dists)/max(np.mean(intra_scene_dists), 0.001):.4f}")

    # City distance matrix (S5_final)
    print("\nComputing city distance matrix...")
    n_cities = len(valid_cities)
    city_dist_matrix = np.zeros((n_cities, n_cities))
    for i, city_a in enumerate(valid_cities):
        for j, city_b in enumerate(valid_cities):
            if i == j:
                continue
            feats_a = city_features[city_a]['S5_final']
            feats_b = city_features[city_b]['S5_final']
            if len(feats_a) < 5 or len(feats_b) < 5:
                continue
            a_dist, _ = compute_a_distance(feats_a, feats_b)
            city_dist_matrix[i, j] = a_dist

    # ============================================================
    # Visualizations
    # ============================================================
    print("\nGenerating visualizations...")

    # Figure 1: Scene analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: Scene separability
    ax = axes[0]
    accs = [scene_sep[s]['acc'] for s in stages]
    stds = [scene_sep[s]['std'] for s in stages]
    ax.bar(range(len(stages)), accs, yerr=stds, capsize=5,
           color='steelblue', edgecolor='black', linewidth=0.5)
    ax.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5, label='Random (33%)')
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(['S1', 'S2', 'S3', 'S5'])
    ax.set_ylabel('Scene Classification Accuracy')
    ax.set_title('Scene Separability (Airport/Port/Urban)')
    ax.set_ylim(0, 1.05)
    ax.legend()
    for i, (acc, std) in enumerate(zip(accs, stds)):
        ax.text(i, acc + std + 0.02, f'{acc:.3f}', ha='center', fontsize=9)

    # Panel B: Scene A-distance heatmap
    ax = axes[1]
    n_scenes = len(SCENE_NAMES)
    scene_dist = np.zeros((n_scenes, n_scenes))
    for i, sa in enumerate(SCENE_NAMES):
        for j, sb in enumerate(SCENE_NAMES):
            if i == j:
                continue
            key = f"{sa} vs {sb}" if i < j else f"{sb} vs {sa}"
            if key in scene_results['S5_final']:
                scene_dist[i, j] = scene_results['S5_final'][key]['a_distance']
    im = ax.imshow(scene_dist, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n_scenes))
    ax.set_yticks(range(n_scenes))
    ax.set_xticklabels(['Airport', 'Port', 'Urban'], rotation=45, ha='right')
    ax.set_yticklabels(['Airport', 'Port', 'Urban'])
    ax.set_title('Scene A-distance (S5)')
    for i in range(n_scenes):
        for j in range(n_scenes):
            if i != j:
                ax.text(j, i, f'{scene_dist[i,j]:.3f}', ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel C: Intra vs Inter scene
    ax = axes[2]
    box_data = [intra_scene_dists, inter_scene_dists]
    try:
        bp = ax.boxplot(box_data, tick_labels=['Intra-scene\n(same scene)', 'Inter-scene\n(diff scene)'],
                        patch_artist=True)
    except TypeError:
        bp = ax.boxplot(box_data, labels=['Intra-scene\n(same scene)', 'Inter-scene\n(diff scene)'],
                        patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('A-distance')
    ax.set_title(f'City Distances by Scene Relationship\n'
                 f'Ratio: {np.mean(inter_scene_dists)/max(np.mean(intra_scene_dists), 0.001):.3f}')

    plt.suptitle('Layer 2: Cross-Scene Domain Gap', fontsize=14)
    plt.tight_layout()
    path1 = os.path.join(args.output_dir, 'scene_analysis.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path1}")

    # Figure 2: City analysis
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, hspace=0.3, wspace=0.3)

    # Panel A: City separability
    ax = fig.add_subplot(gs[0, 0])
    accs = [city_sep[s]['acc'] for s in stages]
    stds = [city_sep[s]['std'] for s in stages]
    ax.bar(range(len(stages)), accs, yerr=stds, capsize=5,
           color='coral', edgecolor='black', linewidth=0.5)
    ax.axhline(y=1/len(valid_cities), color='gray', linestyle='--', alpha=0.5,
               label=f'Random ({1/len(valid_cities):.1%})')
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(['S1', 'S2', 'S3', 'S5'])
    ax.set_ylabel('City Classification Accuracy')
    ax.set_title(f'City Separability ({len(valid_cities)} cities)')
    ax.set_ylim(0, 1.05)
    ax.legend()
    for i, (acc, std) in enumerate(zip(accs, stds)):
        ax.text(i, acc + std + 0.02, f'{acc:.3f}', ha='center', fontsize=9)

    # Panel B: Scene vs City separability comparison
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(stages))
    width = 0.35
    scene_accs = [scene_sep[s]['acc'] for s in stages]
    city_accs = [city_sep[s]['acc'] for s in stages]
    ax.bar(x - width/2, scene_accs, width, label='Scene (3 classes)',
           color='steelblue', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, city_accs, width, label=f'City ({len(valid_cities)} classes)',
           color='coral', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['S1', 'S2', 'S3', 'S5'])
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('Scene vs City Separability')
    ax.legend()

    # Panel C: City distance heatmap
    ax = fig.add_subplot(gs[1, :])
    im = ax.imshow(city_dist_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n_cities))
    ax.set_yticks(range(n_cities))
    ax.set_xticklabels(valid_cities, rotation=90, fontsize=7)
    ax.set_yticklabels(valid_cities, fontsize=7)
    ax.set_title('City A-distance Matrix (S5)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Color code by scene
    scene_colors = {'Airports': '#e74c3c', 'Ports': '#3498db', 'Urban-Rural Areas': '#2ecc71'}
    for i, city in enumerate(valid_cities):
        scene = CITY_PRIMARY_SCENE.get(city, '?')
        color = scene_colors.get(scene, 'gray')
        ax.get_yticklabels()[i].set_color(color)
        ax.get_xticklabels()[i].set_color(color)

    plt.suptitle('Layer 3: Cross-City Domain Gap', fontsize=14)
    path2 = os.path.join(args.output_dir, 'city_analysis.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path2}")

    # ============================================================
    # Summary
    # ============================================================
    summary_path = os.path.join(args.output_dir, 'summary.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Domain Gap Analysis: Layer 2 + Layer 3\n\n")
        f.write(f"- Data: {args.data_dir}\n")
        f.write(f"- Max per scene: {args.max_per_scene}, Max per city: {args.max_per_city}\n\n")

        f.write("## City-Scene Mapping\n\n")
        f.write("| City | Primary Scene | All Scenes |\n")
        f.write("|------|---------------|------------|\n")
        for city in sorted(CITY_PRIMARY_SCENE.keys()):
            primary = CITY_PRIMARY_SCENE[city]
            all_scenes = ', '.join(CITY_SCENE_MAP.get(city, []))
            f.write(f"| {city} | {primary} | {all_scenes} |\n")

        f.write("\n## Layer 2: Cross-Scene Analysis\n\n")
        f.write("### Scene Separability\n\n")
        f.write("| Stage | Accuracy | Std |\n")
        f.write("|-------|----------|-----|\n")
        for stage in stages:
            f.write(f"| {stage} | {scene_sep[stage]['acc']:.4f} | {scene_sep[stage]['std']:.4f} |\n")

        f.write("\n### Scene A-distance (S5)\n\n")
        f.write("| Pair | A-distance |\n")
        f.write("|------|------------|\n")
        for key, vals in scene_results['S5_final'].items():
            f.write(f"| {key} | {vals['a_distance']:.4f} |\n")

        f.write("\n## Layer 3: Cross-City Analysis\n\n")
        f.write("### City Separability\n\n")
        f.write("| Stage | Accuracy | Std |\n")
        f.write("|-------|----------|-----|\n")
        for stage in stages:
            f.write(f"| {stage} | {city_sep[stage]['acc']:.4f} | {city_sep[stage]['std']:.4f} |\n")

        f.write("\n### Intra vs Inter Scene\n\n")
        f.write(f"- Intra-scene mean A-distance: {np.mean(intra_scene_dists):.4f}\n")
        f.write(f"- Inter-scene mean A-distance: {np.mean(inter_scene_dists):.4f}\n")
        f.write(f"- Ratio: {np.mean(inter_scene_dists)/max(np.mean(intra_scene_dists), 0.001):.4f}\n\n")

        if np.mean(inter_scene_dists) > np.mean(intra_scene_dists) * 1.2:
            f.write("**Finding**: Inter-scene city distances are significantly larger than "
                    "intra-scene distances, confirming scene type is a meaningful factor.\n\n")
        else:
            f.write("**Finding**: Intra-scene and inter-scene city distances are similar, "
                    "suggesting city identity matters more than scene type.\n\n")

        f.write("## Key Findings\n\n")
        f.write(f"1. Scene separability (S5): {scene_sep['S5_final']['acc']:.3f}\n")
        f.write(f"2. City separability (S5): {city_sep['S5_final']['acc']:.3f}\n")
        f.write(f"3. City >> Scene: {city_sep['S5_final']['acc'] > scene_sep['S5_final']['acc']}\n")

    print(f"\nSummary: {summary_path}")
    print(f"All outputs: {args.output_dir}")


if __name__ == '__main__':
    args = parse_args()
    run_analysis(args)
