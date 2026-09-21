"""
Domain Gap Visualization v2: 改进的可视化

改进:
  1. 用 A-distance 而非 cosine similarity（区分度更高）
  2. 归一化热力图，突出相对差异
  3. 加入按数据集对的趋势线，看差距如何随深度变化
  4. 加入 intra-dataset 对比（数据集内方差 vs 数据集间距离）
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Hardcoded results from the analysis run
# Format: stage -> pair -> {mmd, a_distance, cosine_mean, cosine_pairwise}
results = {
    'S0_input': {
        '0617final vs XBD': {'mmd': 0.185, 'a_distance': 0.86, 'cosine_mean': 0.997},
        '0617final vs SECOND': {'mmd': 0.245, 'a_distance': 1.12, 'cosine_mean': 0.999},
        '0617final vs LEVIR-CD': {'mmd': 0.235, 'a_distance': 1.34, 'cosine_mean': 0.920},
        'XBD vs SECOND': {'mmd': 0.704, 'a_distance': 1.64, 'cosine_mean': 0.994},
        'XBD vs LEVIR-CD': {'mmd': 0.432, 'a_distance': 1.30, 'cosine_mean': 0.942},
        'SECOND vs LEVIR-CD': {'mmd': 0.386, 'a_distance': 1.76, 'cosine_mean': 0.919},
    },
    'S1_early': {
        '0617final vs XBD': {'mmd': 0.048, 'a_distance': 1.64, 'cosine_mean': 0.994},
        '0617final vs SECOND': {'mmd': 0.186, 'a_distance': 1.80, 'cosine_mean': 0.975},
        '0617final vs LEVIR-CD': {'mmd': 0.147, 'a_distance': 1.72, 'cosine_mean': 0.979},
        'XBD vs SECOND': {'mmd': 0.273, 'a_distance': 1.92, 'cosine_mean': 0.962},
        'XBD vs LEVIR-CD': {'mmd': 0.160, 'a_distance': 1.96, 'cosine_mean': 0.978},
        'SECOND vs LEVIR-CD': {'mmd': 0.231, 'a_distance': 1.92, 'cosine_mean': 0.970},
    },
    'S2_mid': {
        '0617final vs XBD': {'mmd': 0.054, 'a_distance': 1.60, 'cosine_mean': 0.980},
        '0617final vs SECOND': {'mmd': 0.274, 'a_distance': 1.70, 'cosine_mean': 0.879},
        '0617final vs LEVIR-CD': {'mmd': 0.049, 'a_distance': 1.86, 'cosine_mean': 0.981},
        'XBD vs SECOND': {'mmd': 0.397, 'a_distance': 1.92, 'cosine_mean': 0.820},
        'XBD vs LEVIR-CD': {'mmd': 0.094, 'a_distance': 1.92, 'cosine_mean': 0.964},
        'SECOND vs LEVIR-CD': {'mmd': 0.273, 'a_distance': 1.88, 'cosine_mean': 0.895},
    },
    'S3_late': {
        '0617final vs XBD': {'mmd': 0.060, 'a_distance': 1.44, 'cosine_mean': 0.953},
        '0617final vs SECOND': {'mmd': 0.252, 'a_distance': 1.72, 'cosine_mean': 0.864},
        '0617final vs LEVIR-CD': {'mmd': 0.039, 'a_distance': 1.62, 'cosine_mean': 0.971},
        'XBD vs SECOND': {'mmd': 0.406, 'a_distance': 1.82, 'cosine_mean': 0.722},
        'XBD vs LEVIR-CD': {'mmd': 0.104, 'a_distance': 1.50, 'cosine_mean': 0.910},
        'SECOND vs LEVIR-CD': {'mmd': 0.228, 'a_distance': 1.94, 'cosine_mean': 0.899},
    },
    'S4_deep': {
        '0617final vs XBD': {'mmd': 0.054, 'a_distance': 1.48, 'cosine_mean': 0.986},
        '0617final vs SECOND': {'mmd': 0.208, 'a_distance': 1.76, 'cosine_mean': 0.971},
        '0617final vs LEVIR-CD': {'mmd': 0.037, 'a_distance': 1.60, 'cosine_mean': 0.993},
        'XBD vs SECOND': {'mmd': 0.311, 'a_distance': 1.94, 'cosine_mean': 0.942},
        'XBD vs LEVIR-CD': {'mmd': 0.072, 'a_distance': 1.62, 'cosine_mean': 0.983},
        'SECOND vs LEVIR-CD': {'mmd': 0.229, 'a_distance': 1.94, 'cosine_mean': 0.965},
    },
    'S5_final': {
        '0617final vs XBD': {'mmd': 0.052, 'a_distance': 1.54, 'cosine_mean': 0.943},
        '0617final vs SECOND': {'mmd': 0.184, 'a_distance': 1.76, 'cosine_mean': 0.880},
        '0617final vs LEVIR-CD': {'mmd': 0.035, 'a_distance': 1.56, 'cosine_mean': 0.969},
        'XBD vs SECOND': {'mmd': 0.274, 'a_distance': 1.96, 'cosine_mean': 0.764},
        'XBD vs LEVIR-CD': {'mmd': 0.069, 'a_distance': 1.56, 'cosine_mean': 0.926},
        'SECOND vs LEVIR-CD': {'mmd': 0.211, 'a_distance': 1.96, 'cosine_mean': 0.868},
    },
}

separability = {
    'S0_input': {'acc': 0.707, 'std': 0.043},
    'S1_early': {'acc': 0.897, 'std': 0.047},
    'S2_mid': {'acc': 0.807, 'std': 0.085},
    'S3_late': {'acc': 0.743, 'std': 0.079},
    'S4_deep': {'acc': 0.742, 'std': 0.068},
    'S5_final': {'acc': 0.722, 'std': 0.068},
}

stages = list(results.keys())
pairs = list(results['S0_input'].keys())
dataset_names = ['0617final', 'XBD', 'SECOND', 'LEVIR-CD']

output_dir = 'F:/mambacd/working_notes/domain_gap_v2'

# ============================================================
# Figure 1: Combined overview (2 panels)
# ============================================================
fig = plt.figure(figsize=(16, 6))
gs = GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.3)

# Panel A: Dataset separability (bar chart with trend)
ax1 = fig.add_subplot(gs[0])
stages_short = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']
accs = [separability[s]['acc'] for s in stages]
stds = [separability[s]['std'] for s in stages]
colors = ['#e74c3c' if a > 0.8 else '#3498db' for a in accs]

bars = ax1.bar(range(len(stages)), accs, yerr=stds, capsize=5,
               color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
ax1.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Random (25%)')
ax1.axhline(y=np.mean(accs), color='green', linestyle=':', alpha=0.7, label=f'Mean ({np.mean(accs):.3f})')

ax1.set_xticks(range(len(stages)))
ax1.set_xticklabels(stages_short, fontsize=11)
ax1.set_ylabel('Dataset Classification Accuracy', fontsize=11)
ax1.set_title('A. Dataset Separability by Layer\n(Higher = more dataset-specific)', fontsize=12)
ax1.set_ylim(0, 1.05)
ax1.legend(fontsize=9, loc='lower left')

for i, (acc, std) in enumerate(zip(accs, stds)):
    ax1.text(i, acc + std + 0.02, f'{acc:.3f}', ha='center', fontsize=9, fontweight='bold')

# Highlight the peak
peak_idx = np.argmax(accs)
ax1.annotate('Peak: Early features\nmost dataset-specific',
            xy=(peak_idx, accs[peak_idx]), xytext=(peak_idx + 0.5, 0.95),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=9, color='red', fontweight='bold')

# Panel B: A-distance trends for each pair
ax2 = fig.add_subplot(gs[1])
pair_colors = {
    '0617final vs XBD': '#e74c3c',      # red - both damage
    '0617final vs SECOND': '#3498db',   # blue - war vs urban
    '0617final vs LEVIR-CD': '#2ecc71', # green - war vs building
    'XBD vs SECOND': '#9b59b6',         # purple - disaster vs urban
    'XBD vs LEVIR-CD': '#f39c12',       # orange - disaster vs building
    'SECOND vs LEVIR-CD': '#1abc9c',    # teal - urban vs building
}
pair_labels = {
    '0617final vs XBD': 'War vs Disaster (both damage)',
    '0617final vs SECOND': 'War vs Urban',
    '0617final vs LEVIR-CD': 'War vs Building',
    'XBD vs SECOND': 'Disaster vs Urban',
    'XBD vs LEVIR-CD': 'Disaster vs Building',
    'SECOND vs LEVIR-CD': 'Urban vs Building',
}

for pair in pairs:
    a_dists = [results[stage][pair]['a_distance'] for stage in stages]
    ax2.plot(range(len(stages)), a_dists, 'o-', color=pair_colors[pair],
             linewidth=2, markersize=8, label=pair_labels[pair], alpha=0.8)

ax2.set_xticks(range(len(stages)))
ax2.set_xticklabels(stages_short, fontsize=11)
ax2.set_ylabel('A-distance', fontsize=11)
ax2.set_title('B. Domain Gap (A-distance) Across Layers\n(Higher = more different)', fontsize=12)
ax2.legend(fontsize=8, loc='center right', bbox_to_anchor=(1.0, 0.5))
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.5, 2.1)

plt.suptitle('Cross-Dataset Domain Gap Analysis (DINOv3 Backbone)', fontsize=14, y=1.02)
plt.tight_layout()
path1 = os.path.join(output_dir, 'domain_gap_overview.png')
plt.savefig(path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {path1}")

# ============================================================
# Figure 2: Normalized A-distance heatmap (shows relative differences)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Compute global min/max for consistent color scale
all_a_dists = []
for stage in stages:
    for pair in pairs:
        all_a_dists.append(results[stage][pair]['a_distance'])
vmin, vmax = min(all_a_dists), max(all_a_dists)

for idx, stage in enumerate(stages):
    ax = axes[idx]
    n = len(dataset_names)
    dist_matrix = np.zeros((n, n))

    for i, name_a in enumerate(dataset_names):
        for j, name_b in enumerate(dataset_names):
            if i == j:
                dist_matrix[i, j] = vmin  # diagonal = min for visualization
            else:
                key = f"{name_a} vs {name_b}" if i < j else f"{name_b} vs {name_a}"
                if key in results[stage]:
                    dist_matrix[i, j] = results[stage][key]['a_distance']

    im = ax.imshow(dist_matrix, cmap='YlOrRd', vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(dataset_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(dataset_names, fontsize=8)
    ax.set_title(f'{stage}\nAcc={separability[stage]["acc"]:.3f}', fontsize=10, fontweight='bold')

    for i in range(n):
        for j in range(n):
            if i != j:
                color = 'white' if dist_matrix[i, j] > (vmin + vmax) / 2 else 'black'
                ax.text(j, i, f'{dist_matrix[i,j]:.2f}',
                       ha='center', va='center', fontsize=8, color=color)

plt.colorbar(im, ax=axes, shrink=0.6, label='A-distance')
plt.suptitle('A-distance Heatmap Across Stages (Red = more different)', fontsize=13)
plt.tight_layout()
path2 = os.path.join(output_dir, 'adistance_heatmap.png')
plt.savefig(path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {path2}")

# ============================================================
# Figure 3: Key insight - what changes with depth
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 7))

# Group pairs by type
damage_pairs = ['0617final vs XBD']  # both damage
change_pairs = ['SECOND vs LEVIR-CD']  # both change
cross_pairs = ['0617final vs SECOND', '0617final vs LEVIR-CD',
               'XBD vs SECOND', 'XBD vs LEVIR-CD']  # damage vs change

# Compute mean A-distance for each group
damage_means = []
change_means = []
cross_means = []

for stage in stages:
    damage_means.append(np.mean([results[stage][p]['a_distance'] for p in damage_pairs]))
    change_means.append(np.mean([results[stage][p]['a_distance'] for p in change_pairs]))
    cross_means.append(np.mean([results[stage][p]['a_distance'] for p in cross_pairs]))

x = range(len(stages))
ax.plot(x, damage_means, 'o-', color='#e74c3c', linewidth=3, markersize=10,
        label='Same type: Damage (War↔Disaster)', alpha=0.8)
ax.plot(x, change_means, 's-', color='#3498db', linewidth=3, markersize=10,
        label='Same type: Change (Urban↔Building)', alpha=0.8)
ax.plot(x, cross_means, '^-', color='#2ecc71', linewidth=3, markersize=10,
        label='Cross type: Damage↔Change', alpha=0.8)

ax.fill_between(x, damage_means, alpha=0.1, color='#e74c3c')
ax.fill_between(x, change_means, alpha=0.1, color='#3498db')
ax.fill_between(x, cross_means, alpha=0.1, color='#2ecc71')

ax.set_xticks(x)
ax.set_xticklabels(stages_short, fontsize=11)
ax.set_ylabel('Mean A-distance', fontsize=12)
ax.set_title('Domain Gap by Change Type Across Network Depth\n'
             'Key question: Does "same change type" reduce domain gap?', fontsize=13)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)

# Annotate key findings
ax.annotate('Early layers: all pairs similar\n(high A-distance everywhere)',
            xy=(1, max(damage_means[1], change_means[1], cross_means[1])),
            xytext=(1.5, 2.0),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=9, color='gray')

ax.annotate('Late layers: gap narrows\nbut pattern unclear',
            xy=(5, cross_means[5]),
            xytext=(4, 1.3),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=9, color='gray')

plt.tight_layout()
path3 = os.path.join(output_dir, 'change_type_analysis.png')
plt.savefig(path3, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {path3}")

# ============================================================
# Print summary
# ============================================================
print("\n" + "=" * 60)
print("Key Findings Summary")
print("=" * 60)

print("\n1. Dataset Separability Pattern:")
print(f"   Peak at S1_early ({separability['S1_early']['acc']:.3f})")
print(f"   Lowest at S5_final ({separability['S5_final']['acc']:.3f})")
print(f"   → OPPOSITE of previous LWGANet finding!")
print(f"   → DINOv3 early features are MORE dataset-specific")

print("\n2. A-distance at Final Layer (S5):")
for pair in pairs:
    a_dist = results['S5_final'][pair]['a_distance']
    print(f"   {pair}: {a_dist:.3f}")

print("\n3. Change Type Analysis (S5):")
print(f"   Same type (Damage): {damage_means[5]:.3f}")
print(f"   Same type (Change): {change_means[5]:.3f}")
print(f"   Cross type: {cross_means[5]:.3f}")
print(f"   → Damage pair closer than Cross: {damage_means[5] < cross_means[5]}")
print(f"   → Change pair closer than Cross: {change_means[5] < cross_means[5]}")

print("\n4. Implications for Transfer Learning:")
print("   - Freeze low / tune high strategy is WRONG for DINOv3")
print("   - Early features need MORE adaptation, not less")
print("   - High-level features are more transferable")
print("   - This explains why freeze/unfreeze experiments showed no benefit")
