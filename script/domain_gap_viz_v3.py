"""
Domain Gap Visualization v3: 500 samples per dataset results
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Results from 500-sample run
results = {
    'S0_input': {
        '0617final vs XBD': {'a_distance': 0.772},
        '0617final vs SECOND': {'a_distance': 1.448},
        '0617final vs LEVIR-CD': {'a_distance': 1.230},
        'XBD vs SECOND': {'a_distance': 1.794},
        'XBD vs LEVIR-CD': {'a_distance': 1.554},
        'SECOND vs LEVIR-CD': {'a_distance': 1.826},
    },
    'S1_early': {
        '0617final vs XBD': {'a_distance': 1.796},
        '0617final vs SECOND': {'a_distance': 1.960},
        '0617final vs LEVIR-CD': {'a_distance': 1.882},
        'XBD vs SECOND': {'a_distance': 1.982},
        'XBD vs LEVIR-CD': {'a_distance': 1.984},
        'SECOND vs LEVIR-CD': {'a_distance': 1.994},
    },
    'S2_mid': {
        '0617final vs XBD': {'a_distance': 1.808},
        '0617final vs SECOND': {'a_distance': 1.950},
        '0617final vs LEVIR-CD': {'a_distance': 1.886},
        'XBD vs SECOND': {'a_distance': 1.994},
        'XBD vs LEVIR-CD': {'a_distance': 1.992},
        'SECOND vs LEVIR-CD': {'a_distance': 1.998},
    },
    'S3_late': {
        '0617final vs XBD': {'a_distance': 1.670},
        '0617final vs SECOND': {'a_distance': 1.934},
        '0617final vs LEVIR-CD': {'a_distance': 1.770},
        'XBD vs SECOND': {'a_distance': 1.976},
        'XBD vs LEVIR-CD': {'a_distance': 1.956},
        'SECOND vs LEVIR-CD': {'a_distance': 1.970},
    },
    'S4_deep': {
        '0617final vs XBD': {'a_distance': 1.606},
        '0617final vs SECOND': {'a_distance': 1.896},
        '0617final vs LEVIR-CD': {'a_distance': 1.668},
        'XBD vs SECOND': {'a_distance': 1.976},
        'XBD vs LEVIR-CD': {'a_distance': 1.894},
        'SECOND vs LEVIR-CD': {'a_distance': 1.970},
    },
    'S5_final': {
        '0617final vs XBD': {'a_distance': 1.608},
        '0617final vs SECOND': {'a_distance': 1.908},
        '0617final vs LEVIR-CD': {'a_distance': 1.704},
        'XBD vs SECOND': {'a_distance': 1.976},
        'XBD vs LEVIR-CD': {'a_distance': 1.874},
        'SECOND vs LEVIR-CD': {'a_distance': 1.974},
    },
}

separability = {
    'S0_input': {'acc': 0.761, 'std': 0.052},
    'S1_early': {'acc': 0.932, 'std': 0.038},
    'S2_mid': {'acc': 0.874, 'std': 0.051},
    'S3_late': {'acc': 0.783, 'std': 0.070},
    'S4_deep': {'acc': 0.780, 'std': 0.062},
    'S5_final': {'acc': 0.778, 'std': 0.057},
}

stages = list(results.keys())
stages_short = ['S0\ninput', 'S1\nearly', 'S2\nmid', 'S3\nlate', 'S4\ndeep', 'S5\nfinal']
pairs = list(results['S0_input'].keys())
output_dir = 'F:/mambacd/working_notes/domain_gap_v2_large'

# ============================================================
# Figure 1: Main findings overview
# ============================================================
fig = plt.figure(figsize=(16, 6))
gs = GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.3)

# Panel A: Dataset separability
ax1 = fig.add_subplot(gs[0])
accs = [separability[s]['acc'] for s in stages]
stds = [separability[s]['std'] for s in stages]
colors = ['#e74c3c' if a > 0.85 else '#f39c12' if a > 0.8 else '#3498db' for a in accs]

bars = ax1.bar(range(len(stages)), accs, yerr=stds, capsize=5,
               color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
ax1.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Random (25%)')
ax1.axhline(y=np.mean(accs), color='green', linestyle=':', alpha=0.7, label=f'Mean ({np.mean(accs):.3f})')

ax1.set_xticks(range(len(stages)))
ax1.set_xticklabels(stages_short, fontsize=10)
ax1.set_ylabel('Dataset Classification Accuracy', fontsize=11)
ax1.set_title('A. Dataset Separability (n=500/dataset)\n'
              'Red = most dataset-specific, Blue = most transferable', fontsize=11)
ax1.set_ylim(0, 1.05)
ax1.legend(fontsize=9, loc='lower left')

for i, (acc, std) in enumerate(zip(accs, stds)):
    ax1.text(i, acc + std + 0.02, f'{acc:.3f}', ha='center', fontsize=9, fontweight='bold')

# Highlight peak and trough
peak_idx = np.argmax(accs)
trough_idx = np.argmin(accs)
ax1.annotate('Peak: 93.2%\nEarly = most specific',
            xy=(peak_idx, accs[peak_idx]), xytext=(peak_idx - 0.3, 0.55),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=9, color='red', fontweight='bold')
ax1.annotate('Final: 77.8%\nDeep = most transferable',
            xy=(trough_idx, accs[trough_idx]), xytext=(trough_idx - 2, 0.45),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
            fontsize=9, color='blue', fontweight='bold')

# Panel B: A-distance trends
ax2 = fig.add_subplot(gs[1])
pair_colors = {
    '0617final vs XBD': '#e74c3c',
    '0617final vs SECOND': '#3498db',
    '0617final vs LEVIR-CD': '#2ecc71',
    'XBD vs SECOND': '#9b59b6',
    'XBD vs LEVIR-CD': '#f39c12',
    'SECOND vs LEVIR-CD': '#1abc9c',
}
pair_labels = {
    '0617final vs XBD': 'War ↔ Disaster (both damage)',
    '0617final vs SECOND': 'War ↔ Urban',
    '0617final vs LEVIR-CD': 'War ↔ Building',
    'XBD vs SECOND': 'Disaster ↔ Urban',
    'XBD vs LEVIR-CD': 'Disaster ↔ Building',
    'SECOND vs LEVIR-CD': 'Urban ↔ Building',
}

for pair in pairs:
    a_dists = [results[stage][pair]['a_distance'] for stage in stages]
    ax2.plot(range(len(stages)), a_dists, 'o-', color=pair_colors[pair],
             linewidth=2, markersize=8, label=pair_labels[pair], alpha=0.8)

ax2.set_xticks(range(len(stages)))
ax2.set_xticklabels(stages_short, fontsize=10)
ax2.set_ylabel('A-distance', fontsize=11)
ax2.set_title('B. Domain Gap (A-distance) Across Layers\n'
              'Lower = more similar, Higher = more different', fontsize=11)
ax2.legend(fontsize=8, loc='center right', bbox_to_anchor=(1.0, 0.5))
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.5, 2.1)

# Highlight War↔Disaster as closest
war_disaster = [results[stage]['0617final vs XBD']['a_distance'] for stage in stages]
ax2.fill_between(range(len(stages)), war_disaster, alpha=0.15, color='#e74c3c')
ax2.annotate('War↔Disaster closest\nat all depths',
            xy=(5, war_disaster[5]), xytext=(3.5, 0.8),
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5),
            fontsize=9, color='#e74c3c', fontweight='bold')

plt.suptitle('Cross-Dataset Domain Gap Analysis (DINOv3, n=500/dataset)', fontsize=14, y=1.02)
plt.tight_layout()
path1 = os.path.join(output_dir, 'domain_gap_main_findings.png')
plt.savefig(path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {path1}")

# ============================================================
# Figure 2: Change type analysis
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

damage_pairs = ['0617final vs XBD']
change_pairs = ['SECOND vs LEVIR-CD']
cross_pairs = ['0617final vs SECOND', '0617final vs LEVIR-CD',
               'XBD vs SECOND', 'XBD vs LEVIR-CD']

damage_means = [np.mean([results[s][p]['a_distance'] for p in damage_pairs]) for s in stages]
change_means = [np.mean([results[s][p]['a_distance'] for p in change_pairs]) for s in stages]
cross_means = [np.mean([results[s][p]['a_distance'] for p in cross_pairs]) for s in stages]

x = range(len(stages))
ax.plot(x, damage_means, 'o-', color='#e74c3c', linewidth=3, markersize=10,
        label='Same: Damage (War↔Disaster)', alpha=0.8)
ax.plot(x, change_means, 's-', color='#3498db', linewidth=3, markersize=10,
        label='Same: Change (Urban↔Building)', alpha=0.8)
ax.plot(x, cross_means, '^-', color='#2ecc71', linewidth=3, markersize=10,
        label='Cross: Damage↔Change', alpha=0.8)

ax.fill_between(x, damage_means, alpha=0.1, color='#e74c3c')
ax.fill_between(x, change_means, alpha=0.1, color='#3498db')
ax.fill_between(x, cross_means, alpha=0.1, color='#2ecc71')

ax.set_xticks(x)
ax.set_xticklabels(stages_short, fontsize=10)
ax.set_ylabel('Mean A-distance', fontsize=12)
ax.set_title('Domain Gap by Change Type (n=500/dataset)\n'
             'Does "same change type" reduce domain gap?', fontsize=13)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)

# Annotate S5 values
ax.annotate(f'{damage_means[5]:.3f}', xy=(5, damage_means[5]),
            xytext=(5.1, damage_means[5]), fontsize=10, color='#e74c3c', fontweight='bold')
ax.annotate(f'{change_means[5]:.3f}', xy=(5, change_means[5]),
            xytext=(5.1, change_means[5]), fontsize=10, color='#3498db', fontweight='bold')
ax.annotate(f'{cross_means[5]:.3f}', xy=(5, cross_means[5]),
            xytext=(5.1, cross_means[5]), fontsize=10, color='#2ecc71', fontweight='bold')

# Key finding annotation
ax.annotate('Key finding: Only Damage pair is closer\nthan Cross-type. Change pair is FARTHEST!',
            xy=(0.5, 0.15), fontsize=10, color='darkred',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
path2 = os.path.join(output_dir, 'change_type_analysis_v2.png')
plt.savefig(path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {path2}")

# ============================================================
# Print final summary
# ============================================================
print("\n" + "=" * 60)
print("CONFIRMED FINDINGS (n=500/dataset)")
print("=" * 60)

print("\n1. Separability Pattern (CONFIRMED):")
print(f"   S0: {separability['S0_input']['acc']:.3f}")
print(f"   S1: {separability['S1_early']['acc']:.3f} ← PEAK (most dataset-specific)")
print(f"   S2: {separability['S2_mid']['acc']:.3f}")
print(f"   S3: {separability['S3_late']['acc']:.3f}")
print(f"   S4: {separability['S4_deep']['acc']:.3f}")
print(f"   S5: {separability['S5_final']['acc']:.3f} ← LOWEST (most transferable)")
print(f"\n   → OPPOSITE of LWGANet finding")
print(f"   → DINOv3 early features are MORE dataset-specific")

print("\n2. Change Type Analysis (S5):")
print(f"   Damage (War↔Disaster): {damage_means[5]:.3f} ← CLOSEST")
print(f"   Change (Urban↔Building): {change_means[5]:.3f} ← FARTHEST!")
print(f"   Cross-type mean: {cross_means[5]:.3f}")
print(f"\n   → 'Same change type' does NOT guarantee smaller gap")
print(f"   → Only Damage pair shows reduced gap")

print("\n3. Implications:")
print("   - Freeze-low/tune-high is WRONG for DINOv3")
print("   - S1-S2 layers need MOST adaptation")
print("   - S4-S5 layers are already transferable")
print("   - Explains why freeze/unfreeze experiments failed")
