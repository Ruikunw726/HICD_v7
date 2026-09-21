# -*- coding: utf-8 -*-
"""
Compute confusion matrix for target classification.
Focus on understanding Building false positives.
"""
import os
import sys
import yaml
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import build_hicd_v7
from datasets import create_dataloader

TARGET_NAMES = [
    'Farmland', 'Runway', 'Taxiway', 'Apron', 'Bridge', 'Highway',
    'Building', 'Shelter', 'Tower', 'Pier', 'Dock', 'Tank',
    'Aircraft', 'Vessel', 'Crater'
]


def main():
    ckpt_path = 'checkpoints/v7_synth_dense/best_model.pth'
    config_path = 'datasets/configs/0617final.yaml'
    data_dir = 'D:/CD/0617final_256_v2'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(config_path, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)

    model_config = dict(dataset_config)
    model_config['backbone'] = {
        'type': 'dinov3', 'variant': 'dinov3_base',
        'pretrained': True, 'pretrained_path': 'weights/model.safetensors',
    }
    model_config['hidden_dim'] = 128
    model_config['embed_dim'] = 64

    print("Building model...")
    model = build_hicd_v7(model_config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    model.eval()

    print("Loading test data...")
    test_loader = create_dataloader(
        data_dir=data_dir, config=dataset_config,
        split='test', batch_size=8, num_workers=0,
        img_size=256, pin_memory=False,
    )

    # Confusion matrix: 15 classes + background (255)
    num_classes = 16  # 0-14 + 255->15
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    print("Running inference...")
    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            if (bi + 1) % 500 == 0:
                print(f"  {bi+1}/{len(test_loader)}")

            img_t1 = batch['img_t1'].to(device)
            img_t2 = batch['img_t2'].to(device)

            targets = {}
            if 'label' in batch:
                label = batch['label'].to(device)
                if label.dim() == 3:
                    label_s = F.interpolate(label.unsqueeze(1).float(), scale_factor=0.25, mode='nearest').squeeze(1).long()
                    targets['target_gt'] = label_s

            predictions, _, _ = model(img_t1, img_t2, targets)

            pred = predictions['target_logits'].argmax(dim=1).cpu().numpy()
            gt = batch['label'].numpy()

            # Downsample GT
            for b in range(pred.shape[0]):
                gt_t = torch.from_numpy(gt[b]).unsqueeze(0).unsqueeze(0).float()
                gt_t = F.interpolate(gt_t, size=pred.shape[1:], mode='nearest').squeeze().numpy().astype(np.int64)

                pred_b = pred[b].flatten()
                gt_b = gt_t.flatten()

                # Map 255 -> 15 (background)
                gt_b[gt_b == 255] = 15
                pred_b[pred_b == 255] = 15

                # Accumulate confusion
                for p, g in zip(pred_b, gt_b):
                    if 0 <= g < num_classes and 0 <= p < num_classes:
                        confusion[g, p] += 1

    # Print confusion matrix
    print("\n" + "=" * 80)
    print("Confusion Matrix (rows=GT, cols=Pred)")
    print("=" * 80)

    # Header
    header = f"{'':>12}"
    for i in range(15):
        header += f"{TARGET_NAMES[i][:6]:>8}"
    header += f"{'BG':>8}"
    print(header)

    for i in range(num_classes):
        name = TARGET_NAMES[i] if i < 15 else 'Background'
        row = f"{name:>12}"
        for j in range(num_classes):
            row += f"{confusion[i, j]:>8}"
        print(row)

    # Focus on Building (class 6)
    print("\n" + "=" * 80)
    print("Building Analysis (GT=Building)")
    print("=" * 80)

    building_row = confusion[6, :]
    building_total = building_row.sum()
    print(f"Total GT Building pixels: {building_total:,}")
    print(f"\nPredicted as:")
    for j in range(num_classes):
        if building_row[j] > 0:
            name = TARGET_NAMES[j] if j < 15 else 'Background'
            pct = 100.0 * building_row[j] / building_total
            print(f"  {name:<12}: {building_row[j]:>10,} ({pct:.1f}%)")

    # What is being predicted as Building
    print("\n" + "=" * 80)
    print("Building False Positives (Pred=Building, GT≠Building)")
    print("=" * 80)

    building_col = confusion[:, 6]
    fp_total = building_col.sum() - confusion[6, 6]
    print(f"Total FP pixels: {fp_total:,}")
    print(f"\nActually is:")
    for i in range(num_classes):
        if i == 6:
            continue
        if building_col[i] > 0:
            name = TARGET_NAMES[i] if i < 15 else 'Background'
            pct = 100.0 * building_col[i] / max(fp_total, 1)
            print(f"  {name:<12}: {building_col[i]:>10,} ({pct:.1f}%)")

    # Save confusion matrix as image
    fig, ax = plt.subplots(figsize=(12, 10))
    # Normalize by row (GT)
    confusion_norm = confusion.astype(np.float64)
    row_sums = confusion_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    confusion_norm = confusion_norm / row_sums

    im = ax.imshow(confusion_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    labels = TARGET_NAMES + ['BG']
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')
    ax.set_title('Target Classification Confusion Matrix (normalized by GT)')

    # Annotate Building row and column
    for j in range(num_classes):
        if confusion_norm[6, j] > 0.01:
            ax.text(j, 6, f'{confusion_norm[6, j]:.2f}', ha='center', va='center',
                   fontsize=7, color='red', fontweight='bold')

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    output_path = 'checkpoints/v7_synth_dense/confusion_matrix.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")

    # Save raw confusion matrix
    np.save('checkpoints/v7_synth_dense/confusion_matrix.npy', confusion)
    print(f"Saved: checkpoints/v7_synth_dense/confusion_matrix.npy")


if __name__ == '__main__':
    main()
