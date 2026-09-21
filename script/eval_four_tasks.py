"""
Four-task evaluation on test set.

Tasks:
  T1 Building damage: Building
  T2 Surface cover change: Farmland, Runway, Taxiway, Apron, Highway, Bridge
  T3 Mobile target change: Aircraft, Vessel
  T4 Facility change: Crater, Tank, Pier, Dock, Shelter, Tower

Each task computes: Precision, Recall, F1, IoU, SeK independently.
"""

import os
import sys
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from models import build_hicd_v7
from datasets import create_dataloader


# Task definitions
TASKS = {
    'T1_Building': {
        'name': '建筑损毁',
        'classes': [6],  # Building
        'class_names': ['Building'],
    },
    'T2_Surface': {
        'name': '地表覆盖变化',
        'classes': [0, 1, 2, 3, 4, 5],  # Farmland, Runway, Taxiway, Apron, Bridge, Highway
        'class_names': ['Farmland', 'Runway', 'Taxiway', 'Apron', 'Bridge', 'Highway'],
    },
    'T3_Mobile': {
        'name': '移动目标变化',
        'classes': [12, 13],  # Aircraft, Vessel
        'class_names': ['Aircraft', 'Vessel'],
    },
    'T4_Facility': {
        'name': '设施变化',
        'classes': [7, 8, 9, 10, 11, 14],  # Shelter, Tower, Pier, Dock, Tank, Crater
        'class_names': ['Shelter', 'Tower', 'Pier', 'Dock', 'Tank', 'Crater'],
    },
}

TARGET_NAMES = [
    'Farmland', 'Runway', 'Taxiway', 'Apron', 'Bridge', 'Highway',
    'Building', 'Shelter', 'Tower', 'Pier', 'Dock', 'Tank',
    'Aircraft', 'Vessel', 'Crater'
]

STATE_NAMES = ['no-change', 'damaged', 'reduced', 'added', 'extended', 'replaced', 'none']


def compute_sek(pred_changed, gt_changed):
    """Compute Separated Kappa (SeK)."""
    # Changed pixels
    gt_c = gt_changed
    pred_c = pred_changed

    # Confusion matrix for changed
    tp_c = np.logical_and(pred_c, gt_c).sum()
    fp_c = np.logical_and(pred_c, ~gt_c).sum()
    fn_c = np.logical_and(~pred_c, gt_c).sum()
    tn_c = np.logical_and(~pred_c, ~gt_c).sum()

    n = tp_c + fp_c + fn_c + tn_c
    if n == 0:
        return 0.0, 0.0, 0.0

    # Kappa for changed
    po_c = (tp_c + tn_c) / n
    pe_c = ((tp_c + fp_c) * (tp_c + fn_c) + (fn_c + tn_c) * (fp_c + tn_c)) / (n * n)
    kappa_c = (po_c - pe_c) / (1 - pe_c + 1e-8)

    # For unchanged (invert)
    po_u = po_c
    pe_u = ((tn_c + fn_c) * (tn_c + fp_c) + (fp_c + tp_c) * (fn_c + tp_c)) / (n * n)
    kappa_u = (po_u - pe_u) / (1 - pe_u + 1e-8)

    sek = (kappa_c + kappa_u) / 2
    return sek, kappa_c, kappa_u


def evaluate_task(pred_target, pred_state, gt_target, gt_state, task_classes):
    """
    Evaluate a single task.

    Args:
        pred_target: [N] predicted target classes
        pred_state: [N] predicted state classes
        gt_target: [N] ground truth target classes
        gt_state: [N] ground truth state classes
        task_classes: list of class IDs for this task

    Returns:
        metrics dict
    """
    # Mask: only pixels belonging to this task's classes
    task_mask = np.isin(gt_target, task_classes)

    if task_mask.sum() == 0:
        return None

    pred_t = pred_target[task_mask]
    pred_s = pred_state[task_mask]
    gt_t = gt_target[task_mask]
    gt_s = gt_state[task_mask]

    # Per-class IoU
    class_ious = {}
    for cls in task_classes:
        cls_name = TARGET_NAMES[cls]
        gt_mask = (gt_t == cls)
        pred_mask = (pred_t == cls)

        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()

        if union > 0:
            class_ious[cls_name] = intersection / union
        else:
            class_ious[cls_name] = 0.0

    # Mean IoU for this task
    miou = np.mean(list(class_ious.values()))

    # Change detection metrics (changed vs unchanged)
    # Changed = state is not no-change (0) and not none (6)
    gt_changed = (gt_s != 0) & (gt_s != 6)
    pred_changed = (pred_s != 0) & (pred_s != 6)

    tp = np.logical_and(pred_changed, gt_changed).sum()
    fp = np.logical_and(pred_changed, ~gt_changed).sum()
    fn = np.logical_and(~pred_changed, gt_changed).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # SeK
    sek, kappa_c, kappa_u = compute_sek(pred_changed, gt_changed)

    # State IoU (per state class, only for changed pixels)
    state_ious = {}
    for s in range(7):
        gt_mask = (gt_s == s)
        pred_mask = (pred_s == s)
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        if union > 0:
            state_ious[STATE_NAMES[s]] = intersection / union

    return {
        'class_ious': class_ious,
        'miou': miou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sek': sek,
        'kappa_changed': kappa_c,
        'kappa_unchanged': kappa_u,
        'state_ious': state_ious,
        'n_pixels': int(task_mask.sum()),
    }


def main():
    # Config
    ckpt_path = 'checkpoints/v7_synth_dense/best_model.pth'
    config_path = 'datasets/configs/0617final.yaml'
    data_dir = 'D:/CD/0617final_256_v2'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)

    model_config = dict(dataset_config)
    model_config['backbone'] = {
        'type': 'dinov3',
        'variant': 'dinov3_base',
        'pretrained': True,
        'pretrained_path': 'weights/model.safetensors',
    }
    model_config['hidden_dim'] = 128
    model_config['embed_dim'] = 64

    # Build model
    print("Building model...")
    model = build_hicd_v7(model_config).to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    model.eval()
    print(f"Loaded epoch: {ckpt.get('epoch', '?')}")

    # Load test data
    print("Loading test data...")
    test_loader = create_dataloader(
        data_dir=data_dir,
        config=dataset_config,
        split='test',
        batch_size=4,
        num_workers=0,
        img_size=256,
        pin_memory=False,
    )
    print(f"Test samples: {len(test_loader.dataset)}")

    # Accumulate predictions
    all_pred_target = []
    all_pred_state = []
    all_gt_target = []
    all_gt_state = []

    print("\nRunning inference...")
    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            if (bi + 1) % 100 == 0:
                print(f"  Batch {bi+1}/{len(test_loader)}")

            img_t1 = batch['img_t1'].to(device)
            img_t2 = batch['img_t2'].to(device)

            # Prepare targets
            targets = {}
            if 'label' in batch:
                label = batch['label'].to(device)
                if label.dim() == 3:
                    label_s = label.unsqueeze(1).float()
                    label_s = F.interpolate(label_s, scale_factor=0.25, mode='nearest').squeeze(1).long()
                    targets['target_gt'] = label_s
            if 'change_state' in batch:
                cs = batch['change_state'].to(device)
                if cs.dim() == 3:
                    cs_s = cs.unsqueeze(1).float()
                    cs_s = F.interpolate(cs_s, scale_factor=0.25, mode='nearest').squeeze(1).long()
                    targets['change_state'] = cs_s
                    targets['state_gt'] = cs_s

            # Forward
            predictions, loss, loss_dict = model(img_t1, img_t2, targets)

            # Get predictions (at H/4 resolution)
            pred_target = predictions['target_logits'].argmax(dim=1).cpu().numpy()
            pred_state = predictions['state_logits'].argmax(dim=1).cpu().numpy()

            # Downsample GT to match prediction resolution
            gt_target = batch['label'].numpy()
            gt_state = batch['change_state'].numpy()

            # Resize GT from 256x256 to 64x64
            gt_target_ds = []
            gt_state_ds = []
            for b in range(gt_target.shape[0]):
                gt_t = torch.from_numpy(gt_target[b]).unsqueeze(0).unsqueeze(0).float()
                gt_s = torch.from_numpy(gt_state[b]).unsqueeze(0).unsqueeze(0).float()
                gt_t = F.interpolate(gt_t, size=pred_target.shape[1:], mode='nearest').squeeze().numpy()
                gt_s = F.interpolate(gt_s, size=pred_state.shape[1:], mode='nearest').squeeze().numpy()
                gt_target_ds.append(gt_t.flatten())
                gt_state_ds.append(gt_s.flatten())

            # Flatten and append
            for b in range(pred_target.shape[0]):
                all_pred_target.append(pred_target[b].flatten())
                all_pred_state.append(pred_state[b].flatten())
                all_gt_target.append(gt_target_ds[b])
                all_gt_state.append(gt_state_ds[b])

    # Concatenate all
    pred_target_all = np.concatenate(all_pred_target)
    pred_state_all = np.concatenate(all_pred_state)
    gt_target_all = np.concatenate(all_gt_target)
    gt_state_all = np.concatenate(all_gt_state)

    print(f"\nTotal pixels: {len(pred_target_all):,}")

    # Evaluate each task
    print("\n" + "=" * 80)
    print("Four-Task Evaluation Results")
    print("=" * 80)

    results = {}
    for task_id, task_info in TASKS.items():
        print(f"\n{'─' * 60}")
        print(f"Task: {task_id} - {task_info['name']}")
        print(f"Classes: {task_info['class_names']}")
        print(f"{'─' * 60}")

        result = evaluate_task(
            pred_target_all, pred_state_all,
            gt_target_all, gt_state_all,
            task_info['classes']
        )

        if result is None:
            print("  No pixels found for this task!")
            continue

        results[task_id] = result

        print(f"  Pixels: {result['n_pixels']:,}")
        print(f"\n  Change Detection:")
        print(f"    Precision: {result['precision']:.4f}")
        print(f"    Recall:    {result['recall']:.4f}")
        print(f"    F1:        {result['f1']:.4f}")
        print(f"\n  SeK: {result['sek']:.4f}")
        print(f"    Kappa (changed):   {result['kappa_changed']:.4f}")
        print(f"    Kappa (unchanged): {result['kappa_unchanged']:.4f}")
        print(f"\n  mIoU: {result['miou']:.4f}")
        print(f"\n  Per-class IoU:")
        for cls_name, iou in sorted(result['class_ious'].items(), key=lambda x: -x[1]):
            print(f"    {cls_name:<12}: {iou:.4f}")

    # Summary table
    print("\n" + "=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(f"{'Task':<20} {'Prec':>8} {'Recall':>8} {'F1':>8} {'IoU':>8} {'SeK':>8}")
    print("-" * 60)
    for task_id, result in results.items():
        task_name = TASKS[task_id]['name']
        print(f"{task_name:<20} {result['precision']:>8.4f} {result['recall']:>8.4f} "
              f"{result['f1']:>8.4f} {result['miou']:>8.4f} {result['sek']:>8.4f}")

    # Save results
    import json
    output_path = 'checkpoints/v7_synth_dense/four_task_eval.json'
    save_results = {}
    for task_id, result in results.items():
        save_results[task_id] = {
            'name': TASKS[task_id]['name'],
            'classes': TASKS[task_id]['class_names'],
            'n_pixels': result['n_pixels'],
            'precision': float(result['precision']),
            'recall': float(result['recall']),
            'f1': float(result['f1']),
            'miou': float(result['miou']),
            'sek': float(result['sek']),
            'kappa_changed': float(result['kappa_changed']),
            'kappa_unchanged': float(result['kappa_unchanged']),
            'class_ious': {k: float(v) for k, v in result['class_ious'].items()},
            'state_ious': {k: float(v) for k, v in result['state_ious'].items()},
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
