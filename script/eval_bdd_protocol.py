# -*- coding: utf-8 -*-
"""
Convert our predictions to WD-CD paper's BDD protocol and compute comparable metrics.

Paper protocol:
  ND (no-damage) = no-change, added, extended
  D (damaged) = damaged, reduced
  
  F_loc = building vs background F1 (merge ND+D into single building class)
  F_clf = harmonic mean of per-class F1 of ND and D (only on true building pixels)
  F_overall = 0.3 * F_loc + 0.7 * F_clf
"""

import os
import sys
import yaml
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from models import build_hicd_v7
from datasets import create_dataloader


def compute_f1(tp, fp, fn):
    """Compute F1 score."""
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return f1, precision, recall


def evaluate_bdd(pred_target, pred_state, gt_target, gt_state):
    """
    Evaluate using WD-CD paper's BDD protocol.
    
    Only considers Building class (target_id = 6).
    """
    # GT building mask (all building pixels, regardless of state)
    gt_building = (gt_target == 6)
    
    if gt_building.sum() == 0:
        return None
    
    # Valid states for F_clf (exclude replaced=5 and none=6)
    valid_state_mask = gt_building & (gt_state != 5) & (gt_state != 6)
    
    # F_loc: building vs background (on ALL pixels)
    pred_building = (pred_target == 6)
    
    tp_loc = np.logical_and(pred_building, gt_building).sum()
    fp_loc = np.logical_and(pred_building, ~gt_building).sum()
    fn_loc = np.logical_and(~pred_building, gt_building).sum()
    f1_loc, prec_loc, rec_loc = compute_f1(tp_loc, fp_loc, fn_loc)
    
    # F_clf: ND vs D (only on GT building pixels with valid states)
    gt_nd_mask = valid_state_mask & np.isin(gt_state, [0, 3, 4])  # no-change, added, extended
    gt_d_mask = valid_state_mask & np.isin(gt_state, [1, 2])       # damaged, reduced
    
    pred_nd_mask = pred_building & np.isin(pred_state, [0, 3, 4])
    pred_d_mask = pred_building & np.isin(pred_state, [1, 2])
    
    # ND F1
    tp_nd = np.logical_and(pred_nd_mask, gt_nd_mask).sum()
    fp_nd = np.logical_and(pred_nd_mask, gt_d_mask).sum()
    fn_nd = np.logical_and(~pred_nd_mask, gt_nd_mask).sum()
    f1_nd, prec_nd, rec_nd = compute_f1(tp_nd, fp_nd, fn_nd)
    
    # D F1
    tp_d = np.logical_and(pred_d_mask, gt_d_mask).sum()
    fp_d = np.logical_and(pred_d_mask, gt_nd_mask).sum()
    fn_d = np.logical_and(~pred_d_mask, gt_d_mask).sum()
    f1_d, prec_d, rec_d = compute_f1(tp_d, fp_d, fn_d)
    
    # F_clf = harmonic mean of ND and D F1
    f1_clf = 2 * f1_nd * f1_d / max(f1_nd + f1_d, 1e-8)
    
    # F_overall = 0.3 * F_loc + 0.7 * F_clf
    f1_overall = 0.3 * f1_loc + 0.7 * f1_clf
    
    # F_macro = arithmetic mean of ND and D F1
    f1_macro = (f1_nd + f1_d) / 2
    
    return {
        'F_loc': f1_loc,
        'F_clf': f1_clf,
        'F_overall': f1_overall,
        'F_macro': f1_macro,
        'F_ND': f1_nd,
        'F_D': f1_d,
        'prec_loc': prec_loc,
        'rec_loc': rec_loc,
        'prec_ND': prec_nd,
        'rec_ND': rec_nd,
        'prec_D': prec_d,
        'rec_D': rec_d,
    }


def main():
    ckpt_path = 'checkpoints/v7_synth_dense/best_model.pth'
    config_path = 'datasets/configs/0617final.yaml'
    data_dir = 'D:/CD/0617final_256_v2'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load config and model
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
    print(f"Loaded epoch: {ckpt.get('epoch', '?')}")

    # Load test data
    print("Loading test data...")
    test_loader = create_dataloader(
        data_dir=data_dir, config=dataset_config,
        split='test', batch_size=4, num_workers=0,
        img_size=256, pin_memory=False,
    )

    # Run inference
    all_pred_target = []
    all_pred_state = []
    all_gt_target = []
    all_gt_state = []

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
            if 'change_state' in batch:
                cs = batch['change_state'].to(device)
                if cs.dim() == 3:
                    cs_s = F.interpolate(cs.unsqueeze(1).float(), scale_factor=0.25, mode='nearest').squeeze(1).long()
                    targets['change_state'] = cs_s
                    targets['state_gt'] = cs_s

            predictions, _, _ = model(img_t1, img_t2, targets)

            pred_target = predictions['target_logits'].argmax(dim=1).cpu().numpy()
            pred_state = predictions['state_logits'].argmax(dim=1).cpu().numpy()

            # Downsample GT
            gt_target = batch['label'].numpy()
            gt_state = batch['change_state'].numpy()

            for b in range(pred_target.shape[0]):
                gt_t = torch.from_numpy(gt_target[b]).unsqueeze(0).unsqueeze(0).float()
                gt_s = torch.from_numpy(gt_state[b]).unsqueeze(0).unsqueeze(0).float()
                gt_t = F.interpolate(gt_t, size=pred_target.shape[1:], mode='nearest').squeeze().numpy()
                gt_s = F.interpolate(gt_s, size=pred_state.shape[1:], mode='nearest').squeeze().numpy()

                all_pred_target.append(pred_target[b].flatten())
                all_pred_state.append(pred_state[b].flatten())
                all_gt_target.append(gt_t.flatten())
                all_gt_state.append(gt_s.flatten())

    pred_target_all = np.concatenate(all_pred_target)
    pred_state_all = np.concatenate(all_pred_state)
    gt_target_all = np.concatenate(all_gt_target)
    gt_state_all = np.concatenate(all_gt_state)

    print(f"Total pixels: {len(pred_target_all):,}")

    # Evaluate BDD
    print("\n" + "=" * 70)
    print("BDD Evaluation (WD-CD Paper Protocol)")
    print("=" * 70)

    result = evaluate_bdd(pred_target_all, pred_state_all, gt_target_all, gt_state_all)

    if result:
        print(f"\n{'Metric':<20} {'Value':>10}")
        print("-" * 30)
        print(f"{'F_loc (localization)':<20} {result['F_loc']*100:>9.2f}%")
        print(f"{'F_clf (classification)':<20} {result['F_clf']*100:>9.2f}%")
        print(f"{'F_overall':<20} {result['F_overall']*100:>9.2f}%")
        print(f"{'F_macro':<20} {result['F_macro']*100:>9.2f}%")
        print(f"\n{'F_ND (no-damage)':<20} {result['F_ND']*100:>9.2f}%")
        print(f"{'F_D (damaged)':<20} {result['F_D']*100:>9.2f}%")

        print(f"\n{'Comparison with paper:'}")
        print(f"{'─' * 50}")
        print(f"{'Method':<25} {'F_overall':>10}")
        print(f"{'─' * 50}")
        print(f"{'DeepSiameseNet':<25} {'72.46%':>10}")
        print(f"{'CD-Lamba (best)':<25} {'79.53%':>10}")
        print(f"{'Ours (HICD v7)':<25} {result['F_overall']*100:>9.2f}%")
    else:
        print("No building pixels found!")

    # Save results
    import json
    output_path = 'checkpoints/v7_synth_dense/bdd_eval.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({k: float(v) for k, v in result.items()}, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    main()
