"""
Inference on 0617final test set with majority voting refinement.
Computes metrics on test set and saves results.
"""
import os, sys, yaml, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_hicd_v7
from datasets import create_dataloader
from evaluation.icd_eval_v6 import ICDEvaluatorV6 as ICDEvaluator
import torch.nn as nn

# Config
with open('datasets/configs/0617final.yaml', 'r', encoding='utf-8') as f:
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_hicd_v7(model_config).to(device)

# Load best checkpoint
ckpt_path = 'checkpoints/v7_0617_freq_branch/checkpoint_epoch_50.pth'
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()
print(f"Loaded {ckpt_path}, epoch={ckpt.get('epoch', '?')}")

# Load test data
test_loader = create_dataloader(
    data_dir='/mnt/d/CD/0617final_256', config=dataset_config,
    split='test', batch_size=4, num_workers=2, img_size=256, pin_memory=False
)

# Initialize evaluator
evaluator = ICDEvaluator(dataset_config)

# Run inference
total_loss = 0
num_batches = 0

with torch.no_grad():
    for bi, batch in enumerate(test_loader):
        img_t1 = batch['img_t1'].to(device)
        img_t2 = batch['img_t2'].to(device)
        
        # Prepare targets (downsample 4x to match model output)
        targets = {}
        if 'label' in batch:
            label = batch['label'].to(device)
            if label.dim() == 3:
                label_s = label.unsqueeze(1).float()
                label_s = torch.nn.functional.interpolate(label_s, scale_factor=0.25, mode='nearest').squeeze(1).long()
                targets['target_gt'] = label_s
        if 'change_state' in batch:
            cs = batch['change_state'].to(device)
            if cs.dim() == 3:
                cs_s = cs.unsqueeze(1).float()
                cs_s = torch.nn.functional.interpolate(cs_s, scale_factor=0.25, mode='nearest').squeeze(1).long()
                targets['change_state'] = cs_s
                targets['state_gt'] = cs_s
        
        # Forward pass
        predictions, loss, loss_dict = model(img_t1, img_t2, targets)
        
        # Update evaluator
        evaluator.update(predictions, targets, loss_dict)
        
        # Compute F1 for change detection
        if 'target_gt' in targets and 'change_state' in targets:
            target_pred = predictions['target_logits'].argmax(dim=1)  # [B, H, W]
            state_pred = predictions['state_logits'].argmax(dim=1)    # [B, H, W]
            target_gt = targets['target_gt']
            state_gt = targets['change_state']
            
            # Only count non-background pixels
            non_bg = (target_gt != 255)
            if non_bg.sum() > 0:
                # Binary change detection: changed (state > 0 and state != 6) vs unchanged (state == 0)
                gt_changed = (state_gt[non_bg] > 0) & (state_gt[non_bg] != 6)
                pred_changed = (state_pred[non_bg] > 0) & (state_pred[non_bg] != 6)
                
                if not hasattr(evaluator, 'f1_stats'):
                    evaluator.f1_stats = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
                
                evaluator.f1_stats['tp'] += (gt_changed & pred_changed).sum().item()
                evaluator.f1_stats['fp'] += (~gt_changed & pred_changed).sum().item()
                evaluator.f1_stats['fn'] += (gt_changed & ~pred_changed).sum().item()
                evaluator.f1_stats['tn'] += (~gt_changed & ~pred_changed).sum().item()
                
                # Per-state F1
                if not hasattr(evaluator, 'state_f1'):
                    evaluator.state_f1 = {}
                
                for state_id in range(7):
                    if state_id == 6:  # skip none
                        continue
                    gt_mask = (state_gt[non_bg] == state_id)
                    pred_mask = (state_pred[non_bg] == state_id)
                    tp = (gt_mask & pred_mask).sum().item()
                    fp = (~gt_mask & pred_mask).sum().item()
                    fn = (gt_mask & ~pred_mask).sum().item()
                    
                    if state_id not in evaluator.state_f1:
                        evaluator.state_f1[state_id] = {'tp': 0, 'fp': 0, 'fn': 0}
                    evaluator.state_f1[state_id]['tp'] += tp
                    evaluator.state_f1[state_id]['fp'] += fp
                    evaluator.state_f1[state_id]['fn'] += fn
                
                # Per-target-class change detection F1
                # target class mapping: 0=farmland, 1=runway, 2=taxiway, 3=apron, 
                #                       4=bridge, 5=highway, 6=building, 7=shelter, 8=tower,
                #                       9=pier, 10=dock, 11=tank, 12=aircraft, 13=vessel, 14=crater
                target_class_names = ['farmland', 'runway', 'taxiway', 'apron', 'bridge', 'highway', 
                                      'building', 'shelter', 'tower', 'pier', 'dock', 'tank', 
                                      'aircraft', 'vessel', 'crater']
                
                if not hasattr(evaluator, 'per_target_f1'):
                    evaluator.per_target_f1 = {}
                
                for cls_id in range(15):
                    cls_mask = (target_gt[non_bg] == cls_id)
                    if cls_mask.sum() > 0:
                        cls_gt_changed = gt_changed[cls_mask]
                        cls_pred_changed = pred_changed[cls_mask]
                        tp = (cls_gt_changed & cls_pred_changed).sum().item()
                        fp = (~cls_gt_changed & cls_pred_changed).sum().item()
                        fn = (cls_gt_changed & ~cls_pred_changed).sum().item()
                        
                        if cls_id not in evaluator.per_target_f1:
                            evaluator.per_target_f1[cls_id] = {'tp': 0, 'fp': 0, 'fn': 0, 'name': target_class_names[cls_id]}
                        evaluator.per_target_f1[cls_id]['tp'] += tp
                        evaluator.per_target_f1[cls_id]['fp'] += fp
                        evaluator.per_target_f1[cls_id]['fn'] += fn
        
        # Compute joint accuracy (target AND state both correct)
        if 'target_gt' in targets and 'change_state' in targets:
            target_pred = predictions['target_logits'].argmax(dim=1)  # [B, H, W]
            state_pred = predictions['state_logits'].argmax(dim=1)    # [B, H, W]
            target_gt = targets['target_gt']
            state_gt = targets['change_state']
            
            # Only count non-background pixels
            non_bg = (target_gt != 255)
            if non_bg.sum() > 0:
                target_correct = (target_pred[non_bg] == target_gt[non_bg])
                state_correct = (state_pred[non_bg] == state_gt[non_bg])
                joint_correct = target_correct & state_correct
                
                if not hasattr(evaluator, 'joint_stats'):
                    evaluator.joint_stats = {'target_correct': 0, 'state_correct': 0, 'joint_correct': 0, 'total': 0}
                
                evaluator.joint_stats['target_correct'] += target_correct.sum().item()
                evaluator.joint_stats['state_correct'] += state_correct.sum().item()
                evaluator.joint_stats['joint_correct'] += joint_correct.sum().item()
                evaluator.joint_stats['total'] += non_bg.sum().item()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (bi + 1) % 100 == 0:
            print(f"  Batch {bi+1}/{len(test_loader)}")

# Compute results
avg_loss = total_loss / max(num_batches, 1)
results = evaluator.compute()

print(f"\n{'='*60}")
print(f"TEST SET RESULTS (v7_0617_freq_branch, majority voting)")
print(f"{'='*60}")
print(f"Avg Loss: {avg_loss:.4f}")
print(evaluator.format_results(results))

# Print joint accuracy
if hasattr(evaluator, 'joint_stats') and evaluator.joint_stats['total'] > 0:
    stats = evaluator.joint_stats
    print(f"\n{'='*60}")
    print(f"JOINT ACCURACY (target AND state correct)")
    print(f"{'='*60}")
    print(f"Target Accuracy:  {stats['target_correct']/stats['total']*100:.2f}%")
    print(f"State Accuracy:   {stats['state_correct']/stats['total']*100:.2f}%")
    print(f"Joint Accuracy:   {stats['joint_correct']/stats['total']*100:.2f}%")
    print(f"Total Pixels:     {stats['total']:,}")

# Print F1 scores
if hasattr(evaluator, 'f1_stats'):
    f1 = evaluator.f1_stats
    print(f"\n{'='*60}")
    print(f"CHANGE DETECTION F1 (binary)")
    print(f"{'='*60}")
    precision = f1['tp'] / max(f1['tp'] + f1['fp'], 1)
    recall = f1['tp'] / max(f1['tp'] + f1['fn'], 1)
    f1_score = 2 * precision * recall / max(precision + recall, 1e-8)
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1 Score:  {f1_score*100:.2f}%")
    print(f"TP: {f1['tp']:,}  FP: {f1['fp']:,}  FN: {f1['fn']:,}  TN: {f1['tn']:,}")

# Print per-state F1
if hasattr(evaluator, 'state_f1'):
    state_names = ['no-change', 'damaged', 'reduced', 'added', 'extended', 'replaced']
    print(f"\n{'='*60}")
    print(f"PER-STATE F1 SCORES")
    print(f"{'='*60}")
    for state_id, name in enumerate(state_names):
        if state_id in evaluator.state_f1:
            s = evaluator.state_f1[state_id]
            p = s['tp'] / max(s['tp'] + s['fp'], 1)
            r = s['tp'] / max(s['tp'] + s['fn'], 1)
            f = 2 * p * r / max(p + r, 1e-8)
            print(f"  {name:12s}: P={p*100:.1f}%  R={r*100:.1f}%  F1={f*100:.1f}%")

# Print per-target-class change detection F1
if hasattr(evaluator, 'per_target_f1'):
    print(f"\n{'='*60}")
    print(f"PER-TARGET-CLASS CHANGE DETECTION F1")
    print(f"{'='*60}")
    # Building only (for paper comparison)
    if 6 in evaluator.per_target_f1:
        b = evaluator.per_target_f1[6]
        bp = b['tp'] / max(b['tp'] + b['fp'], 1)
        br = b['tp'] / max(b['tp'] + b['fn'], 1)
        bf = 2 * bp * br / max(bp + br, 1e-8)
        print(f"  Building ONLY: P={bp*100:.1f}%  R={br*100:.1f}%  F1={bf*100:.1f}%")
    
    print(f"\n  Surface Base classes:")
    for cls_id in [0, 2, 3, 1]:  # farmland, taxiway, apron, runway
        if cls_id in evaluator.per_target_f1:
            s = evaluator.per_target_f1[cls_id]
            p = s['tp'] / max(s['tp'] + s['fp'], 1)
            r = s['tp'] / max(s['tp'] + s['fn'], 1)
            f = 2 * p * r / max(p + r, 1e-8)
            print(f"    {s['name']:12s}: P={p*100:.1f}%  R={r*100:.1f}%  F1={f*100:.1f}%")

    print(f"\n  Mobile Targets:")
    for cls_id in [12, 13]:  # aircraft, vessel
        if cls_id in evaluator.per_target_f1:
            s = evaluator.per_target_f1[cls_id]
            p = s['tp'] / max(s['tp'] + s['fp'], 1)
            r = s['tp'] / max(s['tp'] + s['fn'], 1)
            f = 2 * p * r / max(p + r, 1e-8)
            print(f"    {s['name']:12s}: P={p*100:.1f}%  R={r*100:.1f}%  F1={f*100:.1f}%")

    print(f"\n  Fixed Targets:")
    for cls_id in [4, 5, 7, 8, 9, 10, 11, 14]:  # bridge, highway, shelter, tower, pier, dock, tank, crater
        if cls_id in evaluator.per_target_f1:
            s = evaluator.per_target_f1[cls_id]
            p = s['tp'] / max(s['tp'] + s['fp'], 1)
            r = s['tp'] / max(s['tp'] + s['fn'], 1)
            f = 2 * p * r / max(p + r, 1e-8)
            print(f"    {s['name']:12s}: P={p*100:.1f}%  R={r*100:.1f}%  F1={f*100:.1f}%")

