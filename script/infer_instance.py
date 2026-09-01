"""
Instance branch inference: extract proposals from pixel branch, classify states.
"""
import os, sys, yaml, torch, numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import build_hicd_v7
from datasets import create_dataloader
import torch.nn as nn

with open('datasets/configs/second.yaml', 'r', encoding='utf-8') as f:
    dataset_config = yaml.safe_load(f)

model_config = dict(dataset_config)
model_config['backbone'] = {'variant': 'L1', 'pretrained': False, 'pretrained_path': None}
model_config['hidden_dim'] = 128
model_config['active_heads'] = dataset_config.get('active_heads', {})
model_config['pixel_class_weights'] = [2.931, 0.160, 0.061, 2.452, 0.090, 0.307]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_hicd_v7(model_config).to(device)

ckpt = torch.load('checkpoints/v7_second_fix/best_model.pth', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Loaded best_model (epoch {ckpt.get('epoch')})")

val_loader = create_dataloader(
    data_dir='/mnt/c/SECOND_6class_v2', config=dataset_config,
    split='val', batch_size=1, num_workers=0, img_size=1024
)

state_names = ['no-change', 'increase', 'decrease']
results = []

with torch.no_grad():
    for bi, batch in enumerate(val_loader):
        if bi >= 20:
            break
        img_t1 = batch['img_t1'].to(device)
        img_t2 = batch['img_t2'].to(device)

        # Run backbone + heads to get features + heatmap
        feat_t1_list = model.backbone(img_t1)
        feat_t2_list = model.backbone(img_t2)
        nfa_t1 = model.nfa_t1(*feat_t1_list)
        nfa_t2 = model.nfa_t2(*feat_t2_list)
        fused, t1_feat, t2_feat = model.tfm(nfa_t1, nfa_t2)
        pixel_feat = model.semantic_adapter(fused)
        instance_feat = model.instance_adapter(fused)
        change_heatmap, _ = model.semantic_head(pixel_feat)

        # Extract building proposals from heatmap
        probs = torch.softmax(change_heatmap, dim=1)
        building_prob = probs[0, 4]  # class 4 = Building
        building_mask = (building_prob > 0.5).cpu().numpy().astype(np.uint8)

        # Connected components
        labeled, nf = ndimage.label(building_mask)
        pred_boxes = []
        pred_scores = []
        for i in range(1, nf + 1):
            ys, xs = np.where(labeled == i)
            if len(ys) < 50:  # min area filter
                continue
            H, W = building_mask.shape
            cx = (xs.min() + xs.max()) / 2 / W
            cy = (ys.min() + ys.max()) / 2 / H
            bw = (xs.max() - xs.min()) / W
            bh = (ys.max() - ys.min()) / H
            score = building_prob[labeled == i].mean().item()
            pred_boxes.append([cx, cy, bw, bh])
            pred_scores.append(score)

        # Run instance head with predicted proposals
        if pred_boxes:
            proposal_tensor = torch.tensor([pred_boxes], dtype=torch.float32, device=device)[0]
            inst_feat = instance_feat[0]
            state_logits, roi_features = model.instance_head(inst_feat, [proposal_tensor])
            pred_states = state_logits.argmax(dim=1).cpu().numpy()
            pred_state_probs = torch.softmax(state_logits, dim=1).cpu().numpy()
        else:
            pred_states = np.array([])
            pred_state_probs = np.array([])

        # GT boxes
        gt_boxes = batch['instance_boxes'][0]
        gt_states = batch['instance_states'][0] if 'instance_states' in batch and batch['instance_states'][0] is not None else None

        # Compute IoU matching
        matched_gt = 0
        matched_state_correct = 0
        total_gt = len(gt_boxes) if gt_boxes is not None else 0
        total_pred = len(pred_boxes)

        if gt_boxes is not None and len(gt_boxes) > 0 and len(pred_boxes) > 0:
            gt_np = gt_boxes.numpy()
            for gi in range(len(gt_np)):
                g_cx, g_cy, g_w, g_h = gt_np[gi]
                g_x1, g_y1 = g_cx - g_w/2, g_cy - g_h/2
                g_x2, g_y2 = g_cx + g_w/2, g_cy + g_h/2
                g_area = g_w * g_h

                best_iou = 0
                best_pi = -1
                for pi, pb in enumerate(pred_boxes):
                    p_cx, p_cy, p_w, p_h = pb
                    p_x1, p_y1 = p_cx - p_w/2, p_cy - p_h/2
                    p_x2, p_y2 = p_cx + p_w/2, p_cy + p_h/2
                    p_area = p_w * p_h

                    ix1 = max(g_x1, p_x1); iy1 = max(g_y1, p_y1)
                    ix2 = min(g_x2, p_x2); iy2 = min(g_y2, p_y2)
                    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                    union = g_area + p_area - inter
                    iou = inter / max(union, 1e-6)
                    if iou > best_iou:
                        best_iou = iou
                        best_pi = pi

                if best_iou >= 0.5:
                    matched_gt += 1
                    if gt_states is not None:
                        gt_s = gt_states[gi].item()
                        pred_s = pred_states[best_pi]
                        if gt_s == pred_s:
                            matched_state_correct += 1

        rec = matched_gt / max(total_gt, 1)
        state_acc = matched_state_correct / max(matched_gt, 1)

        if total_gt > 0 or total_pred > 0:
            results.append({
                'gt_boxes': total_gt, 'pred_boxes': total_pred,
                'matched': matched_gt, 'recall': rec,
                'state_correct': matched_state_correct, 'state_acc': state_acc
            })
            print(f"  [{bi:2d}] GT={total_gt:3d} Pred={total_pred:3d} Matched@0.5={matched_gt:3d} Recall={rec:.2f} StateAcc={state_acc:.2f}")

# Summary
print("\n" + "="*60)
print("Instance Branch Inference Results (20 val samples)")
print("="*60)
if results:
    total_gt = sum(r['gt_boxes'] for r in results)
    total_pred = sum(r['pred_boxes'] for r in results)
    total_matched = sum(r['matched'] for r in results)
    total_state_correct = sum(r['state_correct'] for r in results)
    print(f"  Total GT boxes: {total_gt}")
    print(f"  Total predicted boxes: {total_pred}")
    print(f"  Matched (IoU>=0.5): {total_matched}")
    print(f"  Recall: {total_matched/max(total_gt,1):.4f}")
    print(f"  State accuracy (matched only): {total_state_correct/max(total_matched,1):.4f}" if total_matched > 0 else "  State accuracy: N/A")

    # Per-state breakdown
    print("\n  Per-image stats:")
    avg_gt = np.mean([r['gt_boxes'] for r in results])
    avg_pred = np.mean([r['pred_boxes'] for r in results])
    avg_recall = np.mean([r['recall'] for r in results])
    print(f"    Avg GT/image: {avg_gt:.1f}")
    print(f"    Avg Pred/image: {avg_pred:.1f}")
    print(f"    Avg Recall: {avg_recall:.4f}")
else:
    print("  No results!")
