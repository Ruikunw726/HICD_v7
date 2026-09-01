"""
ICD Unified Evaluation Protocol for HICD v7.
Evaluates change detection at instance level:
- Extracts connected components from GT labels as GT instances
- Extracts connected components from predicted change map as predicted instances
- Matches via IoU at multiple thresholds
- Computes ICD-Precision, ICD-Recall, ICD-F1, ICD-mAP

Supports both instance branch (bbox proposals) and pixel branch (connected components).
"""
import os, sys, yaml, torch, numpy as np
from collections import defaultdict
from scipy import ndimage
import tifffile


def read_label(path):
    arr = tifffile.imread(path)
    if arr.ndim == 3:
        arr = arr[:,:,0]
    return arr


def extract_instances_from_label(label, target_classes=None, min_area=20):
    """Extract connected components from a label map as GT instances.
    Returns list of dicts: {class_id, bbox_cxcywh, area, mask}
    """
    instances = []
    H, W = label.shape
    for cls in target_classes:
        mask = (label == cls)
        if mask.sum() < min_area:
            continue
        labeled, nf = ndimage.label(mask)
        for i in range(1, nf + 1):
            ys, xs = np.where(labeled == i)
            area = len(ys)
            if area < min_area:
                continue
            y1, y2 = ys.min(), ys.max() + 1
            x1, x2 = xs.min(), xs.max() + 1
            cx = (x1 + x2) / 2 / W
            cy = (y1 + y2) / 2 / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            inst_mask = np.zeros((H, W), dtype=bool)
            inst_mask[ys, xs] = True
            instances.append({
                'class_id': cls,
                'bbox': [cx, cy, bw, bh],
                'area': area,
                'mask': inst_mask,
            })
    return instances


def extract_instances_from_prediction(pred_map, target_classes=None, min_area=20):
    """Extract connected components from a predicted segmentation map."""
    instances = []
    H, W = pred_map.shape
    for cls in target_classes:
        mask = (pred_map == cls)
        if mask.sum() < min_area:
            continue
        labeled, nf = ndimage.label(mask)
        for i in range(1, nf + 1):
            ys, xs = np.where(labeled == i)
            area = len(ys)
            if area < min_area:
                continue
            y1, y2 = ys.min(), ys.max() + 1
            x1, x2 = xs.min(), xs.max() + 1
            cx = (x1 + x2) / 2 / W
            cy = (y1 + y2) / 2 / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            inst_mask = np.zeros((H, W), dtype=bool)
            inst_mask[ys, xs] = True
            instances.append({
                'class_id': cls,
                'bbox': [cx, cy, bw, bh],
                'area': area,
                'mask': inst_mask,
            })
    return instances


def mask_iou(m1, m2):
    inter = (m1 & m2).sum()
    union = (m1 | m2).sum()
    return inter / max(union, 1)


def box_iou_cxcywh(b1, b2):
    """IoU between two boxes in [cx, cy, w, h] normalized format."""
    def to_xyxy(b):
        return [b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2]
    a = to_xyxy(b1)
    b = to_xyxy(b2)
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    area1 = (a[2]-a[0]) * (a[3]-a[1])
    area2 = (b[2]-b[0]) * (b[3]-b[1])
    return inter / max(area1 + area2 - inter, 1e-6)


def compute_icd_metrics(gt_instances, pred_instances, iou_thresholds=None, use_mask=True):
    """Compute ICD-Precision, Recall, F1, mAP.
    Args:
        gt_instances: list of {class_id, bbox, area, mask}
        pred_instances: list of {class_id, bbox, area, mask, score}
        iou_thresholds: list of IoU thresholds
        use_mask: use mask IoU (True) or box IoU (False)
    """
    if iou_thresholds is None:
        iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Sort predictions by score
    pred_instances = sorted(pred_instances, key=lambda x: -x.get('score', 1.0))

    results = {}
    for iou_t in iou_thresholds:
        matched_gt = set()
        tp, fp = 0, 0
        for pred in pred_instances:
            best_iou, best_idx = 0, -1
            for idx, gt in enumerate(gt_instances):
                if idx in matched_gt:
                    continue
                if pred['class_id'] != gt['class_id']:
                    continue
                if use_mask:
                    iou = mask_iou(pred['mask'], gt['mask'])
                else:
                    iou = box_iou_cxcywh(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_t and best_idx >= 0:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1

        n_gt = len(gt_instances)
        fn = n_gt - len(matched_gt)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(n_gt, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        results[iou_t] = {
            'tp': tp, 'fp': fp, 'fn': fn, 'n_gt': n_gt,
            'precision': precision, 'recall': recall, 'f1': f1,
        }

    # mAP = mean of F1 at each threshold (simplified)
    aps = [results[t]['f1'] for t in iou_thresholds]
    results['mAP'] = float(np.mean(aps))
    results['mAP_50'] = results.get(0.5, {}).get('f1', 0)
    results['mAP_75'] = results.get(0.75, {}).get('f1', 0)
    return results


def evaluate_icd(data_dir, split, model, device, config, 
                 target_classes=None, iou_thresholds=None, max_samples=None):
    """Run ICD evaluation on a dataset split.
    Args:
        data_dir: path to dataset root
        split: 'val' or 'test'
        model: HICD_v7 model
        device: torch device
        config: dataset config dict
        target_classes: list of class IDs to evaluate (non-background)
        iou_thresholds: list of IoU thresholds
        max_samples: limit number of samples (for speed)
    """
    if target_classes is None:
        target_classes = [i for i in range(1, config.get('num_pixel_targets', 6))]
    if iou_thresholds is None:
        iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Collect all results
    all_results = []
    per_class_results = defaultdict(list)

    # Iterate over scenes
    scenes = []
    for item in sorted(os.listdir(data_dir)):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item not in ['train', 'val', 'test']:
            if os.path.exists(os.path.join(item_path, split)):
                scenes.append(item)

    count = 0
    for scene in scenes:
        split_dir = os.path.join(data_dir, scene, split)
        img_dir = os.path.join(split_dir, 'image')
        lbl_dir = os.path.join(split_dir, 'label')
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            continue

        pre_suffix = config.get('pre_suffix', '_pre_war')
        post_suffix = config.get('post_suffix', '_post_war')
        label_suffix = config.get('label_suffix', '_target')
        image_ext = config.get('image_ext', '.tif')

        label_decode_map = config.get('label_decode_map', None)

        files = sorted(os.listdir(img_dir))
        for fname in files:
            if not fname.endswith(pre_suffix + image_ext):
                continue
            if max_samples and count >= max_samples:
                break

            base = fname.replace(pre_suffix + image_ext, '')
            pre_path = os.path.join(img_dir, fname)
            post_path = os.path.join(img_dir, base + post_suffix + image_ext)
            lbl_path = os.path.join(lbl_dir, base + label_suffix + '.tif')

            if not os.path.exists(post_path) or not os.path.exists(lbl_path):
                continue

            # Load images
            pre_img = read_label(pre_path).astype(np.float32)
            post_img = read_label(post_path).astype(np.float32)
            if pre_img.ndim == 2:
                pre_img = np.stack([pre_img] * 3, axis=0)
            elif pre_img.shape[0] > 3:
                pre_img = pre_img[:3]
            if post_img.ndim == 2:
                post_img = np.stack([post_img] * 3, axis=0)
            elif post_img.shape[0] > 3:
                post_img = post_img[:3]
            pre_img = pre_img / 255.0
            post_img = post_img / 255.0

            # Load and decode label
            raw_label = read_label(lbl_path)
            if raw_label.ndim == 3:
                raw_label = raw_label[0]

            # Decode target type
            if label_decode_map:
                gt_label = np.zeros_like(raw_label)
                for src, dst in label_decode_map.items():
                    gt_label[raw_label == src] = dst
            else:
                gt_label = raw_label

            # Resize to model input size
            S = 256
            from PIL import Image
            pre_pil = Image.fromarray((pre_img.transpose(1, 2, 0) * 255).astype(np.uint8))
            post_pil = Image.fromarray((post_img.transpose(1, 2, 0) * 255).astype(np.uint8))
            pre_resized = np.array(pre_pil.resize((S, S), Image.BILINEAR)).transpose(2, 0, 1).astype(np.float32) / 255.0
            post_resized = np.array(post_pil.resize((S, S), Image.BILINEAR)).transpose(2, 0, 1).astype(np.float32) / 255.0

            # Resize GT label
            gt_pil = Image.fromarray(gt_label.astype(np.uint8))
            gt_resized = np.array(gt_pil.resize((S, S), Image.NEAREST))

            # Model inference
            t1 = torch.from_numpy(pre_resized).unsqueeze(0).to(device)
            t2 = torch.from_numpy(post_resized).unsqueeze(0).to(device)
            with torch.no_grad():
                predictions = model(t1, t2)
            pred_map = predictions['change_heatmap'].argmax(dim=1).squeeze(0).cpu().numpy()
            # Resize prediction to GT size if needed
            if pred_map.shape != gt_resized.shape:
                from PIL import Image as PILImage
                pred_pil = PILImage.fromarray(pred_map.astype(np.uint8))
                pred_map = np.array(pred_pil.resize((gt_resized.shape[1], gt_resized.shape[0]), PILImage.NEAREST))

            # Extract GT instances
            gt_instances = extract_instances_from_label(gt_resized, target_classes, min_area=20)

            # Extract predicted instances
            pred_instances = extract_instances_from_prediction(pred_map, target_classes, min_area=20)
            for inst in pred_instances:
                inst['score'] = 1.0  # uniform score for pixel predictions

            # Compute ICD metrics
            result = compute_icd_metrics(gt_instances, pred_instances, iou_thresholds)
            all_results.append({
                'scene': scene,
                'base': base,
                'n_gt': len(gt_instances),
                'n_pred': len(pred_instances),
                'result': result,
            })

            # Per-class
            for cls in target_classes:
                gt_cls = [g for g in gt_instances if g['class_id'] == cls]
                pred_cls = [p for p in pred_instances if p['class_id'] == cls]
                if gt_cls or pred_cls:
                    cls_result = compute_icd_metrics(gt_cls, pred_cls, iou_thresholds)
                    per_class_results[cls].append(cls_result)

            count += 1
            if count % 100 == 0:
                print(f"  Evaluated {count} samples...")

    # Aggregate
    total_gt = sum(r['n_gt'] for r in all_results)
    total_pred = sum(r['n_pred'] for r in all_results)

    # Micro-average across samples
    agg_results = {}
    for iou_t in iou_thresholds:
        total_tp = sum(r['result'][iou_t]['tp'] for r in all_results)
        total_fp = sum(r['result'][iou_t]['fp'] for r in all_results)
        total_fn = sum(r['result'][iou_t]['fn'] for r in all_results)
        p = total_tp / max(total_tp + total_fp, 1)
        r = total_tp / max(total_gt, 1)
        f1 = 2 * p * r / max(p + r, 1e-6)
        agg_results[iou_t] = {'precision': p, 'recall': r, 'f1': f1, 'tp': total_tp, 'fp': total_fp}

    aps = [agg_results[t]['f1'] for t in iou_thresholds]
    agg_results['mAP'] = float(np.mean(aps))
    agg_results['mAP_50'] = agg_results.get(0.5, {}).get('f1', 0)
    agg_results['mAP_75'] = agg_results.get(0.75, {}).get('f1', 0)

    # Print
    print(f"\n{'='*60}")
    print(f"ICD Evaluation Results ({count} samples, {total_gt} GT instances, {total_pred} predictions)")
    print(f"{'='*60}")
    print(f"ICD-mAP@[0.1:0.9]:  {agg_results['mAP']:.4f}")
    print(f"ICD-mAP@0.5:        {agg_results['mAP_50']:.4f}")
    print(f"ICD-mAP@0.75:       {agg_results['mAP_75']:.4f}")
    print()
    print("Per-IoU-threshold:")
    for iou_t in iou_thresholds:
        r = agg_results[iou_t]
        print(f"  IoU={iou_t:.2f}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")

    # Per-class results
    pixel_target_names = config.get('pixel_target_names', [])
    print(f"\nPer-class ICD-mAP@0.5:")
    for cls in target_classes:
        cls_name = pixel_target_names[cls] if cls < len(pixel_target_names) else f'cls_{cls}'
        if cls in per_class_results and per_class_results[cls]:
            cls_50 = np.mean([r.get(0.5, {}).get('f1', 0) for r in per_class_results[cls]])
            cls_map = np.mean([r.get('mAP', 0) for r in per_class_results[cls]])
            n = len(per_class_results[cls])
            print(f"  {cls_name:15s}: mAP@0.5={cls_50:.3f}  mAP={cls_map:.3f}  (n={n})")
        else:
            print(f"  {cls_name:15s}: no instances")

    return agg_results, per_class_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='ICD Evaluation Protocol')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='datasets/configs/0617final.yaml')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import build_hicd_v7

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_config = dict(config)
    model_config['backbone'] = {'variant': 'L1', 'pretrained': False, 'pretrained_path': None}
    model_config['hidden_dim'] = 128
    model_config['active_heads'] = config.get('active_heads', {})
    model_config['pixel_class_weights'] = config.get('pixel_class_weights', None)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = build_hicd_v7(model_config).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch', '?')}")

    target_classes = [i for i in range(1, config.get('num_pixel_targets', 6))]
    evaluate_icd(
        data_dir=args.data_dir, split=args.split,
        model=model, device=device, config=config,
        target_classes=target_classes, max_samples=args.max_samples,
    )


if __name__ == '__main__':
    main()



