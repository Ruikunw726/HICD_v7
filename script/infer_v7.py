"""
Inference demo: load best_model, run on val samples, visualize pixel + instance predictions.
"""
import os, sys, yaml, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_hicd_v7
from datasets import create_dataloader
import torch.nn as nn

# Config
with open('datasets/configs/second.yaml', 'r', encoding='utf-8') as f:
    dataset_config = yaml.safe_load(f)

model_config = dict(dataset_config)
model_config['backbone'] = {'variant': 'L1', 'pretrained': False, 'pretrained_path': None}
model_config['hidden_dim'] = 128
model_config['active_heads'] = dataset_config.get('active_heads', {})
model_config['pixel_class_weights'] = [2.931, 0.160, 0.061, 2.452, 0.090, 0.307]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_hicd_v7(model_config).to(device)

# Load best checkpoint
ckpt_path = 'checkpoints/v7_second_fix/best_model.pth'
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Loaded {ckpt_path}, epoch={ckpt.get('epoch', '?')}")

# Load val data
val_loader = create_dataloader(
    data_dir='/mnt/c/SECOND_6class_v2', config=dataset_config,
    split='val', batch_size=4, num_workers=0, img_size=1024
)

# Run inference on a few batches
all_miou = []
all_acc = []
all_state_acc = []
class_names = ['Non-veg', 'Tree', 'Low-veg', 'Water', 'Building', 'Playground']

with torch.no_grad():
    for bi, batch in enumerate(val_loader):
        if bi >= 10:  # 10 batches = 40 samples
            break

        img_t1 = batch['img_t1'].to(device)
        img_t2 = batch['img_t2'].to(device)

        # Prepare pixel GT
        targets = {}
        label = batch['label'].to(device)
        if label.dim() == 3:
            label_s = label.unsqueeze(1).float()
            label_s = nn.functional.interpolate(label_s, scale_factor=0.25, mode='nearest').squeeze(1).long()
            targets['target_gt'] = label_s

        # Prepare instance GT
        instance_labels = []
        instance_classes = batch.get('instance_classes')
        instance_states = batch.get('instance_states')
        for i, boxes in enumerate(batch['instance_boxes']):
            if boxes is not None and len(boxes) > 0:
                labels = (instance_classes[i].to(device)
                          if instance_classes is not None and instance_classes[i] is not None
                          else torch.zeros(len(boxes), dtype=torch.long, device=device))
                inst_dict = {'labels': labels, 'boxes': boxes.to(device)}
                if instance_states is not None and instance_states[i] is not None:
                    inst_dict['states'] = instance_states[i].to(device)
                instance_labels.append(inst_dict)
            else:
                instance_labels.append({
                    'labels': torch.zeros(0, dtype=torch.long, device=device),
                    'boxes': torch.zeros(0, 4, device=device)
                })
        targets['instance_labels'] = instance_labels

        predictions = model(img_t1, img_t2)

        # Pixel metrics
        pred_map = predictions['change_heatmap'].argmax(dim=1)  # [B, H, W]
        gt_map = targets['target_gt']
        valid = (gt_map != 255)

        if valid.sum() > 0:
            acc = (pred_map[valid] == gt_map[valid]).float().mean().item()
            all_acc.append(acc)

            for c in range(6):
                p = (pred_map == c) & valid
                g = (gt_map == c) & valid
                inter = (p & g).sum().item()
                union = (p | g).sum().item()
                if union > 0:
                    all_miou.append(inter / union)

        # Instance state: skip during pure inference (proposals from pixel branch, not GT)

        print(f"  Batch {bi}: pixel_acc={acc:.3f}, valid_pixels={valid.sum().item()}")

# Summary
print("\n" + "="*60)
print("Inference Results (best_model, 40 val samples)")
print("="*60)
print(f"  Pixel Accuracy (valid only): {np.mean(all_acc):.4f}")
print(f"  Pixel mIoU: {np.mean(all_miou):.4f}")
print("  Instance State Accuracy: N/A (inference mode)")

# Per-class IoU
print("\n  Per-class IoU:")
all_cls_ious = [[] for _ in range(6)]
with torch.no_grad():
    for bi, batch in enumerate(val_loader):
        if bi >= 10:
            break
        img_t1 = batch['img_t1'].to(device)
        img_t2 = batch['img_t2'].to(device)
        label = batch['label'].to(device)
        if label.dim() == 3:
            label_s = label.unsqueeze(1).float()
            label_s = nn.functional.interpolate(label_s, scale_factor=0.25, mode='nearest').squeeze(1).long()
        predictions = model(img_t1, img_t2)
        pred_map = predictions['change_heatmap'].argmax(dim=1)
        valid = (label_s != 255)
        for c in range(6):
            p = (pred_map == c) & valid
            g = (label_s == c) & valid
            inter = (p & g).sum().item()
            union = (p | g).sum().item()
            if union > 0:
                all_cls_ious[c].append(inter / union)

for i, name in enumerate(class_names):
    iou = np.mean(all_cls_ious[i]) if all_cls_ious[i] else 0
    print(f"    {name}: {iou:.4f}")

# Save one batch of predictions as images for visual inspection
print("\nSaving visual samples...")
import torch.nn.functional as F
from PIL import Image

out_dir = 'checkpoints/v7_second_fix/vis_samples'
os.makedirs(out_dir, exist_ok=True)

with torch.no_grad():
    batch = next(iter(val_loader))
    img_t1 = batch['img_t1'].to(device)
    img_t2 = batch['img_t2'].to(device)
    predictions = model(img_t1, img_t2)
    pred_map = predictions['change_heatmap'].argmax(dim=1).cpu()
    gt_map = batch['label']

    color_map = {
        0: [128, 128, 128],  # Non-veg: gray
        1: [0, 128, 0],      # Tree: green
        2: [128, 255, 0],    # Low-veg: light green
        3: [0, 0, 255],      # Water: blue
        4: [255, 0, 0],      # Building: red
        5: [255, 255, 0],    # Playground: yellow
        255: [0, 0, 0],      # Ignore: black
    }

    for i in range(min(4, len(pred_map))):
        # Prediction
        pred_color = np.zeros((256, 256, 3), dtype=np.uint8)
        for c, color in color_map.items():
            mask = (pred_map[i] == c).numpy()
            pred_color[mask] = color
        Image.fromarray(pred_color).save(os.path.join(out_dir, f'pred_{i}.png'))

        # GT (original size)
        gt_color = np.zeros((1024, 1024, 3), dtype=np.uint8)
        for c, color in color_map.items():
            mask = (gt_map[i] == c).numpy()
            gt_color[mask] = color
        Image.fromarray(gt_color).save(os.path.join(out_dir, f'gt_{i}.png'))

        # Input T1
        t1 = (img_t1[i].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(t1).save(os.path.join(out_dir, f't1_{i}.png'))

        # Input T2
        t2 = (img_t2[i].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(t2).save(os.path.join(out_dir, f't2_{i}.png'))

print(f"Saved to {out_dir}/ (pred_*.png, gt_*.png, t1_*.png, t2_*.png)")
