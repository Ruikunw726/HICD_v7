"""
TTA (Test-Time Augmentation) inference for HICD v7.
Applies multiple augmentations and averages predictions.
"""
import os, sys, yaml, torch, numpy as np
sys.path.insert(0, r'F:\mambacd\home\HICD_v7')
from models import build_hicd_v7
from datasets import create_dataloader
from evaluation import ICDEvaluatorV6 as ICDEvaluator
import torch.nn.functional as F


def tta_augment(img1, img2):
    """Return list of (img1, img2, inverse_transform_fn) for TTA."""
    augmentations = [
        ('original', img1, img2, lambda p: p),
        ('hflip', img1.flip(-1), img2.flip(-1), lambda p: p.flip(-1)),
        ('vflip', img1.flip(-2), img2.flip(-2), lambda p: p.flip(-2)),
        ('hvflip', img1.flip(-1).flip(-2), img2.flip(-1).flip(-2), lambda p: p.flip(-1).flip(-2)),
        # Scale variants
        ('scale0.8', None, None, 'scale0.8'),
        ('scale1.2', None, None, 'scale1.2'),
    ]
    result = []
    for name, i1, i2, inv_fn in augmentations:
        if i1 is None:
            # Scale augmentation
            scale = 0.8 if name == 'scale0.8' else 1.2
            B, C, H, W = img1.shape
            new_h, new_w = int(H * scale), int(W * scale)
            i1 = F.interpolate(img1, size=(new_h, new_w), mode='bilinear', align_corners=True)
            i2 = F.interpolate(img2, size=(new_h, new_w), mode='bilinear', align_corners=True)
            # Store scale info for inverse
            result.append((name, i1, i2, ('scale', H, W)))
        else:
            result.append((name, i1, i2, ('none',)))
    return result


def run_tta(model, img1, img2, num_classes=6):
    """Run TTA and return averaged predictions."""
    model.eval()
    augmentations = tta_augment(img1, img2)

    all_target = None
    all_state = None
    all_change = None
    count = 0

    with torch.no_grad():
        for name, i1, i2, inv_info in augmentations:
            predictions = model(i1, i2)
            target_prob = F.softmax(predictions['change_heatmap'], dim=1)
            state_prob = F.softmax(predictions.get('state_logits_pixel', predictions['change_heatmap']), dim=1)
            change_prob = F.softmax(predictions['change_logits'], dim=1)

            # Inverse transform
            if inv_info[0] == 'scale':
                _, orig_h, orig_w = inv_info
                target_prob = F.interpolate(target_prob, size=(orig_h, orig_w), mode='bilinear', align_corners=True)
                state_prob = F.interpolate(state_prob, size=(orig_h, orig_w), mode='bilinear', align_corners=True)
                change_prob = F.interpolate(change_prob, size=(orig_h, orig_w), mode='bilinear', align_corners=True)
            elif inv_info[0] == 'none':
                pass  # already correct

            if all_target is None:
                all_target = target_prob
                all_state = state_prob
                all_change = change_prob
            else:
                all_target += target_prob
                all_state += state_prob
                all_change += change_prob
            count += 1

    return all_target / count, all_state / count, all_change / count


def main():
    with open(r'F:\mambacd\home\HICD_v7\datasets\configs\0617final.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model_config = dict(config)
    model_config['backbone'] = {'variant': 'L1', 'pretrained': False, 'pretrained_path': None}
    model_config['hidden_dim'] = 128
    model_config['active_heads'] = config.get('active_heads', {})
    model_config['pixel_class_weights'] = [2.931, 0.160, 0.061, 2.452, 0.090, 0.307]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_hicd_v7(model_config).to(device)

    # Load best model
    ckpt_path = r'F:\mambacd\home\HICD_v7\checkpoints\v7_0617_256\best_model.pth'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded {ckpt_path}, epoch={ckpt.get('epoch', '?')}")

    val_loader = create_dataloader(
        data_dir=r'D:\CD\0617final', config=config,
        split='val', batch_size=1, num_workers=0, img_size=256
    )

    # Evaluate with TTA
    all_miou = []
    all_acc = []
    all_state_correct = []
    all_state_total = []
    class_names = config['pixel_target_names']

    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if bi >= 200:
                break

            img_t1 = batch['img_t1'].to(device)
            img_t2 = batch['img_t2'].to(device)

            target_prob, state_prob, change_prob = run_tta(model, img_t1, img_t2)
            pred_map = target_prob.argmax(dim=1)

            # Prepare GT
            label = batch['label']
            if label.dim() == 3:
                label = label.unsqueeze(1).float()
                label = F.interpolate(label, scale_factor=0.25, mode='nearest').squeeze(1).long()
            label = label.to(device)
            valid = (label != 255)

            if valid.sum() > 0:
                acc = (pred_map[valid] == label[valid]).float().mean().item()
                all_acc.append(acc)

                for c in range(target_prob.shape[1]):
                    p = (pred_map == c) & valid
                    g = (label == c) & valid
                    inter = (p & g).sum().item()
                    union = (p | g).sum().item()
                    if union > 0:
                        all_miou.append(inter / union)

            if (bi + 1) % 50 == 0:
                print(f"  [{bi+1}/200] running mIoU: {np.mean(all_miou):.4f}")

    print(f"\n=== TTA Results ({bi+1} samples) ===")
    print(f"mIoU: {np.mean(all_miou):.4f}")
    print(f"Accuracy: {np.mean(all_acc):.4f}")

    # Per-class
    per_class = {}
    for bi, batch in enumerate(val_loader):
        if bi >= 200:
            break
        # Already computed above, just print summary
    print("\nDone!")


if __name__ == '__main__':
    main()
