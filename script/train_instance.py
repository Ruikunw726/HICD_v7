"""
Train instance-level classifier for T3/T4 targets.

Usage:
    python script/train_instance.py --data_dir D:/CD/0617final_256_v2 --epochs 50
"""

import os
import sys
import argparse
import time
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.instance_classifier import (
    InstanceDataset, InstanceClassifier, InstanceLoss,
    get_class_weights, INSTANCE_CLASSES, STATE_NAMES
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Instance Classifier')
    parser.add_argument('--data_dir', type=str, default='D:/CD/0617final_256_v2')
    parser.add_argument('--instance_json', type=str, default=None,
                        help='Path to instance_dataset.json (default: data_dir/instance_dataset.json)')
    parser.add_argument('--save_dir', type=str, default='checkpoints/instance_classifier')
    parser.add_argument('--exp_name', type=str, default='inst_v1')
    parser.add_argument('--crop_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--w_state', type=float, default=1.0)
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    return parser.parse_args()


def create_dataloaders(args):
    """Create train/val dataloaders with class-balanced sampling."""
    instance_json = args.instance_json or os.path.join(args.data_dir, 'instance_dataset.json')

    # Full dataset
    full_dataset = InstanceDataset(
        data_dir=args.data_dir,
        instance_json=instance_json,
        crop_size=args.crop_size,
        augment=True,
    )

    # Split train/val
    n_total = len(full_dataset)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val

    torch.manual_seed(args.seed)
    indices = torch.randperm(n_total).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = InstanceDataset(
        data_dir=args.data_dir,
        instance_json=instance_json,
        crop_size=args.crop_size,
        augment=True,
    )
    val_dataset = InstanceDataset(
        data_dir=args.data_dir,
        instance_json=instance_json,
        crop_size=args.crop_size,
        augment=False,
    )

    # Class-balanced sampling for training
    train_class_counts = np.zeros(len(INSTANCE_CLASSES))
    for idx in train_indices:
        train_class_counts[train_dataset.samples[idx]['class_id']] += 1

    sample_weights = np.zeros(len(train_indices))
    for i, idx in enumerate(train_indices):
        cls = train_dataset.samples[idx]['class_id']
        sample_weights[i] = 1.0 / max(train_class_counts[cls], 1)
    sample_weights = sample_weights / sample_weights.sum()

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_indices),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=sampler, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"Train: {n_train} instances, Val: {n_val} instances")
    print(f"Class distribution (train): {train_class_counts.astype(int).tolist()}")

    return train_loader, val_loader, full_dataset


def evaluate(model, loader, device):
    """Evaluate model and return per-class metrics."""
    model.eval()
    correct_target = 0
    correct_state = 0
    total = 0

    per_class_correct = np.zeros(len(INSTANCE_CLASSES))
    per_class_total = np.zeros(len(INSTANCE_CLASSES))

    with torch.no_grad():
        for batch in loader:
            images_t1 = batch['image_t1'].to(device)
            images_t2 = batch['image_t2'].to(device)
            targets = batch['target'].to(device)
            states = batch['state'].to(device)

            target_logits, state_logits = model(images_t1, images_t2)

            pred_target = target_logits.argmax(dim=1)
            pred_state = state_logits.argmax(dim=1)

            correct_target += (pred_target == targets).sum().item()
            correct_state += (pred_state == states).sum().item()
            total += targets.size(0)

            for cls in range(len(INSTANCE_CLASSES)):
                mask = (targets == cls)
                per_class_total[cls] += mask.sum().item()
                per_class_correct[cls] += ((pred_target == targets) & mask).sum().item()

    acc_target = correct_target / max(total, 1)
    acc_state = correct_state / max(total, 1)

    per_class_acc = {}
    for cls in range(len(INSTANCE_CLASSES)):
        if per_class_total[cls] > 0:
            per_class_acc[INSTANCE_CLASSES[cls]] = per_class_correct[cls] / per_class_total[cls]
        else:
            per_class_acc[INSTANCE_CLASSES[cls]] = 0.0

    return acc_target, acc_state, per_class_acc


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct_target = 0
    total = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch in pbar:
        images_t1 = batch['image_t1'].to(device)
        images_t2 = batch['image_t2'].to(device)
        targets = batch['target'].to(device)
        states = batch['state'].to(device)

        target_logits, state_logits = model(images_t1, images_t2)
        loss, loss_dict = criterion(target_logits, state_logits, targets, states)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = target_logits.argmax(dim=1)
        correct_target += (pred == targets).sum().item()
        total += targets.size(0)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct_target/total:.3f}',
        })

    return total_loss / len(loader), correct_target / total


def main():
    args = parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Save dir
    save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # Data
    print("\nLoading data...")
    train_loader, val_loader, full_dataset = create_dataloaders(args)

    # Class weights
    class_weights = get_class_weights(full_dataset)
    print(f"Class weights: {class_weights.numpy().round(3).tolist()}")

    # State weights (inverse frequency)
    state_counts = np.zeros(len(STATE_NAMES))
    for s in full_dataset.samples:
        state_counts[s['state_id']] += 1
    state_counts = np.maximum(state_counts, 1)
    state_weights = torch.FloatTensor(1.0 / state_counts)
    state_weights = state_weights / state_weights.sum() * len(state_weights)
    print(f"State weights: {state_weights.numpy().round(3).tolist()}")

    # Model
    model = InstanceClassifier(
        num_classes=len(INSTANCE_CLASSES),
        num_states=len(STATE_NAMES),
        crop_size=args.crop_size,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.2f}M")

    # Loss and optimizer
    criterion = InstanceLoss(
        class_weights=class_weights.to(device),
        state_weights=state_weights.to(device),
        w_state=args.w_state,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_acc = 0
    log_file = os.path.join(save_dir, 'train_log.csv')

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_acc_target', 'val_acc_state', 'lr'])

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )

            val_acc_target, val_acc_state, per_class_acc = evaluate(
                model, val_loader, device
            )

            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0

            # Log
            writer.writerow([epoch, train_loss, train_acc, val_acc_target, val_acc_state, lr])
            f.flush()

            print(f"Epoch {epoch}/{args.epochs} - "
                  f"Loss: {train_loss:.4f} - "
                  f"Train Acc: {train_acc:.4f} - "
                  f"Val Target: {val_acc_target:.4f} - "
                  f"Val State: {val_acc_state:.4f} - "
                  f"LR: {lr:.6f} - "
                  f"Time: {elapsed:.1f}s")

            # Per-class (every 5 epochs)
            if epoch % 5 == 0 or epoch == args.epochs:
                print("  Per-class accuracy:")
                for cls_name, acc in per_class_acc.items():
                    print(f"    {cls_name:<12}: {acc:.4f}")

            # Save best
            if val_acc_target > best_acc:
                best_acc = val_acc_target
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc_target': val_acc_target,
                    'val_acc_state': val_acc_state,
                    'args': vars(args),
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"  -> Saved best model (val_acc={val_acc_target:.4f})")

            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc_target': val_acc_target,
                    'val_acc_state': val_acc_state,
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))

    print(f"\nTraining complete. Best val acc: {best_acc:.4f}")
    print(f"Model saved to: {save_dir}")


if __name__ == '__main__':
    main()
