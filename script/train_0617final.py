"""
HICD v7 Training Script
Full-image training, proposal-based instance branch + pixel branch.
"""

import os
import sys
import argparse
import yaml
import time
import csv
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_hicd_v7
from datasets import create_dataloader
from evaluation import ICDEvaluatorV6 as ICDEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='HICD v7 Training')

    # Dataset
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--scenes', type=str, default=None)
    parser.add_argument('--boxes_dir', type=str, default=None)
    parser.add_argument('--config_dir', type=str, default='datasets/configs')
    parser.add_argument('--img_size', type=int, default=1024)

    # Model
    parser.add_argument('--backbone', type=str, default='L1')
    parser.add_argument('--pretrained_weight_path', type=str, default='backbone_weights/lwganet_l1.pth')
    parser.add_argument('--hidden_dim', type=int, default=128)

    # Training
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--freeze_backbone_epochs', type=int, default=10)
    parser.add_argument('--unfreeze_ratio', type=float, default=0.5)

    # Loss weights
    parser.add_argument('--w_pixel', type=float, default=1.0)
    parser.add_argument('--w_state', type=float, default=2.0)
    parser.add_argument('--w_bbox', type=float, default=5.0)
    parser.add_argument('--w_giou', type=float, default=2.0)
    parser.add_argument('--w_consistency', type=float, default=0.3)
    parser.add_argument('--pixel_class_weights', type=float, nargs=6, default=None,
                        help='Class weights for pixel CE loss, e.g. --pixel_class_weights 2.9 0.16 0.06 2.5 0.09 0.31')
    parser.add_argument('--inst_state_weights', type=float, nargs=3, default=None,
                        help='Instance state CE weights [no_change, increase, decrease]')
    parser.add_argument('--w_state_pixel', type=float, default=1.0)
    parser.add_argument('--w_change', type=float, default=1.0)

    # Saving
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--exp_name', type=str, default='v7_second')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=50)

    # Resume
    parser.add_argument('--resume', type=str, default=None)

    return parser.parse_args()


class TrainerV7:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load dataset config
        config_path = os.path.join(args.config_dir, f'{args.dataset}.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.dataset_config = yaml.safe_load(f)

        print(f"Dataset config: {self.dataset_config}")

        # Build model
        model_config = self._build_model_config()
        self.model = build_hicd_v7(model_config).to(self.device)
        self._print_model_summary()

        # Dataloaders
        scenes = args.scenes.split(',') if args.scenes else None
        self.train_loader = create_dataloader(
            data_dir=args.data_dir, config=self.dataset_config,
            split='train', batch_size=args.batch_size,
            num_workers=args.num_workers, scenes=scenes,
            img_size=args.img_size, boxes_dir=args.boxes_dir
        )
        self.val_loader = create_dataloader(
            data_dir=args.data_dir, config=self.dataset_config,
            split='val', batch_size=args.batch_size,
            num_workers=args.num_workers, scenes=scenes,
            img_size=args.img_size, boxes_dir=args.boxes_dir
        )

        # Optimizer
        param_groups = self.model.get_param_groups(
            lr=args.learning_rate, weight_decay=args.weight_decay
        )
        self.optimizer = optim.AdamW(param_groups)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.max_epochs,
            eta_min=args.learning_rate * 0.01
        )

        # AMP
        self.scaler = GradScaler('cuda') if args.use_amp else None

        # Evaluator
        self.evaluator = ICDEvaluator(self.dataset_config)

        # Save dir
        self.save_dir = os.path.join(args.save_dir, args.exp_name)
        os.makedirs(self.save_dir, exist_ok=True)

        # CSV log
        self.csv_path = os.path.join(self.save_dir, 'train_log.csv')
        self.csv_file = open(self.csv_path, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        if os.path.getsize(self.csv_path) == 0:
            self.csv_writer.writerow([
                'epoch', 'train_loss', 'val_loss',
                'mAP@0.5', 'mAP@0.75', 'mAP@[.5:.95]',
                'state_acc', 'target_miou', 'target_acc',
                *[f'iou_{n}' for n in self.dataset_config.get('pixel_target_names', ['c0','c1','c2','c3','c4','c5'])],
                'inst_state', 'inst_bbox', 'inst_giou',
                'pixel_ce', 'pixel_dice', 'lr', 'time_s', 'icd_mAP', 'icd_mAP_50', 'icd_mAP_75'
            ])

        # Best loss
        self.best_loss = float('inf')
        self.current_epoch = 0

        # Resume
        if args.resume and os.path.exists(args.resume):
            self._load_checkpoint(args.resume)

        print(f"\nModel parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")

    def _build_model_config(self):
        cfg = dict(self.dataset_config)
        cfg['backbone'] = {
            'variant': self.args.backbone,
            'pretrained': True,
            'pretrained_path': self.args.pretrained_weight_path,
        }
        cfg['hidden_dim'] = self.args.hidden_dim
        cfg['w_pixel'] = self.args.w_pixel
        cfg['w_state'] = self.args.w_state
        cfg['w_bbox'] = self.args.w_bbox
        cfg['w_giou'] = self.args.w_giou
        cfg['w_consistency'] = self.args.w_consistency
        cfg['active_heads'] = self.dataset_config.get('active_heads', {})
        cfg['pixel_class_weights'] = self.args.pixel_class_weights
        cfg['inst_state_weights'] = self.args.inst_state_weights
        cfg['w_state_pixel'] = self.args.w_state_pixel
        cfg['w_change'] = self.args.w_change
        return cfg

    def _print_model_summary(self):
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total params: {total / 1e6:.2f}M | Trainable: {trainable / 1e6:.2f}M")

    def train(self):
        start_epoch = self.current_epoch

        # Stage 1: Freeze backbone
        if self.args.freeze_backbone_epochs > 0:
            print(f"\nStage 1: Freezing backbone for {self.args.freeze_backbone_epochs} epochs")
            self.model.freeze_backbone()

        for epoch in range(start_epoch, self.args.max_epochs):
            self.current_epoch = epoch

            # Unfreeze
            if epoch == self.args.freeze_backbone_epochs:
                print(f"\nStage 2: Unfreezing backbone (ratio={self.args.unfreeze_ratio})")
                self.model.unfreeze_backbone(self.args.unfreeze_ratio)

            # Train
            train_loss, train_dict = self.train_epoch(epoch)

            # Validate
            val_loss, val_results = self.validate(epoch)

            # LR step
            self.scheduler.step()

            # Save best
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss

            if (epoch + 1) % self.args.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)

            # CSV
            lr = self.optimizer.param_groups[0]['lr']
            target_ious = val_results.get('target_iou', [0]*6)
            row = [
                    epoch + 1, f'{train_loss:.4f}', f'{val_loss:.4f}',
                    f'{val_results.get("instance_mAP@0.5", 0):.4f}',
                    f'{val_results.get("instance_mAP@0.75", 0):.4f}',
                    f'{val_results.get("instance_mAP@[0.5:0.95]", 0):.4f}',
                    f'{val_results.get("state_accuracy", 0):.4f}',
                    f'{val_results.get("target_miou", 0):.4f}',
                    f'{val_results.get("target_accuracy", 0):.4f}',
            ]
            for i in range(6):
                row.append(f'{target_ious[i]:.4f}' if i < len(target_ious) else '0.0000')
            row.extend([
                    f'{train_dict.get("inst_state", 0):.4f}',
                    f'{train_dict.get("inst_bbox", 0):.4f}',
                    f'{train_dict.get("inst_giou", 0):.4f}',
                    f'{train_dict.get("pixel_ce", 0):.4f}',
                    f'{train_dict.get("pixel_dice", 0):.4f}',
                    f'{lr:.6f}', '',
                    f'{val_results.get("icd_mAP", 0):.4f}',
                    f'{val_results.get("icd_mAP_50", 0):.4f}',
                    f'{val_results.get("icd_mAP_75", 0):.4f}'
            ])
            print(f"Epoch {epoch+1}/{self.args.max_epochs} - "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"mAP@0.5: {val_results.get('instance_mAP@0.5', 0):.4f} | "
                  f"StateAcc: {val_results.get('state_accuracy', 0):.4f} | "
                  f"mIoU: {val_results.get('target_miou', 0):.4f}")

        print(f"\nTraining completed! Best loss: {self.best_loss:.4f}")
        self.csv_file.close()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0
        loss_accum = {}

        self.evaluator.reset()

        pbar = tqdm(self.train_loader,
                    desc=f'Epoch {epoch+1}/{self.args.max_epochs}',
                    leave=True, ncols=130, mininterval=0.5, file=sys.stdout)

        for batch_idx, batch in enumerate(pbar):
            img_t1 = batch['img_t1'].to(self.device)
            img_t2 = batch['img_t2'].to(self.device)
            targets = self._prepare_targets(batch)

            if self.args.use_amp:
                with autocast('cuda'):
                    predictions, loss, loss_dict = self.model(img_t1, img_t2, targets)
            else:
                predictions, loss, loss_dict = self.model(img_t1, img_t2, targets)

            scaled_loss = loss / self.args.grad_accum
            if self.args.use_amp:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if (batch_idx + 1) % self.args.grad_accum == 0:
                if self.args.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1
            for k, v in loss_dict.items():
                if isinstance(v, (int, float)):
                    loss_accum[k] = loss_accum.get(k, 0) + v

            if targets:
                self.evaluator.update(predictions, targets, loss_dict)

            # Show running loss + state acc in progress bar
            avg = total_loss / num_batches
            px_state = loss_accum.get('loss_state_pixel', 0) / max(num_batches, 1)
            pbar.set_postfix(loss=f'{avg:.4f}', px_state=f'{px_state:.4f}')

        avg_loss = total_loss / max(num_batches, 1)
        avg_loss_dict = {k: v / max(num_batches, 1) for k, v in loss_accum.items()}

        eval_results = self.evaluator.compute()
        print(self.evaluator.format_results(eval_results))
        self.evaluator.reset()

        return avg_loss, avg_loss_dict

    def validate(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                img_t1 = batch['img_t1'].to(self.device)
                img_t2 = batch['img_t2'].to(self.device)
                targets = self._prepare_targets(batch)

                predictions, loss, loss_dict = self.model(img_t1, img_t2, targets)
                total_loss += loss.item()
                num_batches += 1

                self.evaluator.update(predictions, targets, loss_dict)

        avg_loss = total_loss / max(num_batches, 1)
        eval_results = self.evaluator.compute()
        print(f'\n--- Validation Epoch {epoch+1} ---')
        print(f'Val Loss: {avg_loss:.4f}')
        print(self.evaluator.format_results(eval_results))
        print('---')

        # ICD protocol evaluation (every 10 epochs)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            from evaluation.icd_protocol import evaluate_icd
            target_classes = [i for i in range(1, self.dataset_config.get('num_pixel_targets', 6))]
            icd_results, icd_per_class = evaluate_icd(
                data_dir=self.args.data_dir, split='val',
                model=self.model, device=self.device,
                config=self.dataset_config,
                target_classes=target_classes, max_samples=300,
            )
            eval_results['icd_mAP'] = icd_results.get('mAP', 0)
            eval_results['icd_mAP_50'] = icd_results.get('mAP_50', 0)
            eval_results['icd_mAP_75'] = icd_results.get('mAP_75', 0)

        self.model.train()
        return avg_loss, eval_results

    def save_checkpoint(self, epoch, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'args': self.args
        }
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, path)

        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)

    def _load_checkpoint(self, path):
        print(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.current_epoch = ckpt['epoch'] + 1
        self.best_loss = ckpt.get('loss', float('inf'))
        if self.scaler and 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])

    def _prepare_targets(self, batch):
        targets = {}

        if 'instance_boxes' in batch:
            instance_labels = []
            instance_classes = batch.get('instance_classes')
            instance_states = batch.get('instance_states')
            for i, boxes in enumerate(batch['instance_boxes']):
                if boxes is not None and len(boxes) > 0:
                    labels = (instance_classes[i].to(self.device)
                              if instance_classes is not None and instance_classes[i] is not None
                              else torch.zeros(len(boxes), dtype=torch.long, device=self.device))
                    inst_dict = {'labels': labels, 'boxes': boxes.to(self.device)}
                    if instance_states is not None and instance_states[i] is not None:
                        inst_dict['states'] = instance_states[i].to(self.device)
                    instance_labels.append(inst_dict)
                else:
                    instance_labels.append({
                        'labels': torch.zeros(0, dtype=torch.long, device=self.device),
                        'boxes': torch.zeros(0, 4, device=self.device)
                    })
            targets['instance_labels'] = instance_labels

        # Pixel targets
        semantic_classes = self.dataset_config.get('branch_routing', {}).get('semantic', [])
        if 'label' in batch and len(semantic_classes) > 0:
            label = batch['label'].to(self.device)
            if label.dim() == 3:
                label_s = label.unsqueeze(1).float()
                label_s = nn.functional.interpolate(
                    label_s, scale_factor=0.25, mode='nearest'
                ).squeeze(1).long()
                targets['target_gt'] = label_s

        if 'change_state' in batch:
            cs = batch['change_state'].to(self.device)
            if cs.dim() == 3:
                cs_s = cs.unsqueeze(1).float()
                cs_s = nn.functional.interpolate(
                    cs_s, scale_factor=0.25, mode='nearest'
                ).squeeze(1).long()
                targets['change_state'] = cs_s

        # Derive state_gt: use change_state directly if state_decode_map exists,
        # otherwise derive from transition map
        if self.dataset_config.get('state_decode_map') is not None:
            # 0617final-style: change_state already encodes the correct state
            if 'change_state' in targets:
                targets['state_gt'] = targets['change_state']
        else:
            # SECOND-style: derive from transition map
            NUM_CLASSES = 6
            if 'transition' in batch:
                trans = batch['transition'].to(self.device)
                if trans.dim() == 3:
                    trans_s = trans.unsqueeze(1).float()
                    trans_s = nn.functional.interpolate(
                        trans_s, scale_factor=0.25, mode='nearest'
                    ).squeeze(1).long()
                else:
                    trans_s = trans
                pre_cls = trans_s // NUM_CLASSES
                post_cls = trans_s % NUM_CLASSES
                state_gt = torch.full_like(trans_s, 255)
                valid = (trans_s != 255)
                state_gt[valid & (pre_cls == post_cls)] = 0
                state_gt[valid & (pre_cls < post_cls)] = 1
                state_gt[valid & (pre_cls > post_cls)] = 2
                targets['state_gt'] = state_gt

        return targets


def main():
    args = parse_args()
    trainer = TrainerV7(args)
    trainer.train()


if __name__ == '__main__':
    main()







