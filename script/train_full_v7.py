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
    parser.add_argument('--backbone_type', type=str, default='lwganet', choices=['lwganet', 'dinov3'], help='Backbone type: lwganet or dinov3')
    parser.add_argument('--pretrained_weight_path', type=str, default=None, help='Auto-selected based on backbone_type if not set')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension for label structure')

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
    parser.add_argument('--w_consistency', type=float, default=0.3)
    parser.add_argument('--pixel_class_weights', type=float, nargs='+', default=None,
                        help='Instance state CE weights [no_change, increase, decrease]')
    parser.add_argument('--state_pixel_weights', type=float, nargs=7, default=None,
                        help='Class weights for pixel state CE loss [nochange,damaged,reduced,added,extended,replaced,none]')
    parser.add_argument('--w_state_pixel', type=float, default=1.0)
    parser.add_argument('--w_contrastive', type=float, default=0.05, help='Weight for contrastive label structure losses')
    parser.add_argument('--state_aware', action='store_true', help='Enable state-aware sampling for rare state classes')
    parser.add_argument('--rare_class_oversample', action='store_true', help='Oversample rare classes (Bridge/Dock/Shelter)')
    parser.add_argument('--rare_class_boost', type=float, default=50.0, help='Weight boost for rare class samples')

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
            img_size=args.img_size, boxes_dir=args.boxes_dir,
            state_aware=args.state_aware,
            rare_class_oversample=args.rare_class_oversample,
            rare_class_boost=args.rare_class_boost
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
            pixel_names = self.dataset_config.get('pixel_target_names', [])
            header = [
                'epoch', 'train_loss', 'val_loss',
                # BDD Protocol (main metrics)
                'F_loc', 'F_clf', 'F_overall', 'F_macro',
                # Four-task F1
                'T1_f1', 'T1_sek', 'T2_f1', 'T2_sek',
                'T3_f1', 'T3_sek', 'T4_f1', 'T4_sek',
                'bdd_f1', 'bdd_sek',
                # Per-class F1
            ] + [f'f1_{n.lower().replace(" ", "_")}' for n in pixel_names] + [
                # Loss breakdown
                'pixel_ce', 'pixel_dice', 'pixel_state_ce', 'pixel_state_dice',
                'bg_suppress', 'loss_contrastive',
                'lr', 'time_s',
            ]
            self.csv_writer.writerow(header)

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
            'type': self.args.backbone_type,
            'variant': self.args.backbone if self.args.backbone_type != 'dinov3' else 'dinov3_base',
            'pretrained': True,
            'pretrained_path': self.args.pretrained_weight_path,
        }
        cfg['hidden_dim'] = self.args.hidden_dim
        cfg['embed_dim'] = self.args.embed_dim
        cfg['w_pixel'] = self.args.w_pixel
        cfg['w_state'] = self.args.w_state
        cfg['active_heads'] = self.dataset_config.get('active_heads', {})
        cfg['pixel_class_weights'] = self.args.pixel_class_weights if self.args.pixel_class_weights is not None else self.dataset_config.get('pixel_class_weights', None)
        cfg['state_pixel_weights'] = self.args.state_pixel_weights if self.args.state_pixel_weights is not None else self.dataset_config.get('state_pixel_weights', None)
        cfg['w_contrastive'] = self.args.w_contrastive
        return cfg

    def _print_model_summary(self):
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total params: {total / 1e6:.2f}M | Trainable: {trainable / 1e6:.2f}M")

    def train(self):
        start_epoch = self.current_epoch

        # Stage 1: Freeze backbone (only if starting from scratch)
        if self.args.freeze_backbone_epochs > 0 and start_epoch < self.args.freeze_backbone_epochs:
            print(f"\nStage 1: Freezing backbone for {self.args.freeze_backbone_epochs} epochs")
            if self.args.backbone_type == 'dinov3':
                print('DINOv3: early blocks already frozen, freezing remaining trainable blocks')
                for p in self.model.backbone.parameters():
                    p.requires_grad = False
            else:
                self.model.freeze_backbone()
            # Phase 1: rare-class focused weights
            if hasattr(self.model, 'loss_fn') and hasattr(self.model.loss_fn, 'set_phase'):
                self.model.loss_fn.set_phase(1)
        elif start_epoch >= self.args.freeze_backbone_epochs:
            print(f"\nResuming after freeze phase (start_epoch={start_epoch}), backbone already unfrozen")
            if hasattr(self.model, 'loss_fn') and hasattr(self.model.loss_fn, 'set_phase'):
                self.model.loss_fn.set_phase(2, self.dataset_config)

        t_start = time.time()
        for epoch in range(start_epoch, self.args.max_epochs):
            self.current_epoch = epoch
            self.model.config['_current_epoch'] = epoch

            # Unfreeze
            if epoch == self.args.freeze_backbone_epochs:
                print(f"\nStage 2: Unfreezing backbone (ratio={self.args.unfreeze_ratio})")
                if self.args.backbone_type == 'dinov3':
                    blocks = list(self.model.backbone.blocks)
                    num_to_unfreeze = int(len(blocks) * self.args.unfreeze_ratio)
                    for block in blocks[-num_to_unfreeze:]:
                        for p in block.parameters():
                            p.requires_grad = True
                    for proj in self.model.backbone.proj_heads:
                        for p in proj.parameters():
                            p.requires_grad = True
                    print(f'Unfroze last {num_to_unfreeze} blocks + projection heads')
                else:
                    self.model.unfreeze_backbone(self.args.unfreeze_ratio)
                # Phase 2: restore normal weights
                if hasattr(self.model, 'loss_fn') and hasattr(self.model.loss_fn, 'set_phase'):
                    self.model.loss_fn.set_phase(2, self.dataset_config)
            train_loss, train_dict = self.train_epoch(epoch)

            # Validate
            val_loss, val_results = self.validate(epoch)

            # Update target mIoU for refinement warmup scheduling
            current_tmiou = val_results.get('target_miou', 0)
            if hasattr(self.model, 'loss_fn') and hasattr(self.model.loss_fn, 'set_target_miou'):
                self.model.loss_fn.set_target_miou(current_tmiou)

            # LR step
            self.scheduler.step()

            # Save best
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss

            if (epoch + 1) % self.args.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)

            # CSV
            time_s = time.time() - t_start
            lr = self.optimizer.param_groups[0]['lr']
            row = [
                epoch + 1, f'{train_loss:.4f}', f'{val_loss:.4f}',
                # BDD Protocol
                f'{val_results.get("F_loc", 0):.4f}',
                f'{val_results.get("F_clf", 0):.4f}',
                f'{val_results.get("F_overall", 0):.4f}',
                f'{val_results.get("F_macro", 0):.4f}',
                # Four-task F1
                f'{val_results.get("T1_Building_f1", 0):.4f}',
                f'{val_results.get("T1_Building_sek", 0):.4f}',
                f'{val_results.get("T2_Surface_f1", 0):.4f}',
                f'{val_results.get("T2_Surface_sek", 0):.4f}',
                f'{val_results.get("T3_Mobile_f1", 0):.4f}',
                f'{val_results.get("T3_Mobile_sek", 0):.4f}',
                f'{val_results.get("T4_Facility_f1", 0):.4f}',
                f'{val_results.get("T4_Facility_sek", 0):.4f}',
                f'{val_results.get("bdd_f1", 0):.4f}',
                f'{val_results.get("bdd_sek", 0):.4f}',
            ]
            # Per-class F1
            pixel_names = self.dataset_config.get('pixel_target_names', [])
            for name in pixel_names:
                name_lower = name.lower().replace(' ', '_')
                row.append(f'{val_results.get(f"prf_{name_lower}_f1", 0):.4f}')
            # Loss
            row.extend([
                f'{train_dict.get("pixel_ce", 0):.4f}',
                f'{train_dict.get("pixel_dice", 0):.4f}',
                f'{train_dict.get("pixel_state_ce", 0):.4f}',
                f'{train_dict.get("pixel_state_dice", 0):.4f}',
                f'{train_dict.get("bg_suppress", 0):.4f}',
                f'{train_dict.get("loss_contrastive", 0):.6f}',
                f'{lr:.6f}', f'{time_s:.1f}'
            ])
            self.csv_writer.writerow(row)
            self.csv_file.flush()

            print(f"Epoch {epoch+1}/{self.args.max_epochs} - "
                  f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"F_overall: {val_results.get('F_overall', 0):.4f} | "
                  f"F_loc: {val_results.get('F_loc', 0):.4f} | "
                  f"F_clf: {val_results.get('F_clf', 0):.4f} | "
                  f"SeK: {val_results.get('bdd_sek', 0):.4f}")

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

            # Show running loss in progress bar
            avg = total_loss / num_batches
            st_ce = loss_accum.get('pixel_state_ce', 0) / max(num_batches, 1)
            pbar.set_postfix(loss=f'{avg:.4f}', sCE=f'{st_ce:.4f}')

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

        # Update adaptive class weights based on validation IoU
        if hasattr(self.model, "loss_fn") and hasattr(self.model.loss_fn, "update_adaptive_weights"):
            iou_per_class = eval_results.get("target_iou", None)
            if iou_per_class is not None:
                state_iou = eval_results.get("state_iou", None)
                self.model.loss_fn.update_adaptive_weights(iou_per_class, per_state_iou=state_iou)
                w = self.model.loss_fn.get_class_weights()
                wlist = [round(x, 3) for x in w.tolist()]
                print(f"  Adaptive target weights: {wlist}")
                if hasattr(self.model.loss_fn, "state_weighter"):
                    sw = self.model.loss_fn.state_weighter.get_weights()
                    swlist = [round(x, 3) for x in sw.tolist()]
                    print(f"  Adaptive state weights: {swlist}")
        print('---')

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
        self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
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
                targets['state_gt'] = cs_s  # state_gt = change_state (decoded by dataset)

        return targets


def main():
    args = parse_args()
    # Auto-resolve pretrained weight path based on backbone type
    if args.pretrained_weight_path is None:
        if args.backbone_type == 'dinov3':
            args.pretrained_weight_path = '/mnt/f/mambacd/home/HICD_v7/weights/model.safetensors'
        else:
            args.pretrained_weight_path = '/mnt/f/mambacd/home/HICD_v7/backbone_weights/lwganet_l1.pth'
        print(f'Auto-selected pretrained weights: {args.pretrained_weight_path}')
    trainer = TrainerV7(args)
    trainer.train()


if __name__ == '__main__':
    main()








