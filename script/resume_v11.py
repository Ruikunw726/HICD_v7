"""Guarded continuation of the existing v11 experiment; no method changes."""
import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from train_full_v7 import TrainerV7
from models import build_hicd_v7


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite_tree(value):
    if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
        require(bool(torch.isfinite(value).all()), 'Non-finite checkpoint tensor')
    elif isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            finite_tree(item)


def load_checkpoint(path):
    require(path.is_file(), f'Resume checkpoint missing: {path}')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('epoch', 'model_state_dict', 'optimizer_state_dict',
                'scheduler_state_dict', 'scaler_state_dict', 'loss', 'args'):
        require(key in ckpt, f'Checkpoint missing {key}')
    require(math.isfinite(ckpt['loss']), 'Non-finite saved loss')
    return ckpt


def restore_dynamic_buffers(model, state):
    # CrossEntropyLoss starts with weight=None; legacy forward created this buffer.
    key = 'loss_fn.state_ce.weight'
    if key in state and model.loss_fn.state_ce.weight is None:
        require(state[key].shape == (model.config.get('num_states', 7),), 'Unexpected state CE weight shape')
        model.loss_fn.state_ce.weight = state[key].to(next(model.parameters()).device).clone()


def prepare(ckpt, checkpoint):
    args = argparse.Namespace(**vars(ckpt['args']))
    require(args.exp_name == 'v11_no_t4' and args.dataset == '0617final', 'Not original v11')
    require(args.max_epochs == 50 and args.use_amp, 'Unexpected v11 training budget/AMP')
    args.resume = str(checkpoint)
    require(Path(args.pretrained_weight_path).is_file(), 'Local pretrained weights missing')
    config_path = ROOT / args.config_dir / f'{args.dataset}.yaml'
    config = yaml.safe_load(config_path.read_text())
    log = ROOT / args.save_dir / args.exp_name / 'train.log'
    old_config = next(ast.literal_eval(line.split('Dataset config: ', 1)[1])
                      for line in log.read_text().splitlines() if line.startswith('Dataset config: '))
    require(config == old_config, 'Current YAML differs from original v11 logged config')
    paths = [config_path, Path(args.pretrained_weight_path)]
    counts = {}
    for split in ('train', 'val', 'test'):
        names = Path(args.data_dir) / split / 'names.txt'
        require(names.is_file(), f'Missing data list: {names}')
        paths.append(names)
        counts[split] = len([line for line in names.read_text().splitlines() if line.strip()])
    paths += [p for folder in ('models', 'datasets', 'evaluation')
              for p in (ROOT / folder).rglob('*.py')]
    paths += [ROOT / 'script/train_full_v7.py', Path(__file__).resolve(), ROOT / 'launch_v11_resume.sh']
    hashes = {str(p.resolve()): digest(p) for p in sorted(set(paths))}
    csv_path = log.parent / 'train_log.csv'
    with csv_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(rows and int(rows[-1]['epoch']) == ckpt['epoch'] + 1,
            'CSV last epoch differs from checkpoint; refusing duplicate/inconsistent continuation')
    require(ckpt['epoch'] + 1 < args.max_epochs, 'Training already at target epoch')
    return args, config, hashes, counts, rows


def verify(checkpoint, audit):
    torch.set_num_threads(4)
    ckpt = load_checkpoint(checkpoint)
    args, config, hashes, counts, rows = prepare(ckpt, checkpoint)
    finite_tree(ckpt['model_state_dict'])
    finite_tree(ckpt['optimizer_state_dict'])
    builder = TrainerV7.__new__(TrainerV7)
    builder.args, builder.dataset_config = args, config
    model = build_hicd_v7(builder._build_model_config())
    restore_dynamic_buffers(model, ckpt['model_state_dict'])
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    optimizer = torch.optim.AdamW(model.get_param_groups(lr=args.learning_rate, weight_decay=args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs,
                                                         eta_min=args.learning_rate * 0.01)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    require(optimizer.state, 'Empty optimizer state')
    require(scheduler.last_epoch == ckpt['epoch'] + 1, 'Scheduler epoch mismatch')
    require(torch.cuda.is_available(), 'CUDA unavailable')
    scaler = torch.amp.GradScaler('cuda')
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    require(scaler.state_dict() == ckpt['scaler_state_dict'], 'AMP state mismatch')
    best = load_checkpoint(checkpoint.parent / 'best_model.pth')
    best_loss = min(ckpt['loss'], best['loss'], min(float(row['val_loss']) for row in rows))
    manifest = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                    args=vars(args), file_sha256=hashes, list_counts=counts,
                    epoch_zero_based=ckpt['epoch'], next_display_epoch=ckpt['epoch'] + 2,
                    best_loss=best_loss, optimizer_groups=len(optimizer.param_groups),
                    optimizer_states=len(optimizer.state), scheduler_state=scheduler.state_dict(),
                    learning_rates=[group['lr'] for group in optimizer.param_groups],
                    scaler_state=scaler.state_dict(), torch=torch.__version__,
                    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                    limitations=['Legacy checkpoint lacks RNG, sampler, adaptive weight and running IoU state',
                                 'Current code/data hashes recorded; historical hashes were not saved'])
    audit.mkdir(parents=True, exist_ok=False)
    for name in ('train.log', 'train_log.csv', 'best_model.pth'):
        shutil.copy2(checkpoint.parent / name, audit / name)
    shutil.copy2(checkpoint, audit / checkpoint.name)
    shutil.copy2(ROOT / args.config_dir / f'{args.dataset}.yaml', audit / '0617final.yaml')
    for split in counts:
        shutil.copy2(Path(args.data_dir) / split / 'names.txt', audit / f'{split}_names.txt')
    for rel, command in [('git_status.txt', ['git', 'status', '--short']),
                         ('git_head.txt', ['git', 'rev-parse', 'HEAD']),
                         ('git_diff.patch', ['git', 'diff', '--binary', 'HEAD'])]:
        (audit / rel).write_bytes(subprocess.check_output(command, cwd=ROOT))
    # Preserve exact effective implementation, including untracked source files.
    for source in hashes:
        p = Path(source)
        if p.is_relative_to(ROOT) and p.suffix in ('.py', '.sh'):
            target = audit / 'source' / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
    (audit / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('PREFLIGHT_OK', json.dumps({key: manifest[key] for key in (
        'next_display_epoch', 'list_counts', 'optimizer_states', 'learning_rates', 'scaler_state', 'best_loss')}), flush=True)


class GuardedTrainer(TrainerV7):
    def _load_checkpoint(self, path):
        ckpt = load_checkpoint(Path(path))
        restore_dynamic_buffers(self.model, ckpt['model_state_dict'])
        self.model.load_state_dict(ckpt['model_state_dict'], strict=True)
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        self.current_epoch = ckpt['epoch'] + 1
        self.best_loss = self.resume_manifest['best_loss']
        require(self.scheduler.state_dict() == self.resume_manifest['scheduler_state'], 'Scheduler restoration mismatch')
        require(self.scaler.state_dict() == self.resume_manifest['scaler_state'], 'AMP restoration mismatch')
        require([g['lr'] for g in self.optimizer.param_groups] == self.resume_manifest['learning_rates'], 'LR restoration mismatch')
        require(len(self.optimizer.state) == self.resume_manifest['optimizer_states'], 'Optimizer restoration mismatch')
        print(f'RESUME_OK epoch={self.current_epoch + 1}/50 strict_keys=True optimizer_states={len(self.optimizer.state)} '
              f'lr={[g["lr"] for g in self.optimizer.param_groups]} scaler={self.scaler.state_dict()} best_loss={self.best_loss}', flush=True)


def run(checkpoint, audit):
    manifest = json.loads((audit / 'manifest.json').read_text())
    require(str(checkpoint) == manifest['checkpoint'], 'Wrong resume checkpoint')
    require(digest(checkpoint) == manifest['checkpoint_sha256'], 'Checkpoint changed since verification')
    for path, expected in manifest['file_sha256'].items():
        require(digest(path) == expected, f'Input changed since verification: {path}')
    require(torch.cuda.is_available(), 'CUDA unavailable')
    torch.ones(1, device='cuda').sum().item()
    args = argparse.Namespace(**manifest['args'])
    (audit / 'python.pid').write_text(str(os.getpid()))
    GuardedTrainer.resume_manifest = manifest
    print('RESUME_LIMITATION: legacy RNG/sampler/adaptive loss history was not saved; original phase reset retained.', flush=True)
    trainer = GuardedTrainer(args)
    try:
        trainer.train()
    finally:
        if not trainer.csv_file.closed:
            trainer.csv_file.flush()
            trainer.csv_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['verify', 'run'], required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--audit-dir', required=True)
    options = parser.parse_args()
    os.chdir(ROOT)
    (verify if options.mode == 'verify' else run)(Path(options.checkpoint).resolve(), Path(options.audit_dir).resolve())
