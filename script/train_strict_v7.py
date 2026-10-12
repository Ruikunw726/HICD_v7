"""Fresh HICD v7 training with explicit strict target+original-state evaluation.

Model/optimizer/data/loss mechanics follow train_full_v7. Legacy BDD is never
used for reporting or best selection. IoU feedback preserves the old loss policy.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import yaml
from torch.amp import GradScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_full_v7 import TrainerV7, parse_args
from models import build_hicd_v7
from datasets import create_dataloader
from evaluation.strict_joint_evaluator import StrictJointModelEvaluator
from evaluation.strict_joint_protocol import PROTOCOL_VERSION, TARGET_NAMES, TASKS
from evaluation.farmland_semantics import recompute_report, POLICY_VERSION
from evaluation.strict_joint_reporting import metric_number, format_csv_row

BEST_CRITERION = 'validation.all_foreground.strict_joint_f1_micro'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def prepare_run(args):
    if args.resume is not None:
        raise ValueError('Fresh training requires no resume')
    run = ROOT / args.save_dir / args.exp_name
    if run.exists():
        raise FileExistsError(f'Refusing to overwrite existing experiment: {run}')
    config_path = ROOT / args.config_dir / f'{args.dataset}.yaml'
    config = yaml.safe_load(config_path.read_text())
    # Confirm protocol compatibility before creating any training artifact.
    TrainingEvaluator(config)
    sources = [config_path, Path(args.pretrained_weight_path)]
    counts = {}
    for split in ('train', 'val', 'test'):
        path = Path(args.data_dir) / split / 'names.txt'
        counts[split] = len([line for line in path.read_text().splitlines() if line.strip()])
        sources.append(path)
    metadata = {}
    split_manifest_path = Path(args.data_dir) / 'split_manifest.json'
    split_manifest = json.loads(split_manifest_path.read_text()) if split_manifest_path.is_file() else None
    extra_data_files = {}
    if split_manifest is not None:
        if counts != split_manifest['published_counts']:
            raise ValueError('Clean data split manifest counts disagree with names')
        sources.append(split_manifest_path)
        extra_data_files[split_manifest_path] = Path('data/split_manifest.json')
        for split in counts:
            catalog = Path(args.data_dir) / split / 'resolved_samples.json'
            if sha256(catalog) != split_manifest['catalog_sha256'][split]:
                raise ValueError('Clean data catalog digest mismatch')
            sources.append(catalog)
            extra_data_files[catalog] = Path('data') / split / 'resolved_samples.json'
    for name in ('scene_state_info.json', 'instances.json'):
        path = Path(args.data_dir) / name
        metadata[name] = {'exists': path.is_file()}
        if path.is_file():
            sources.append(path)
    for folder in ('models', 'datasets', 'evaluation'):
        sources += list((ROOT / folder).rglob('*.py'))
    sources += [ROOT / 'script/train_full_v7.py', Path(__file__).resolve(), ROOT / 'launch_v12_strict.sh']
    hashes = {str(path.resolve()): sha256(path) for path in sorted(set(sources))}
    gt_ids = sorted(set(config['label_decode_map'].values()) - {255, 15})
    identity = hashlib.sha256(json.dumps({'args': vars(args), 'hashes': hashes}, sort_keys=True).encode()).hexdigest()
    manifest = {'protocol_version': PROTOCOL_VERSION, 'run_identity': identity, 'args': vars(args),
                'data_split_identity': split_manifest['split_identity'] if split_manifest else None,
                'prediction_semantics_version': config.get('prediction_semantics_version', 'raw_state_logits'),
                'file_sha256': hashes, 'list_counts': counts, 'metadata': metadata,
                'fresh_initialization': {'task_checkpoint': None, 'optimizer': 'new AdamW',
                                         'scheduler': 'new CosineAnnealingLR', 'AMP': 'new GradScaler', 'epoch': 0},
                'best_selection_criterion': BEST_CRITERION,
                'secondary_checkpoint_criterion': 'validation.all_foreground.strict_joint_f1_macro',
                'spatial_policy': {'input_image_hw': [args.img_size, args.img_size],
                                   'prepared_gt_hw': [args.img_size // 4, args.img_size // 4],
                                   'state_head': 'raw state_logits', 'GT_resampling': 'nearest, scale_factor=0.25'},
                'current_scope': {'configured_GT_foreground_ids': gt_ids,
                                  'configured_GT_foreground_names': [TARGET_NAMES[i] for i in gt_ids],
                                  'T4_configured_GT_names': [TARGET_NAMES[i] for i in TASKS['T4'] if i in gt_ids],
                                  'T4_ignored_GT_names': [TARGET_NAMES[i] for i in TASKS['T4'] if i not in gt_ids],
                                  'all_foreground_bins_remain_scored_for_false_positives': True}}
    run.mkdir(parents=True)
    (run / 'metrics').mkdir()
    for path in sources:
        if path.is_relative_to(ROOT) and path.suffix in ('.py', '.sh'):
            destination = run / 'snapshot/source' / path.relative_to(ROOT)
        elif path == config_path:
            destination = run / 'snapshot' / config_path.name
        elif path.name == 'names.txt':
            destination = run / 'snapshot' / f'{path.parent.name}_names.txt'
        elif path.name in metadata:
            destination = run / 'snapshot' / path.name
        elif path in extra_data_files:
            destination = run / 'snapshot' / extra_data_files[path]
        else:
            continue  # Pretrained backbone is hashed and referenced, not recopied.
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    for name, command in [('git_status.txt', ['git', 'status', '--short']),
                          ('git_head.txt', ['git', 'rev-parse', 'HEAD']),
                          ('git_diff.patch', ['git', 'diff', '--binary', 'HEAD']),
                          ('resources.txt', ['nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu', '--format=csv,noheader'])]:
        (run / name).write_bytes(subprocess.check_output(command, cwd=ROOT))
    atomic_json(run / 'manifest.json', manifest)
    print(f'FRESH_RUN_PREPARED {run} identity={identity} configured_T4_GT={manifest["current_scope"]["T4_configured_GT_names"]}', flush=True)


class TrainingEvaluator(StrictJointModelEvaluator):
    """04's strict evaluator plus streaming *loss feedback*, not a second protocol."""
    def __init__(self, config):
        self.config = config
        super().__init__(config, state_head='raw')

    def reset(self):
        super().reset()
        self.feedback_target_cm = np.zeros((16, 16), dtype=np.int64)
        self.feedback_state_cm = np.zeros((7, 7), dtype=np.int64)

    def update(self, predictions, targets, loss_dict=None):
        super().update(predictions, targets, loss_dict)
        # Prepared GT is already downsampled by inherited _prepare_targets.
        pt = predictions['target_logits'].detach().argmax(1).cpu().numpy()
        ps = predictions['state_logits'].detach().argmax(1).cpu().numpy()
        gt = targets['target_gt'].detach().cpu().numpy()
        gs = targets['change_state'].detach().cpu().numpy()
        if gt.shape != pt.shape:
            raise RuntimeError('Training feedback requires the original aligned output grid')
        valid_t = gt != 255
        self.feedback_target_cm += np.bincount(
            (gt[valid_t] * 16 + pt[valid_t]).reshape(-1), minlength=256).reshape(16, 16)
        # Match the old adaptive loss feedback mask, including background none.
        valid_s = valid_t & (gs != 255)
        for target in self.config.get('state_ignore_targets', []):
            valid_s &= gt != target
        self.feedback_state_cm += np.bincount(
            (gs[valid_s] * 7 + ps[valid_s]).reshape(-1), minlength=49).reshape(7, 7)

    @staticmethod
    def _iou(cm):
        intersection = np.diag(cm)
        union = cm.sum(0) + cm.sum(1) - intersection
        return intersection / np.maximum(union, 1), union > 0

    def compute(self):
        result = super().compute()
        if self.config.get('prediction_semantics_version') == POLICY_VERSION:
            result = recompute_report(result)
        tiou, present = self._iou(self.feedback_target_cm)
        siou, _ = self._iou(self.feedback_state_cm)
        # Keys consumed by inherited validate() and the old refinement schedule.
        result['target_iou'] = tiou.tolist()
        result['state_iou'] = siou.tolist()
        result['target_miou'] = float(tiou[present].mean()) if present.any() else 0.0
        result['loss_feedback_policy'] = {
            'purpose': 'legacy adaptive loss feedback only; not strict score selection',
            'target_mask': 'GT target != 255',
            'state_mask': 'GT target != 255, GT state != 255, configured state_ignore_targets',
            'state_ignore_targets': self.config.get('state_ignore_targets', []),
            'target_mean': 'union-positive classes including background, as original implementation',
        }
        return result

    @staticmethod
    def format_results(result):
        return StrictJointModelEvaluator.format_results(result)


def csv_fields():
    fields = ['epoch', 'train_loss', 'val_loss', 'strict_joint_precision_micro',
              'strict_joint_recall_micro', 'strict_joint_f1_micro', 'strict_joint_f1_macro',
              'strict_joint_tp', 'strict_joint_fp', 'strict_joint_fn', 'strict_joint_support',
              'best_selection_criterion', 'is_best', 'lr_groups_json', 'elapsed_s']
    for task in TASKS:
        fields += [f'{task}_strict_joint_f1_micro', f'{task}_strict_joint_f1_macro']
    for name in TARGET_NAMES[:-1]:
        fields += [f'{name}_strict_joint_{key}' for key in ('precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'support')]
    return fields


class StrictTrainer(TrainerV7):
    def __init__(self, args, smoke=False):
        if args.resume is not None:
            raise ValueError('Fresh strict experiment forbids --resume and old task checkpoints')
        self.args = args
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for this training budget')
        self.device = torch.device('cuda')
        self.save_dir = str(ROOT / args.save_dir / args.exp_name)
        run_dir = Path(self.save_dir)
        manifest = json.loads((run_dir / 'manifest.json').read_text())
        if vars(args) != manifest['args']:
            raise RuntimeError('Actual arguments differ from frozen experiment arguments')
        for name, metadata in manifest['metadata'].items():
            if (Path(args.data_dir) / name).is_file() != metadata['exists']:
                raise RuntimeError(f'Dataset metadata existence changed: {name}')
        for source, expected in manifest['file_sha256'].items():
            if sha256(source) != expected:
                raise RuntimeError(f'Frozen training input changed: {source}')
        self.run_identity = manifest['run_identity']
        self.data_split_identity = manifest.get('data_split_identity')
        self.dataset_config = yaml.safe_load((ROOT / args.config_dir / f'{args.dataset}.yaml').read_text())
        self.model = build_hicd_v7(self._build_model_config()).to(self.device)
        self._print_model_summary()
        if smoke:
            real_train = create_dataloader(data_dir=args.data_dir, config=self.dataset_config,
                                           split='train', batch_size=2, num_workers=0,
                                           img_size=args.img_size, state_aware=args.state_aware,
                                           boxes_dir=args.boxes_dir)
            iterator = iter(real_train)
            self.train_loader = [next(iterator), next(iterator)]
            allowed = {(target, self.dataset_config['state_decode_map'][raw])
                       for raw, target in self.dataset_config['label_decode_map'].items()
                       if target != 255 and self.dataset_config['state_decode_map'][raw] != 255}
            for batch in self.train_loader:
                pairs = torch.stack((batch['label'], batch['change_state']), dim=-1).reshape(-1, 2)
                pairs = pairs[(pairs[:, 0] != 255) & (pairs[:, 1] != 255)]
                observed = {tuple(pair) for pair in pairs.unique(dim=0).tolist()}
                if not observed.issubset(allowed):
                    raise RuntimeError(f'Augmented train target/state pairs are not source-decodable: {observed - allowed}')
            # Separate preflight process: actual val images + inherited target
            # preparation/forward/evaluator path, not a reported experiment score.
            real_val = create_dataloader(data_dir=args.data_dir, config=self.dataset_config,
                                         split='val', batch_size=2, num_workers=0,
                                         img_size=args.img_size, boxes_dir=args.boxes_dir)
            self.val_loader = [next(iter(real_val))]
        else:
            if (run_dir / 'train_log.csv').exists() or (run_dir / 'latest.pth').exists():
                raise RuntimeError('Existing training artifacts; refusing restart/overwrite')
            scenes = args.scenes.split(',') if args.scenes else None
            common = dict(data_dir=args.data_dir, config=self.dataset_config, batch_size=args.batch_size,
                          num_workers=args.num_workers, scenes=scenes, img_size=args.img_size, boxes_dir=args.boxes_dir)
            self.train_loader = create_dataloader(split='train', state_aware=args.state_aware,
                                                 rare_class_oversample=args.rare_class_oversample,
                                                 rare_class_boost=args.rare_class_boost, **common)
            self.val_loader = create_dataloader(split='val', **common)
        self.optimizer = torch.optim.AdamW(self.model.get_param_groups(lr=args.learning_rate, weight_decay=args.weight_decay))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.max_epochs,
                                                                   eta_min=args.learning_rate * 0.01)
        self.scaler = GradScaler('cuda') if args.use_amp else None
        self.evaluator = TrainingEvaluator(self.dataset_config)
        self.current_epoch = 0
        self.best_score = -math.inf
        self.best_epoch = None
        self.best_macro_score = -math.inf
        self.best_macro_epoch = None
        assert not self.optimizer.state and self.scheduler.last_epoch == 0
        self.model.register_forward_hook(self._finite_loss_hook)
        print(f'FRESH_START epoch=1/{args.max_epochs} task_checkpoint=None optimizer_states=0 '
              f'scheduler_last_epoch=0 pretrained_backbone={args.pretrained_weight_path} '
              f'protocol={PROTOCOL_VERSION} raw_state=True GT_nearest_scale=0.25 '
              f'semantics={self.dataset_config.get("prediction_semantics_version", "raw_state_logits")} '
              f'data_split={self.data_split_identity} scheduler_T_max={args.max_epochs}', flush=True)

    @staticmethod
    def _finite_loss_hook(module, inputs, output):
        if not torch.isfinite(output[1]).all():
            raise RuntimeError('Non-finite model loss; refusing optimizer update')

    def _checkpoint(self, epoch, loss, score):
        weights = self.model.loss_fn
        return {'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                'loss': loss, 'strict_joint_score': score, 'best_score': self.best_score, 'best_epoch': self.best_epoch,
                'best_macro_score': self.best_macro_score, 'best_macro_epoch': self.best_macro_epoch,
                'best_selection_criterion': BEST_CRITERION, 'protocol_version': PROTOCOL_VERSION,
                'args': self.args, 'dataset_config': self.dataset_config, 'run_identity': self.run_identity,
                'data_split_identity': self.data_split_identity,
                'prediction_semantics_version': self.dataset_config.get('prediction_semantics_version', 'raw_state_logits'),
                'rng': {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
                        'cuda': torch.cuda.get_rng_state_all()},
                'loss_adaptive_state': {'target_weights': weights.adaptive_weighter.weights,
                                        'target_running_iou': weights.adaptive_weighter.running_iou,
                                        'state_weights': weights.state_weighter.weights,
                                        'state_running_iou': weights.state_weighter.running_iou,
                                        'current_target_miou': weights.current_target_miou},
                'resume_limitations': 'Epoch boundary only; dataloader worker RNG and in-flight batches not saved'}

    def train(self):
        # v11 freeze_backbone_epochs=0: preserve its phase2 and last-half unfreeze.
        if self.args.freeze_backbone_epochs != 0:
            raise ValueError('This fresh v12 entry preserves the v11 zero-freeze budget')
        self.model.loss_fn.set_phase(2, self.dataset_config)
        blocks = list(self.model.backbone.blocks)
        count = int(len(blocks) * self.args.unfreeze_ratio)
        if not 0.0 <= self.args.unfreeze_ratio <= 1.0:
            raise ValueError('unfreeze ratio must be between 0 and 1')
        for block in (blocks[-count:] if count else []):
            for parameter in block.parameters():
                parameter.requires_grad = True
        for projection in self.model.backbone.proj_heads:
            for parameter in projection.parameters():
                parameter.requires_grad = True
        self.model.loss_fn.set_phase(2, self.dataset_config)
        print('OPTIMIZER_COVERAGE_OK', self.model.assert_optimizer_coverage(self.optimizer.param_groups), flush=True)
        started = time.time()
        with (Path(self.save_dir) / 'train_log.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=csv_fields())
            writer.writeheader()
            stream.flush()
            for epoch in range(self.args.max_epochs):
                self.current_epoch = epoch
                self.model.config['_current_epoch'] = epoch
                train_loss, train_parts = self.train_epoch(epoch)
                train_metrics = self.evaluator.last_train_metrics
                val_loss, result = self.validate(epoch)
                self.model.loss_fn.set_target_miou(result['target_miou'])
                self.scheduler.step()
                score = result['strict_joint_f1']
                if score is None or not math.isfinite(score):
                    raise RuntimeError('No evaluable foreground validation strict score')
                is_best = score > self.best_score
                if is_best:
                    self.best_score, self.best_epoch = score, epoch + 1
                macro_score = result['strict_joint_macro_f1']
                if macro_score is None or not math.isfinite(macro_score):
                    raise RuntimeError('No evaluable validation joint macro score')
                is_macro_best = macro_score > self.best_macro_score
                if is_macro_best:
                    self.best_macro_score, self.best_macro_epoch = macro_score, epoch + 1
                artifact = {'epoch': epoch + 1, 'run_identity': self.run_identity,
                            'best_selection_criterion': BEST_CRITERION, 'best_score': self.best_score,
                            'best_macro_score': self.best_macro_score, 'best_macro_epoch': self.best_macro_epoch,
                            'train_loss': train_loss, 'train_loss_parts': train_parts,
                            'validation_loss': val_loss, 'train': train_metrics, 'validation': result}
                atomic_json(Path(self.save_dir) / 'metrics' / f'epoch_{epoch+1:03d}.json', artifact)
                micro = result['strict_joint_micro']
                row = dict(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                           strict_joint_precision_micro=micro['precision'], strict_joint_recall_micro=micro['recall'],
                           strict_joint_f1_micro=score, strict_joint_f1_macro=result['strict_joint_macro_f1'],
                           strict_joint_tp=micro['tp'], strict_joint_fp=micro['fp'], strict_joint_fn=micro['fn'],
                           strict_joint_support=micro['support'], best_selection_criterion=BEST_CRITERION,
                           is_best=is_best, lr_groups_json=json.dumps([g['lr'] for g in self.optimizer.param_groups]),
                           elapsed_s=time.time() - started)
                for task, metric in result['tasks'].items():
                    row[f'{task}_strict_joint_f1_micro'] = metric['strict_joint_f1']
                    row[f'{task}_strict_joint_f1_macro'] = metric['strict_joint_macro_f1']
                for metric in result['per_target'].values():
                    for key in ('precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'support'):
                        row[f"{metric['target_name']}_strict_joint_{key}"] = metric['strict_joint_micro'][key]
                writer.writerow(format_csv_row(row))
                stream.flush()
                os.fsync(stream.fileno())
                checkpoint = self._checkpoint(epoch, val_loss, score)
                atomic_checkpoint(Path(self.save_dir) / 'latest.pth', checkpoint)
                if is_best:
                    atomic_checkpoint(Path(self.save_dir) / 'best_model.pth', checkpoint)
                if is_macro_best:
                    atomic_checkpoint(Path(self.save_dir) / 'best_macro.pth', checkpoint)
                if (epoch + 1) % self.args.save_interval == 0:
                    atomic_checkpoint(Path(self.save_dir) / f'checkpoint_epoch_{epoch+1}.pth', checkpoint)
                print(f'STRICT_EPOCH {epoch+1}/{self.args.max_epochs} val_micro_F1={metric_number(score)} '
                      f'val_macro_F1={metric_number(result["strict_joint_macro_f1"])} '
                      f'best={metric_number(self.best_score)}', flush=True)
        print(f'TRAINING_COMPLETED best_strict_joint_f1={metric_number(self.best_score)} '
              f'best_epoch={self.best_epoch}', flush=True)

    def train_epoch(self, epoch):
        # Retain the inherited optimizer/AMP path while capturing the streaming report
        # before the base method resets its evaluator at the epoch boundary.
        original = self.evaluator.compute
        def capture():
            result = original()
            self.evaluator.last_train_metrics = result
            return result
        self.evaluator.compute = capture
        try:
            return super().train_epoch(epoch)
        finally:
            self.evaluator.compute = original


def main():
    smoke = '--smoke-test' in sys.argv
    if smoke:
        sys.argv.remove('--smoke-test')
    prepare = '--prepare-run' in sys.argv
    if prepare:
        sys.argv.remove('--prepare-run')
    args = parse_args()
    if args.backbone_type != 'dinov3' or args.resume is not None:
        raise ValueError('Expected fresh DINOv3 initialization, no --resume')
    if args.pretrained_weight_path is None:
        args.pretrained_weight_path = str(ROOT / 'weights/model.safetensors')
    if prepare:
        prepare_run(args)
        return
    run = ROOT / args.save_dir / args.exp_name
    trainer = StrictTrainer(args, smoke=smoke)
    if smoke:
        frequency_before = {name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()
                            if name.startswith(('freq_branch.', 'freq_fusion.'))}
        frequency_gradients = {}
        hooks = []
        for name, parameter in trainer.model.named_parameters():
            if name.startswith(('freq_branch.', 'freq_fusion.')):
                def record(gradient, key=name):
                    if not torch.isfinite(gradient).all():
                        raise RuntimeError(f'Non-finite frequency gradient: {key}')
                    frequency_gradients[key] = float(gradient.abs().sum().item())
                hooks.append(parameter.register_hook(record))
        train_loss, _ = trainer.train_epoch(0)
        frequency_updated = [name for name, parameter in trainer.model.named_parameters()
                             if name in frequency_before and not torch.equal(parameter.detach(), frequency_before[name])]
        for prefix in ('freq_branch.', 'freq_fusion.'):
            if not any(name.startswith(prefix) for name in frequency_updated) or not any(
                    name.startswith(prefix) and norm > 0 for name, norm in frequency_gradients.items()):
                raise RuntimeError(f'Frequency branch did not receive gradients and update: {prefix}')
        for hook in hooks:
            hook.remove()
        train_metrics = trainer.evaluator.last_train_metrics
        val_loss, val_metrics = trainer.validate(0)
        if train_metrics['updates'] != 2 or val_metrics['updates'] != 1:
            raise RuntimeError('Strict streaming evaluator was not called in both training and validation')
        atomic_json(run / 'preflight_smoke.json', {'synthetic_train_only': False, 'real_train_samples': 4,
                    'augmented_training_pairs_source_decodable': True,
                    'frequency_updated_parameters': frequency_updated, 'frequency_gradient_l1': frequency_gradients,
                    'real_val_samples': 2, 'train_loss': train_loss,
                    'val_loss': val_loss, 'train': train_metrics, 'validation': val_metrics,
                    'optimizer_states_after_smoke': len(trainer.optimizer.state),
                    'actual_experiment_will_reinitialize': True})
        print('STRICT_TRAIN_VALIDATE_SMOKE_OK train_updates=2 val_updates=1 fresh_actual_run=True', flush=True)
    else:
        if not (run / 'preflight_smoke.json').is_file():
            raise RuntimeError('Required actual train/validation smoke has not passed')
        (run / 'python.pid').write_text(str(os.getpid()))
        trainer.train()


if __name__ == '__main__':
    main()
