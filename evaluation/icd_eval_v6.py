"""
HICD Evaluation Protocol v3
Pixel-level: Target mIoU + State mIoU + Separated Kappa (SeK)
Only classes with GT pixels are included in mIoU average.
SeK splits changed/unchanged groups for balanced evaluation.
"""
import torch
import numpy as np
from collections import defaultdict


class ICDEvaluatorV6:
    def __init__(self, dataset_config):
        self.dataset_config = dataset_config
        self.semantic_classes = dataset_config.get('branch_routing', {}).get('semantic', [])
        self.state_names = dataset_config.get('state_names', [])
        self.state_ignore_targets = set(dataset_config.get('state_ignore_targets', []))
        self.reset()

    def reset(self):
        self.semantic_target_preds = []
        self.semantic_target_gts = []
        self.semantic_state_preds = []
        self.semantic_state_gts = []
        self.raw_state_preds = []
        self.raw_state_gts = []
        self.refined_state_preds = []
        self.refined_state_gts = []
        self.change_preds = []
        self.change_gts = []
        self.full_state_preds = []  # unfiltered, for BDD protocol
        self.full_state_gts = []
        self.loss_accum = defaultdict(float)
        self.loss_count = 0

    def update(self, predictions, targets, loss_dict=None):
        # Target prediction (pixel-level)
        target_key = 'target_logits' if 'target_logits' in predictions else 'change_heatmap'
        if target_key in predictions and 'target_gt' in targets:
            pred = predictions[target_key].argmax(dim=1)
            self.semantic_target_preds.append(pred.cpu())
            self.semantic_target_gts.append(targets['target_gt'].cpu())

        # State prediction (pixel-level, exclude background)
        state_key = 'state_logits' if 'state_logits' in predictions else 'target_type_map'
        if state_key in predictions and 'change_state' in targets:
            pred = predictions[state_key].argmax(dim=1)
            gt_state = targets['change_state']
            # Store unfiltered for BDD protocol
            self.full_state_preds.append(pred.cpu())
            self.full_state_gts.append(gt_state.cpu())
            if 'target_gt' in targets:
                non_bg = targets['target_gt'] != 255
                # Also exclude state_ignore_targets (e.g., Farmland, Tower)
                for ign_cls in self.state_ignore_targets:
                    non_bg = non_bg & (targets['target_gt'] != ign_cls)
                pred = pred[non_bg]
                gt_state = gt_state[non_bg]
            self.semantic_state_preds.append(pred.cpu())
            self.semantic_state_gts.append(gt_state.cpu())

        # Refined state prediction (after VotingRefinementModule)
        if 'refined_state_logits' in predictions and 'change_state' in targets:
            ref_pred = predictions['refined_state_logits'].argmax(dim=1)
            ref_gt = targets['change_state']
            if 'target_gt' in targets:
                non_bg = targets['target_gt'] != 255
                for ign_cls in self.state_ignore_targets:
                    non_bg = non_bg & (targets['target_gt'] != ign_cls)
                ref_pred = ref_pred[non_bg]
                ref_gt = ref_gt[non_bg]
            self.refined_state_preds.append(ref_pred.cpu())
            self.refined_state_gts.append(ref_gt.cpu())

        # Change detection (binary) - derived from state_logits if no dedicated change_logits
        if 'change_state' in targets:
            # Get change predictions: use change_logits if available, else derive from state_logits
            if 'change_logits' in predictions:
                change_pred = predictions['change_logits'].argmax(dim=1)
            elif 'state_logits' in predictions:
                # state != no-change (0) and state != none (6) => changed
                state_pred = predictions['state_logits'].argmax(dim=1)
                change_pred = torch.zeros_like(state_pred)
                change_pred[(state_pred > 0) & (state_pred != 6)] = 1
            else:
                change_pred = None
            if change_pred is not None:
                state_gt = targets['change_state']
                change_gt = torch.zeros_like(state_gt)
                change_gt[(state_gt > 0) & (state_gt != 6)] = 1
                if 'target_gt' in targets:
                    non_bg = targets['target_gt'] != 255
                    for ign_cls in self.state_ignore_targets:
                        non_bg = non_bg & (targets['target_gt'] != ign_cls)
                    change_pred = change_pred[non_bg]
                    change_gt = change_gt[non_bg]
                self.change_preds.append(change_pred.cpu())
                self.change_gts.append(change_gt.cpu())

        if loss_dict:
            for k, v in loss_dict.items():
                if isinstance(v, (int, float)):
                    self.loss_accum[k] += v
            self.loss_count += 1

    def compute(self):
        results = {}

        # Target mIoU — only average over classes with GT pixels
        if self.semantic_target_preds:
            preds = torch.cat(self.semantic_target_preds, 0).numpy()
            gts = torch.cat(self.semantic_target_gts, 0).numpy()
            ious, present_mask = self._compute_iou(preds, gts, len(self.semantic_classes))
            results['target_iou'] = ious
            results['target_present'] = present_mask
            non_bg_present = present_mask.copy()  # class 0 is now Farmland, not background
            if non_bg_present.sum() > 0:
                results['target_miou'] = float(np.mean(ious[non_bg_present]))
            else:
                results['target_miou'] = 0.0
            valid_mask = (gts != 255)
            results['target_accuracy'] = float((preds[valid_mask] == gts[valid_mask]).mean()) if valid_mask.sum() > 0 else 0.0
            results['target_class_names'] = self.semantic_classes
        else:
            results['target_iou'] = []; results['target_miou'] = 0.0
            results['target_present'] = np.array([])
            results['target_accuracy'] = 0.0; results['target_class_names'] = []

        # State mIoU (non-background only) — only average over states with GT pixels
        if self.semantic_state_preds:
            preds = torch.cat(self.semantic_state_preds, 0).numpy()
            gts = torch.cat(self.semantic_state_gts, 0).numpy()
            num_states = len(self.state_names) if self.state_names else 7
            ious, present_mask = self._compute_iou(preds, gts, num_states)
            results['state_iou'] = ious
            results['state_present'] = present_mask
            if present_mask.sum() > 0:
                results['state_miou'] = float(np.mean(ious[present_mask]))
            else:
                results['state_miou'] = 0.0
            results['state_class_names'] = self.state_names if self.state_names else ['no-change','damaged','reduced','added','extended','replaced','none']

            # Separated Kappa (SeK)
            sek_results = self._compute_sek(preds, gts, num_states)
            results['sek'] = sek_results['sek']
            results['iou1'] = sek_results.get('iou1', 0.0)
            results['iou2'] = sek_results.get('iou2', 0.0)
            results['rho_hat'] = sek_results.get('rho_hat', 0.0)
            results['eta_hat'] = sek_results.get('eta_hat', 0.0)

            # Refined state comparison (if available)
            if self.refined_state_preds:
                rp = torch.cat(self.refined_state_preds, 0).numpy()
                rg = torch.cat(self.refined_state_gts, 0).numpy()
                r_ious, r_present = self._compute_iou(rp, rg, num_states)
                results['ref_state_miou'] = float(np.mean(r_ious[r_present])) if r_present.sum() > 0 else 0.0
                r_sek = self._compute_sek(rp, rg, num_states)
                results['ref_sek'] = r_sek['sek']
            else:
                results['ref_state_miou'] = 0.0
                results['ref_sek'] = 0.0
        else:
            results['state_iou'] = []; results['state_miou'] = 0.0
            results['state_present'] = np.array([])
            results['state_class_names'] = []
            results['sek'] = 0.0; results['iou1'] = 0.0; results['iou2'] = 0.0; results['rho_hat'] = 0.0; results['eta_hat'] = 0.0
            results['ref_state_miou'] = 0.0
            results['ref_sek'] = 0.0


        results['loss_avg'] = {k: v / max(self.loss_count, 1) for k, v in self.loss_accum.items()}

        # === BDD Protocol Metrics ===
        if self.semantic_target_preds and self.semantic_target_gts:
            pred_t = torch.cat(self.semantic_target_preds, 0).numpy().flatten()
            gt_t = torch.cat(self.semantic_target_gts, 0).numpy().flatten()
            if self.full_state_preds:
                pred_s = torch.cat(self.full_state_preds, 0).numpy().flatten()
                gt_s = torch.cat(self.full_state_gts, 0).numpy().flatten()
            else:
                pred_s = np.zeros_like(pred_t)
                gt_s = np.zeros_like(gt_t)

            bdd_results = self._compute_bdd_metrics(pred_t, pred_s, gt_t, gt_s)
            results.update(bdd_results)

            # Per-class P/R/F1
            per_class_prf = self._compute_per_class_prf(pred_t, gt_t)
            results.update(per_class_prf)

        return results

    def _compute_per_class_prf(self, pred_target, gt_target, num_classes=16):
        """Compute per-class Precision, Recall, F1."""
        results = {}
        class_names = self.dataset_config.get('target_names', [])

        for c in range(num_classes):
            name = class_names[c] if c < len(class_names) else f'class_{c}'
            name_lower = name.lower().replace(' ', '_')

            pred_c = (pred_target == c)
            gt_c = (gt_target == c)

            tp = (pred_c & gt_c).sum()
            fp = (pred_c & ~gt_c).sum()
            fn = (~pred_c & gt_c).sum()

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)

            results[f'prf_{name_lower}_p'] = float(precision)
            results[f'prf_{name_lower}_r'] = float(recall)
            results[f'prf_{name_lower}_f1'] = float(f1)

        return results

    def _compute_bdd_metrics(self, pred_target, pred_state, gt_target, gt_state):
        """
        Compute BDD protocol metrics: F_loc, F_clf, F_overall + global SeK.

        For each target class (excluding Background=15 and Tower=8):
          F_loc: target vs background F1
          F_clf: ND vs D classification F1 (on target pixels)
          F_overall: 0.3 * F_loc + 0.7 * F_clf

        Global SeK: computed on ALL valid pixels using standard SCD formula.
        """
        from evaluation.bdd_protocol import evaluate_bdd_protocol, TASKS, compute_sek

        results = {}

        # Global SeK: compute on all valid pixels (not per-class)
        # Valid = GT state < 7 and pred state < 7 (exclude ignore/nan)
        global_valid = (gt_state < 7) & (pred_state < 7) & (gt_target != 255)
        if global_valid.sum() > 0:
            global_sek, g_iou1, g_iou2, g_rho, g_eta = compute_sek(
                pred_state[global_valid], gt_state[global_valid], num_states=7
            )
            results['global_sek'] = float(global_sek) if not np.isnan(global_sek) else 0.0
            results['global_iou1'] = float(g_iou1) if not np.isnan(g_iou1) else 0.0
            results['global_iou2'] = float(g_iou2) if not np.isnan(g_iou2) else 0.0
        else:
            results['global_sek'] = 0.0
            results['global_iou1'] = 0.0
            results['global_iou2'] = 0.0

        # Evaluate each task
        for task_id, task_info in TASKS.items():
            task_f1s = []
            task_seks = []
            for cls_id in task_info['classes']:
                r = evaluate_bdd_protocol(
                    pred_target, pred_state, gt_target, gt_state,
                    target_class=cls_id, bg_class=15
                )
                if r is not None:
                    task_f1s.append(r['F_overall'])
                    task_seks.append(r.get('SeK', 0))

            if task_f1s:
                results[f'{task_id}_f1'] = float(np.mean(task_f1s))
                results[f'{task_id}_sek'] = float(np.mean(task_seks)) if task_seks else 0.0
            else:
                results[f'{task_id}_f1'] = 0.0
                results[f'{task_id}_sek'] = 0.0

        # Overall BDD F1 (average across tasks)
        task_f1s = [results.get(f'{tid}_f1', 0) for tid in TASKS]
        results['bdd_f1'] = float(np.mean(task_f1s))
        # Use global SeK as the main SeK metric
        results['bdd_sek'] = results['global_sek']

        # Building-specific (main metric from paper)
        bldg_result = evaluate_bdd_protocol(
            pred_target, pred_state, gt_target, gt_state,
            target_class=6, bg_class=15
        )
        if bldg_result:
            results['F_loc'] = bldg_result['F_loc']
            results['F_clf'] = bldg_result['F_clf']
            results['F_overall'] = bldg_result['F_overall']
            results['F_macro'] = bldg_result['F_macro']
        else:
            results['F_loc'] = 0.0
            results['F_clf'] = 0.0
            results['F_overall'] = 0.0
            results['F_macro'] = 0.0

        return results

    def _compute_iou(self, preds, gts, num_classes, ignore_index=255):
        """Compute per-class IoU and a mask of which classes have GT pixels."""
        valid = (gts != ignore_index)
        preds = preds[valid]
        gts = gts[valid]
        ious = []
        present = []
        for c in range(num_classes):
            p = (preds == c); g = (gts == c)
            inter = (p & g).sum(); union = (p | g).sum()
            ious.append(float(inter / max(union, 1)))
            present.append(union > 0)
        return np.array(ious), np.array(present)

    def _compute_kappa(self, confusion_matrix):
        """Cohen's Kappa from a pre-built confusion matrix."""
        N = confusion_matrix.sum()
        if N == 0:
            return 0.0
        p_o = np.trace(confusion_matrix) / N
        row_sum = confusion_matrix.sum(axis=1)
        col_sum = confusion_matrix.sum(axis=0)
        p_e = (row_sum * col_sum).sum() / (N * N)
        denom = 1.0 - p_e
        if denom < 1e-12:
            return 1.0 if p_o > 0.99 else 0.0
        return float((p_o - p_e) / denom)

    def _compute_sek(self, state_preds, state_gts, num_states):
        """
        Separated Kappa (SeK) for semantic change detection.
        Follows the standard SeK formula:
          IOU1 = q00 / (col0 + row0 - q00)
          IOU2 = sum_{i>=1,j>=1} q_ij / (total - q00)
          Q_hat: same as Q but q00=0
          N_change = total - q00
          rho_hat = trace_changed / N_change
          eta_hat = sum_j(row_hat[j]*col_hat[j]) / N_change^2
          SeK = exp(IOU2 - 1) * (rho_hat - eta_hat) / (1 - eta_hat)
        """
        # Filter out state=6 (none/background) and ignore_index
        valid_mask = (state_gts < num_states)
        v_preds = state_preds[valid_mask]
        v_gts = state_gts[valid_mask]

        if len(v_gts) == 0:
            return {'sek': float('nan'), 'iou1': float('nan'), 'iou2': float('nan'),
                    'rho_hat': float('nan'), 'eta_hat': float('nan')}

        # Build C x C confusion matrix Q: rows=GT, cols=Pred
        # Class 0 = no-change, classes 1..C-1 = changed types
        C = num_states
        Q = np.zeros((C, C), dtype=np.float64)
        for i in range(C):
            for j in range(C):
                Q[i, j] = int(((v_gts == i) & (v_preds == j)).sum())

        q00 = Q[0, 0]
        total = Q.sum()

        # IOU1: unchanged class IoU
        col0 = Q[:, 0].sum()  # all predicted as 0
        row0 = Q[0, :].sum()  # all GT=0
        iou1_denom = col0 + row0 - q00
        iou1 = q00 / iou1_denom if iou1_denom > 0 else float('nan')

        # IOU2: changed classes IoU
        N_change = total - q00
        changed_correct = Q[1:, 1:].sum()
        iou2 = changed_correct / N_change if N_change > 0 else float('nan')

        # Modified confusion matrix Q_hat: Q_hat[0,0] = 0
        Q_hat = Q.copy()
        Q_hat[0, 0] = 0.0

        # rho_hat: accuracy among changed pixels
        trace_changed = np.trace(Q_hat)  # sum of diagonal (all diag except q00 is already 0)
        # Actually trace_changed = sum_{i=1}^{C-1} q_ii since Q_hat[0,0]=0
        rho_hat = trace_changed / N_change if N_change > 0 else float('nan')

        # eta_hat: expected agreement under independence on Q_hat
        row_hat = Q_hat.sum(axis=1)
        col_hat = Q_hat.sum(axis=0)
        eta_hat = np.sum(row_hat * col_hat) / (N_change ** 2) if N_change > 0 else float('nan')

        # SeK
        denom = 1.0 - eta_hat
        if N_change == 0 or np.isnan(iou2) or np.isnan(rho_hat) or np.isnan(eta_hat) or abs(denom) < 1e-12:
            sek = float('nan')
        else:
            sek = float(np.exp(iou2 - 1.0) * (rho_hat - eta_hat) / denom)

        return {
            'sek': float(sek),
            'iou1': float(iou1),
            'iou2': float(iou2),
            'rho_hat': float(rho_hat),
            'eta_hat': float(eta_hat),
        }

    def format_results(self, results):
        lines = ['=' * 60, 'ICD Evaluation Results', '=' * 60, '']

        # BDD Protocol (main metrics)
        lines.append('[BDD Protocol - Building Damage Detection]')
        lines.append(f"  F_loc (localization):  {results.get('F_loc', 0):.4f}")
        lines.append(f"  F_clf (classification): {results.get('F_clf', 0):.4f}")
        lines.append(f"  F_overall:              {results.get('F_overall', 0):.4f}")
        lines.append(f"  F_macro:                {results.get('F_macro', 0):.4f}")
        lines.append('')

        # Four-task summary
        lines.append('[Four-Task F1 Summary]')
        for task_id, task_name in [('T1_Building', 'T1 建筑'), ('T2_Surface', 'T2 地表'),
                                    ('T3_Mobile', 'T3 移动'), ('T4_Facility', 'T4 设施')]:
            f1 = results.get(f'{task_id}_f1', 0)
            sek = results.get(f'{task_id}_sek', 0)
            lines.append(f"  {task_name}: F1={f1:.4f}, SeK(per-class)={sek:.4f}")
        lines.append(f"  Overall: F1={results.get('bdd_f1', 0):.4f}")
        lines.append('')
        lines.append('[Global SeK (all pixels)]')
        lines.append(f"  SeK:      {results.get('global_sek', 0):.4f}")
        lines.append(f"  IOU1 (unchanged): {results.get('global_iou1', 0):.4f}")
        lines.append(f"  IOU2 (changed):   {results.get('global_iou2', 0):.4f}")
        lines.append('')

        # Per-class Precision/Recall/F1
        lines.append('[Per-Class Precision / Recall / F1]')
        class_names = self.dataset_config.get('target_names', [])
        lines.append(f"  {'Class':<12} {'Prec':>8} {'Recall':>8} {'F1':>8}")
        lines.append(f"  {'-'*40}")
        for i, name in enumerate(class_names):
            name_lower = name.lower().replace(' ', '_')
            p = results.get(f'prf_{name_lower}_p', 0)
            r = results.get(f'prf_{name_lower}_r', 0)
            f1 = results.get(f'prf_{name_lower}_f1', 0)
            lines.append(f"  {name:<12} {p:>8.4f} {r:>8.4f} {f1:>8.4f}")
        lines.append('')

        # State IoU (for reference)
        lines.append('[State IoU (reference)]')
        lines.append(f"  mIoU: {results.get('state_miou', 0):.4f}")
        state_ious = results.get('state_iou', [])
        state_present = results.get('state_present', np.array([]))
        for i, cls in enumerate(results.get('state_class_names', [])):
            if i < len(state_ious) and i < len(state_present):
                tag = '' if state_present[i] else ' [no GT]'
                lines.append(f"    {cls}: {state_ious[i]:.4f}{tag}")
        lines.append('')

        # Loss
        lines.append('[Loss Breakdown]')
        for k, v in results.get('loss_avg', {}).items():
            lines.append(f'  {k}: {v:.4f}')
        lines.append('=' * 60)
        return '\n'.join(lines)



