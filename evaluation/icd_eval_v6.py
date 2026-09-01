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
        self.reset()

    def reset(self):
        self.semantic_target_preds = []
        self.semantic_target_gts = []
        self.semantic_state_preds = []
        self.semantic_state_gts = []
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
            if 'target_gt' in targets:
                non_bg = targets['target_gt'] != 255
                pred = pred[non_bg]
                gt_state = gt_state[non_bg]
            self.semantic_state_preds.append(pred.cpu())
            self.semantic_state_gts.append(gt_state.cpu())

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
            results['kappa_changed'] = sek_results['kappa_changed']
            results['kappa_unchanged'] = sek_results['kappa_unchanged']
        else:
            results['state_iou'] = []; results['state_miou'] = 0.0
            results['state_present'] = np.array([])
            results['state_class_names'] = []
            results['sek'] = 0.0; results['kappa_changed'] = 0.0; results['kappa_unchanged'] = 0.0

        results['loss_avg'] = {k: v / max(self.loss_count, 1) for k, v in self.loss_accum.items()}
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

        State layout: 0=no-change, 1-5=changed types, 6=none(background).

        Kappa_changed: multi-class Kappa among changed states {1,2,3,4,5},
                        computed only over pixels where GT is changed.
        Kappa_unchanged: binary Kappa (no-change vs changed) computed over
                         ALL non-background pixels, capturing false positives
                         and false negatives for change detection.

        SeK = (Kappa_changed + Kappa_unchanged) / 2
        """
        # Changed states: 1-5 (damaged, reduced, added, extended, replaced)
        changed_ids = [i for i in range(1, num_states) if i != 6]
        unchanged_id = 0

        # Filter out state=6 (none) and ignore_index
        valid_mask = (state_gts < num_states)
        v_preds = state_preds[valid_mask]
        v_gts = state_gts[valid_mask]

        kappa_changed = 0.0
        kappa_unchanged = 0.0

        # --- Kappa_changed: multi-class among changed states ---
        changed_mask = np.isin(v_gts, changed_ids)
        if changed_mask.sum() > 0:
            ch_preds = v_preds[changed_mask]
            ch_gts = v_gts[changed_mask]
            all_ch = sorted(set(ch_gts.tolist()) | set(ch_preds.tolist()))
            if len(all_ch) > 1:
                mapping = {v: i for i, v in enumerate(all_ch)}
                ch_preds_r = np.array([mapping[v] for v in ch_preds])
                ch_gts_r = np.array([mapping[v] for v in ch_gts])
                n_cls = len(all_ch)
                cm = np.zeros((n_cls, n_cls), dtype=np.int64)
                for c in range(n_cls):
                    for p in range(n_cls):
                        cm[c, p] = int(((ch_gts_r == c) & (ch_preds_r == p)).sum())
                kappa_changed = self._compute_kappa(cm)
            elif len(all_ch) == 1:
                kappa_changed = 1.0 if np.array_equal(ch_preds, ch_gts) else 0.0

        # --- Kappa_unchanged: binary Kappa over ALL pixels ---
        # Binary GT: 0 = no-change, 1 = changed (any of states 1-5)
        # Binary Pred: same binarization on predicted states
        bin_gt = np.isin(v_gts, changed_ids).astype(np.int64)
        bin_pred = np.isin(v_preds, changed_ids).astype(np.int64)
        cm_bin = np.zeros((2, 2), dtype=np.int64)
        for c in range(2):
            for p in range(2):
                cm_bin[c, p] = int(((bin_gt == c) & (bin_pred == p)).sum())
        kappa_unchanged = self._compute_kappa(cm_bin)

        # SeK = average of both Kappas
        sek = (kappa_changed + kappa_unchanged) / 2.0
        return {
            'sek': float(sek),
            'kappa_changed': float(kappa_changed),
            'kappa_unchanged': float(kappa_unchanged),
        }

    def format_results(self, results):
        lines = ['=' * 60, 'ICD Evaluation Results', '=' * 60, '']
        lines.append('[Target Classification]')
        lines.append(f"  mIoU (excl BG): {results.get('target_miou', 0):.4f}")
        lines.append(f"  Accuracy: {results.get('target_accuracy', 0):.4f}")
        target_ious = results.get('target_iou', [])
        target_present = results.get('target_present', np.array([]))
        for i, cls in enumerate(results.get('target_class_names', [])):
            if i < len(target_ious) and i < len(target_present):
                tag = '' if target_present[i] else ' [no GT]'
                lines.append(f"  {cls}: {target_ious[i]:.4f}{tag}")
        lines.append('')
        lines.append('[State Classification (non-bg)]')
        lines.append(f"  mIoU: {results.get('state_miou', 0):.4f}")
        state_ious = results.get('state_iou', [])
        state_present = results.get('state_present', np.array([]))
        for i, cls in enumerate(results.get('state_class_names', [])):
            if i < len(state_ious) and i < len(state_present):
                tag = '' if state_present[i] else ' [no GT]'
                lines.append(f"  {cls}: {state_ious[i]:.4f}{tag}")
        lines.append('')
        lines.append('[Separated Kappa (SeK)]')
        lines.append(f"  SeK:            {results.get('sek', 0):.4f}")
        lines.append(f"  Kappa_changed:  {results.get('kappa_changed', 0):.4f}")
        lines.append(f"  Kappa_unchanged:{results.get('kappa_unchanged', 0):.4f}")
        lines.append('')
        lines.append('[Loss Breakdown]')
        for k, v in results.get('loss_avg', {}).items():
            lines.append(f'  {k}: {v:.4f}')
        lines.append('=' * 60)
        return '\n'.join(lines)



