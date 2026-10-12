"""Opt-in domain decoding: predicted Farmland has only no-change state.

No GT is used to decide the prediction. The strict joint scoring formula is
unchanged. Raw-state results must remain separately identified.
"""
import copy
import numpy as np
from .strict_joint_protocol import StrictJointEvaluator, NUM_JOINT
from .strict_joint_evaluator import StrictJointModelEvaluator

POLICY_VERSION = 'hicd.predicted_farmland_nochange.v1'


def decode_farmland_state(pred_target, pred_state):
    """Reusable prediction-side rule, for integer NumPy arrays or Torch tensors."""
    if pred_target.shape != pred_state.shape:
        raise ValueError('Target and state prediction shapes must match')
    if hasattr(pred_state, 'clone'):
        import torch
        integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if pred_target.dtype not in integer_types or pred_state.dtype not in integer_types:
            raise ValueError('Predictions must be integer IDs')
        return torch.where(pred_target == 0, torch.zeros_like(pred_state), pred_state)
    targets, states = np.asarray(pred_target), np.asarray(pred_state)
    if targets.dtype.kind not in 'iu' or states.dtype.kind not in 'iu':
        raise ValueError('Predictions must be integer IDs')
    return np.where(targets == 0, 0, states).astype(states.dtype)


def recompute_report(report):
    """Exact equivalent to decoding pixels before scoring, from saved counts.

    All predicted Farmland columns merge into (Farmland, no-change), including
    false positives on other GT targets/background. GT rows/masks never change.
    State diagnostics also change exactly; the raw report is never mutated.
    """
    if report.get('prediction_policy', {}).get('semantic_policy_version') == POLICY_VERSION:
        return copy.deepcopy(report)
    original = np.zeros((NUM_JOINT, NUM_JOINT), dtype=np.int64)
    for entry in report['joint_confusion']['nonzero_entries']:
        original[entry['gt_joint_id'], entry['pred_joint_id']] += entry['count']
    if original[1:7, :].sum():
        raise ValueError('GT contains Farmland states other than no-change; resolve definition first')
    decoded = original.copy()
    decoded[:, 0] = original[:, :7].sum(axis=1)
    decoded[:, 1:7] = 0
    evaluator = StrictJointEvaluator()
    evaluator.confusion = decoded
    for target in range(15):
        state_counts = np.array(report['per_target'][str(target)]['state_on_gt_target']
                                ['confusion_gt_rows_pred_columns'], dtype=np.int64)
        farmland_predictions = original[target*7:(target+1)*7, :7]
        state_counts -= farmland_predictions
        state_counts[:, 0] += farmland_predictions.sum(axis=1)
        if np.any(state_counts < 0):
            raise ValueError('Saved joint counts and state diagnostics are inconsistent')
        evaluator.state_on_gt_target[target] = state_counts
    for field in ('valid_pixels', 'ignored_pixels', 'updates'):
        setattr(evaluator, field, report[field])
    result = evaluator.compute()
    result['prediction_policy'] = copy.deepcopy(report.get('prediction_policy', {}))
    result['prediction_policy'].update(semantic_policy_version=POLICY_VERSION,
        fixed_state_by_predicted_target={'0': 0}, ground_truth_used_for_decoding=False,
        rule='predicted target=Farmland -> predicted state=no-change; other target predictions unchanged')
    result['recomputation'] = {'source': 'joint + per-GT-target state confusion counts',
                               'exact_pixel_decoding_equivalence': True,
                               'new_model_inference': False, 'ground_truth_unchanged': True}
    return result


class FarmlandNoChangeModelEvaluator(StrictJointModelEvaluator):
    """Opt-in batch adapter; raw core accumulation + exact decoded report.

    The ordinary strict adapter/core remain unchanged. The same rule can be
    applied to exported predictions with decode_farmland_state().
    """
    def compute(self):
        return recompute_report(super().compute())
