"""Frozen HICD strict target-state pixel protocol v1 (NumPy only).

See docs/strict_joint_protocol.md. This module never imports legacy BDD and
never merges states. Rows are GT, columns predictions; background is one bin.
"""
from __future__ import annotations

import numpy as np

PROTOCOL_VERSION = "hicd.strict_joint.v1"
TARGET_NAMES = (
    "Farmland", "Runway", "Taxiway", "Apron", "Bridge", "Highway",
    "Building", "Shelter", "Tower", "Pier", "Dock", "Tank", "Aircraft",
    "Vessel", "Crater", "Background",
)
STATE_NAMES = (
    "no-change", "damaged", "reduced", "added", "extended", "replaced", "none",
)
TASKS = {
    "T1": (6,),
    "T2": (0, 1, 2, 3, 5),
    "T3": (12, 13),
    "T4": (4, 7, 9, 10, 11, 14),
}
BACKGROUND_TARGET = 15
IGNORE_INDEX = 255
NUM_STATES = 7
NUM_FOREGROUND_JOINT = BACKGROUND_TARGET * NUM_STATES
BACKGROUND_JOINT = NUM_FOREGROUND_JOINT
NUM_JOINT = NUM_FOREGROUND_JOINT + 1


def _integer_array(value, name):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer IDs, got {array.dtype}")
    return array


def _prf(tp, fp, fn):
    tp, fp, fn = int(tp), int(fp), int(fn)
    support, predicted = tp + fn, tp + fp
    denominator = 2 * tp + fp + fn
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "support": support, "predicted_support": predicted,
        "precision": tp / predicted if predicted else None,
        "recall": tp / support if support else None,
        "f1": 2 * tp / denominator if denominator else None,
        "evaluable": bool(denominator),
    }


def validate_dataset_schema(config):
    """Reject changed ID semantics instead of silently relabelling protocol v1."""
    targets = config.get("pixel_target_names", config.get("target_names"))
    if targets is None or tuple(targets) != TARGET_NAMES:
        raise ValueError("Protocol v1 requires the exact 16 target names/IDs")
    if tuple(config.get("state_names", ())) != STATE_NAMES:
        raise ValueError("Protocol v1 requires all seven original state names/IDs")
    if config.get("num_states") != NUM_STATES:
        raise ValueError("Protocol v1 requires num_states=7")
    if config.get("num_pixel_targets", config.get("num_targets")) != 16:
        raise ValueError("Protocol v1 requires 16 targets")
    if config.get("ignore_index", IGNORE_INDEX) != IGNORE_INDEX:
        raise ValueError("Protocol v1 requires GT ignore=255")


class StrictJointEvaluator:
    """Streaming evaluator over decoded, aligned integer target/state arrays.

    update() keeps no predictions. State support is determined by observed GT
    and predictions, not by a hand-picked list of permissible target states.
    A foreground (target, none) is scored literally, including if it is only FP.
    """

    def __init__(self):
        self.confusion = np.zeros((NUM_JOINT, NUM_JOINT), dtype=np.int64)
        # Keep state predictions even when pred_target=background; the joint
        # background bin intentionally does not encode its state.
        self.state_on_gt_target = np.zeros((BACKGROUND_TARGET, NUM_STATES, NUM_STATES), dtype=np.int64)
        self.valid_pixels = 0
        self.ignored_pixels = 0
        self.updates = 0

    def update(self, pred_target, pred_state, gt_target, gt_state):
        arrays = [_integer_array(v, n) for v, n in zip(
            (pred_target, pred_state, gt_target, gt_state),
            ("pred_target", "pred_state", "gt_target", "gt_state"),
        )]
        if any(a.shape != arrays[0].shape for a in arrays[1:]):
            raise ValueError("All target/state arrays must have identical shapes")
        pt, ps, gt, gs = (a.reshape(-1) for a in arrays)
        # The only evaluation mask: GT ignore in either component. Never mask
        # by the prediction, the task or whether a class exists in this batch.
        valid = (gt != IGNORE_INDEX) & (gs != IGNORE_INDEX)
        values = [a[valid] for a in (pt, ps, gt, gs)]
        for a, upper, name in zip(values, (16, 7, 16, 7),
                                 ("pred_target", "pred_state", "gt_target", "gt_state")):
            if np.any(a >= upper) or np.any(a < 0):
                raise ValueError(f"Invalid {name} on valid GT pixels; expected 0..{upper - 1}")
        pt, ps, gt, gs = (a.astype(np.int64, copy=False) for a in values)
        gt_joint = np.where(gt == BACKGROUND_TARGET, BACKGROUND_JOINT, gt * NUM_STATES + gs)
        pred_joint = np.where(pt == BACKGROUND_TARGET, BACKGROUND_JOINT, pt * NUM_STATES + ps)
        histogram = np.bincount(gt_joint * NUM_JOINT + pred_joint,
                                minlength=NUM_JOINT * NUM_JOINT).reshape(NUM_JOINT, NUM_JOINT)
        self.confusion += histogram
        foreground = gt != BACKGROUND_TARGET
        state_codes = gt[foreground] * (NUM_STATES ** 2) + gs[foreground] * NUM_STATES + ps[foreground]
        self.state_on_gt_target += np.bincount(
            state_codes, minlength=BACKGROUND_TARGET * NUM_STATES ** 2,
        ).reshape(BACKGROUND_TARGET, NUM_STATES, NUM_STATES)
        self.valid_pixels += int(valid.sum())
        self.ignored_pixels += int(valid.size - valid.sum())
        self.updates += 1
        return self

    def merge(self, other):
        """Merge identically defined shard/worker accumulators."""
        if not isinstance(other, StrictJointEvaluator) or other is self:
            raise ValueError("Can only merge a different StrictJointEvaluator")
        self.confusion += other.confusion
        self.state_on_gt_target += other.state_on_gt_target
        self.valid_pixels += other.valid_pixels
        self.ignored_pixels += other.ignored_pixels
        self.updates += other.updates
        return self

    def compute(self):
        """Return JSON-safe metrics; undefined ratios/fully empty scores are null."""
        q = self.confusion
        tp = np.diag(q)
        fp = q.sum(axis=0) - tp
        fn = q.sum(axis=1) - tp
        class_stats = [_prf(tp[i], fp[i], fn[i]) for i in range(NUM_FOREGROUND_JOINT)]

        def aggregate(indices):
            indices = list(indices)
            micro = _prf(tp[indices].sum(), fp[indices].sum(), fn[indices].sum())
            active = [class_stats[i]["f1"] for i in indices if class_stats[i]["evaluable"]]
            return {"micro": micro,
                    "macro_f1": sum(active) / len(active) if active else None,
                    "evaluable_joint_classes": len(active),
                    "defined_joint_classes": len(indices)}

        foreground = aggregate(range(NUM_FOREGROUND_JOINT))
        per_target, per_joint = {}, []
        for target in range(BACKGROUND_TARGET):
            indices = list(range(target * NUM_STATES, (target + 1) * NUM_STATES))
            # Localization accepts correct target with any state; strict does not.
            loc_tp = int(q[np.ix_(indices, indices)].sum())
            loc_fp = int(q[:, indices].sum()) - loc_tp
            loc_fn = int(q[indices, :].sum()) - loc_tp
            strict = aggregate(indices)
            loc = _prf(loc_tp, loc_fp, loc_fn)
            state_q = self.state_on_gt_target[target]
            state_tp = np.diag(state_q)
            state_fp = state_q.sum(axis=0) - state_tp
            state_fn = state_q.sum(axis=1) - state_tp
            state_stats = [_prf(state_tp[s], state_fp[s], state_fn[s]) for s in range(NUM_STATES)]
            state_active = [s["f1"] for s in state_stats if s["evaluable"]]
            state_summary = {
                "scope": "valid GT pixels of this target; all pred targets retained",
                "pixels": int(state_q.sum()),
                "micro": _prf(state_tp.sum(), state_fp.sum(), state_fn.sum()),
                "macro_f1": sum(state_active) / len(state_active) if state_active else None,
                "per_state": [{"state_id": s, "state_name": STATE_NAMES[s], **state_stats[s]}
                              for s in range(NUM_STATES)],
                "confusion_gt_rows_pred_columns": state_q.tolist(),
            }
            per_target[str(target)] = {
                "target_id": target, "target_name": TARGET_NAMES[target],
                "target_localization": loc,
                "target_localization_precision": loc["precision"],
                "target_localization_recall": loc["recall"],
                "target_localization_f1": loc["f1"],
                "strict_joint_precision": strict["micro"]["precision"],
                "strict_joint_recall": strict["micro"]["recall"],
                "strict_joint_f1": strict["micro"]["f1"],
                "strict_joint_micro": strict["micro"],
                "strict_joint_macro_f1": strict["macro_f1"],
                "state_on_gt_target": state_summary,
            }
            for state, index in enumerate(indices):
                per_joint.append({"joint_id": index, "target_id": target,
                                  "target_name": TARGET_NAMES[target], "state_id": state,
                                  "state_name": STATE_NAMES[state], **class_stats[index]})
        tasks = {}
        for name, targets in TASKS.items():
            indices = [t * NUM_STATES + s for t in targets for s in range(NUM_STATES)]
            summary = aggregate(indices)
            tasks[name] = {"target_ids": list(targets),
                           "target_names": [TARGET_NAMES[t] for t in targets],
                           "strict_joint_f1": summary["micro"]["f1"],
                           "strict_joint_micro": summary["micro"],
                           "strict_joint_macro_f1": summary["macro_f1"],
                           "evaluable_joint_classes": summary["evaluable_joint_classes"]}
        return {
            "protocol_version": PROTOCOL_VERSION,
            "unit": "pixel",
            "strict_joint_f1": foreground["micro"]["f1"],
            "strict_joint_micro": foreground["micro"],
            "strict_joint_macro_f1": foreground["macro_f1"],
            "evaluable_joint_classes": foreground["evaluable_joint_classes"],
            "defined_foreground_joint_classes": NUM_FOREGROUND_JOINT,
            "valid_pixels": self.valid_pixels, "ignored_pixels": self.ignored_pixels,
            "updates": self.updates, "tasks": tasks,
            "per_target": per_target, "per_target_state": per_joint,
            "background": {"joint_id": BACKGROUND_JOINT,
                           **_prf(tp[BACKGROUND_JOINT], fp[BACKGROUND_JOINT], fn[BACKGROUND_JOINT])},
            "outside_task_target_ids": [8],
            "joint_confusion": {
                "orientation": "GT row -> predicted column",
                "background_joint_id": BACKGROUND_JOINT,
                "encoding": "foreground_joint_id = target_id * 7 + state_id",
                "nonzero_entries": [{"gt_joint_id": int(row), "pred_joint_id": int(col),
                                     "count": int(q[row, col])}
                                    for row, col in zip(*np.nonzero(q))],
            },
            "legacy_metrics": "not_computed; old BDD scores require prediction-level re-evaluation",
        }


def evaluate_strict_joint(pred_target, pred_state, gt_target, gt_state):
    """Convenience API for an in-memory batch (same protocol as streaming)."""
    return StrictJointEvaluator().update(pred_target, pred_state, gt_target, gt_state).compute()
