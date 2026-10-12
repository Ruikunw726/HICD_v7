"""Explicit opt-in model-batch adapter; legacy evaluator is untouched.

Interface: StrictJointModelEvaluator(config).update(predictions, targets),
compute(), reset(). Default: raw state_logits, GT nearest to output grid.
"""
from __future__ import annotations

if __package__:
    from .strict_joint_protocol import StrictJointEvaluator, validate_dataset_schema
    from .strict_joint_reporting import format_strict_results
else:
    from strict_joint_protocol import StrictJointEvaluator, validate_dataset_schema
    from strict_joint_reporting import format_strict_results


class StrictJointModelEvaluator:
    def __init__(self, dataset_config, state_head="raw"):
        validate_dataset_schema(dataset_config)
        if state_head not in ("raw", "refined"):
            raise ValueError("state_head must be raw or explicitly opted-in refined")
        self.state_head = state_head
        self.reset()

    def reset(self):
        self.accumulator = StrictJointEvaluator()
        self.grid_counts = {}
        self.samples = 0

    def update(self, predictions, targets, loss_dict=None):
        import torch
        import torch.nn.functional as functional

        state_key = "state_logits" if self.state_head == "raw" else "refined_state_logits"
        if "target_logits" not in predictions or state_key not in predictions:
            raise ValueError(f"Strict evaluation requires target_logits and {state_key}")
        if "target_gt" not in targets:
            raise ValueError("Strict evaluation requires target_gt")
        state_gt = targets.get("state_gt", targets.get("change_state"))
        if state_gt is None:
            raise ValueError("Strict evaluation requires state_gt or change_state")
        target_logits, state_logits = predictions["target_logits"], predictions[state_key]
        if target_logits.ndim != 4 or state_logits.ndim != 4:
            raise ValueError("Logits must be B,C,H,W")
        if target_logits.shape[1] != 16 or state_logits.shape[1] != 7:
            raise ValueError("Expected 16 target and 7 state logits")
        if target_logits.shape[0] != state_logits.shape[0] or target_logits.shape[2:] != state_logits.shape[2:]:
            raise ValueError("Target/state logits must have identical batch and spatial sizes")
        if not torch.isfinite(target_logits).all() or not torch.isfinite(state_logits).all():
            raise ValueError("Non-finite model logits")
        gt_target = targets["target_gt"]
        if gt_target.ndim != 3 or state_gt.ndim != 3 or gt_target.shape != state_gt.shape:
            raise ValueError("GT must be matching B,H,W target and state maps")
        if gt_target.shape[0] != target_logits.shape[0]:
            raise ValueError("Prediction and GT batch sizes differ")
        integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if gt_target.dtype not in integer_dtypes or state_gt.dtype not in integer_dtypes:
            raise ValueError("GT must contain integer IDs")
        gt_shape = tuple(gt_target.shape[1:])
        pred_shape = tuple(target_logits.shape[2:])
        if gt_shape != pred_shape:
            gt_target = functional.interpolate(gt_target.unsqueeze(1).float(), size=pred_shape,
                                               mode="nearest").squeeze(1).long()
            state_gt = functional.interpolate(state_gt.unsqueeze(1).float(), size=pred_shape,
                                              mode="nearest").squeeze(1).long()
        self.accumulator.update(target_logits.argmax(dim=1), state_logits.argmax(dim=1),
                                gt_target, state_gt)
        grid_key = (gt_shape, pred_shape)
        self.grid_counts[grid_key] = self.grid_counts.get(grid_key, 0) + int(gt_target.shape[0])
        self.samples += int(gt_target.shape[0])

    def compute(self):
        result = self.accumulator.compute()
        result["prediction_policy"] = {
            "target_head": "target_logits", "state_head": self.state_head,
            "state_key": "state_logits" if self.state_head == "raw" else "refined_state_logits",
            "spatial_policy": "GT nearest-neighbor to prediction grid; no prediction-based mask",
            "grids": [{"input_gt_hw": list(gt), "evaluation_hw": list(pred), "samples": count}
                      for (gt, pred), count in sorted(self.grid_counts.items())],
            "samples": self.samples,
        }
        return result

    @staticmethod
    def format_results(result):
        return format_strict_results(result)
