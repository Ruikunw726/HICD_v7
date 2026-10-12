"""Numerical acceptance tests for frozen strict v1; no model/data/GPU needed."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from strict_joint_protocol import (StrictJointEvaluator, evaluate_strict_joint,
                                   validate_dataset_schema, TARGET_NAMES, STATE_NAMES)


def evaluate(pt, ps, gt, gs):
    return evaluate_strict_joint(*(np.array(v, dtype=np.int64) for v in (pt, ps, gt, gs)))


class StrictJointProtocolTests(unittest.TestCase):
    def test_legacy_perfect_but_all_specific_states_wrong(self):
        result = evaluate([6] * 4, [3, 0, 2, 1], [6] * 4, [0, 3, 1, 2])
        micro = result["strict_joint_micro"]
        self.assertEqual((micro["tp"], micro["fp"], micro["fn"]), (0, 4, 4))
        self.assertEqual(result["strict_joint_f1"], 0)
        self.assertEqual(result["strict_joint_macro_f1"], 0)
        building = result["per_target"]["6"]
        self.assertEqual(building["target_localization_f1"], 1)
        self.assertEqual(building["state_on_gt_target"]["micro"]["f1"], 0)

    def test_mixed_cross_check(self):
        result = evaluate([6, 6, 6, 6, 6, 15, 6], [0, 1, 6, 3, 5, 0, 0],
                          [6, 6, 6, 15, 12, 15, 255], [3, 1, 0, 6, 5, 6, 1])
        micro = result["strict_joint_micro"]
        self.assertEqual((micro["tp"], micro["fp"], micro["fn"]), (1, 4, 3))
        self.assertAlmostEqual(result["strict_joint_f1"], 2 / 9)
        self.assertAlmostEqual(result["strict_joint_macro_f1"], 1 / 6)
        self.assertEqual(result["evaluable_joint_classes"], 6)
        task = result["tasks"]["T1"]["strict_joint_micro"]
        self.assertEqual((task["tp"], task["fp"], task["fn"]), (1, 4, 2))
        self.assertAlmostEqual(task["f1"], 1 / 4)
        self.assertAlmostEqual(result["tasks"]["T1"]["strict_joint_macro_f1"], 1 / 5)
        self.assertEqual(result["tasks"]["T3"]["strict_joint_f1"], 0)
        self.assertAlmostEqual(result["per_target"]["6"]["target_localization_f1"], 3 / 4)
        self.assertEqual(result["ignored_pixels"], 1)

    def test_wrong_target_same_state_keeps_state_diagnostic(self):
        result = evaluate([12], [3], [6], [3])
        self.assertEqual(result["strict_joint_micro"]["fn"], 1)
        self.assertEqual(result["strict_joint_micro"]["fp"], 1)
        building = result["per_target"]["6"]
        self.assertEqual(building["strict_joint_f1"], 0)
        self.assertEqual(building["target_localization_f1"], 0)
        self.assertEqual(building["state_on_gt_target"]["micro"]["f1"], 1)
        self.assertEqual(result["tasks"]["T1"]["strict_joint_micro"]["fn"], 1)
        self.assertEqual(result["tasks"]["T3"]["strict_joint_micro"]["fp"], 1)

    def test_background_false_positive(self):
        result = evaluate([6], [3], [15], [6])
        self.assertEqual(result["strict_joint_micro"]["fp"], 1)
        self.assertEqual(result["strict_joint_f1"], 0)
        self.assertEqual(result["per_target"]["6"]["strict_joint_precision"], 0)
        self.assertIsNone(result["per_target"]["6"]["strict_joint_recall"])
        self.assertEqual(result["evaluable_joint_classes"], 1)

    def test_background_prediction_with_right_state_is_still_miss(self):
        result = evaluate([15], [3], [6], [3])
        self.assertEqual(result["strict_joint_micro"]["fn"], 1)
        self.assertEqual(result["strict_joint_micro"]["fp"], 0)
        self.assertEqual(result["per_target"]["6"]["state_on_gt_target"]["micro"]["f1"], 1)

    def test_background_state_has_no_effect_on_joint(self):
        a = evaluate([15], [0], [15], [6])
        b = evaluate([15], [6], [15], [6])
        self.assertEqual(a, b)
        self.assertIsNone(a["strict_joint_f1"])

    def test_none_prediction_is_fp_and_fn(self):
        result = evaluate([6], [6], [6], [0])
        self.assertEqual(result["strict_joint_micro"]["fp"], 1)
        self.assertEqual(result["strict_joint_micro"]["fn"], 1)
        none = next(r for r in result["per_target_state"] if r["target_id"] == 6 and r["state_id"] == 6)
        self.assertEqual((none["support"], none["predicted_support"], none["fp"], none["f1"]), (0, 1, 1, 0))
        self.assertEqual(result["per_target"]["6"]["state_on_gt_target"]["per_state"][6]["fp"], 1)

    def test_explicit_foreground_none_gt_is_not_ignored(self):
        result = evaluate([6], [6], [6], [6])
        self.assertEqual(result["strict_joint_f1"], 1)
        self.assertEqual(result["valid_pixels"], 1)

    def test_replaced_is_kept(self):
        self.assertEqual(evaluate([12], [5], [12], [5])["strict_joint_f1"], 1)
        self.assertEqual(evaluate([12], [3], [12], [5])["strict_joint_f1"], 0)

    def test_perfect_single_state_crater(self):
        result = evaluate([14] * 3, [3] * 3, [14] * 3, [3] * 3)
        self.assertEqual(result["strict_joint_f1"], 1)
        self.assertEqual(result["strict_joint_macro_f1"], 1)
        self.assertEqual(result["tasks"]["T4"]["strict_joint_f1"], 1)
        self.assertEqual(result["evaluable_joint_classes"], 1)

    def test_missing_states_do_not_force_zero_macro(self):
        result = evaluate([6, 6], [0, 0], [6, 6], [0, 0])
        self.assertEqual(result["strict_joint_macro_f1"], 1)
        unused = result["per_target_state"][43]
        self.assertFalse(unused["evaluable"])
        self.assertIsNone(unused["f1"])

    def test_farmland_loss_ignore_does_not_apply_to_metric(self):
        result = evaluate([0], [1], [0], [0])
        self.assertEqual(result["valid_pixels"], 1)
        self.assertEqual(result["tasks"]["T2"]["strict_joint_f1"], 0)

    def test_tower_is_foreground_but_outside_tasks(self):
        result = evaluate([8], [3], [8], [3])
        self.assertEqual(result["strict_joint_f1"], 1)
        self.assertTrue(all(task["strict_joint_f1"] is None for task in result["tasks"].values()))

    def test_gt_ignore_in_either_component_masks_all_statistics(self):
        clean = evaluate([6], [3], [6], [3])
        dirty = evaluate([6, -99, 999, 255], [3, 999, -9, 255],
                         [6, 255, 6, 255], [3, 0, 255, 255])
        for key in ("strict_joint_micro", "per_target", "per_target_state", "joint_confusion"):
            self.assertEqual(clean[key], dirty[key])
        self.assertEqual(dirty["ignored_pixels"], 3)

    def test_all_empty_returns_null_and_json_is_standard(self):
        result = evaluate([], [], [], [])
        self.assertIsNone(result["strict_joint_f1"])
        self.assertIsNone(result["strict_joint_macro_f1"])
        self.assertEqual(result["valid_pixels"], 0)
        self.assertNotIn("NaN", json.dumps(result, allow_nan=False))
        ignored = evaluate([6], [0], [255], [0])
        self.assertIsNone(ignored["strict_joint_f1"])

    def test_streaming_and_merge_equal_one_shot_random_reference(self):
        rng = np.random.default_rng(41008)
        pt, gt = rng.integers(0, 16, size=(300,)), rng.integers(0, 16, size=(300,))
        ps, gs = rng.integers(0, 7, size=(300,)), rng.integers(0, 7, size=(300,))
        gt[::11] = 255
        gs[::17] = 255
        once = StrictJointEvaluator().update(pt, ps, gt, gs)
        stream = StrictJointEvaluator()
        merged = StrictJointEvaluator()
        for start in range(0, 300, 13):
            values = [a[start:start + 13] for a in (pt, ps, gt, gs)]
            stream.update(*values)
            merged.merge(StrictJointEvaluator().update(*values))
        np.testing.assert_array_equal(once.confusion, stream.confusion)
        np.testing.assert_array_equal(once.state_on_gt_target, stream.state_on_gt_target)
        np.testing.assert_array_equal(once.confusion, merged.confusion)
        np.testing.assert_array_equal(once.state_on_gt_target, merged.state_on_gt_target)
        # Independent tuple-counter oracle: no joint ID arithmetic.
        true, predicted, correct = Counter(), Counter(), Counter()
        for pred_t, pred_s, gt_t, gt_s in zip(pt, ps, gt, gs):
            if gt_t == 255 or gt_s == 255:
                continue
            if gt_t != 15:
                true[(gt_t, gt_s)] += 1
            if pred_t != 15:
                predicted[(pred_t, pred_s)] += 1
            if gt_t != 15 and (gt_t, gt_s) == (pred_t, pred_s):
                correct[(gt_t, gt_s)] += 1
        active = set(true) | set(predicted)
        tp = sum(correct.values())
        expected_f1 = 2 * tp / (sum(true.values()) + sum(predicted.values()))
        expected_macro = sum(2 * correct[k] / (true[k] + predicted[k]) for k in active) / len(active)
        result = once.compute()
        self.assertAlmostEqual(result["strict_joint_f1"], expected_f1)
        self.assertAlmostEqual(result["strict_joint_macro_f1"], expected_macro)
        comparable = copy.deepcopy(stream.compute())
        comparable["updates"] = result["updates"]
        self.assertEqual(result, comparable)

    def test_failed_update_is_transactional(self):
        evaluator = StrictJointEvaluator()
        evaluator.update(*[np.array([6]), np.array([3]), np.array([6]), np.array([3])])
        before = copy.deepcopy(evaluator.compute())
        for pt, ps, gt, gs in ([255], [0], [6], [0]), ([6], [-1], [6], [0]), ([6], [0], [16], [0]):
            with self.assertRaises(ValueError):
                evaluator.update(*(np.array(v) for v in (pt, ps, gt, gs)))
        self.assertEqual(before, evaluator.compute())

    def test_float_and_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_strict_joint(np.array([6.0]), np.array([0]), np.array([6]), np.array([0]))
        with self.assertRaises(ValueError):
            evaluate_strict_joint(np.array([[6]]), np.array([0]), np.array([6]), np.array([0]))

    def test_schema_missing_or_reordered_states_rejected(self):
        config = {"pixel_target_names": list(TARGET_NAMES), "state_names": list(STATE_NAMES),
                  "num_states": 7, "num_pixel_targets": 16, "ignore_index": 255}
        validate_dataset_schema(config)
        config["state_names"] = list(STATE_NAMES[:-1])
        with self.assertRaises(ValueError):
            validate_dataset_schema(config)
        config["state_names"] = list(reversed(STATE_NAMES))
        with self.assertRaises(ValueError):
            validate_dataset_schema(config)

    def test_merge_self_rejected(self):
        evaluator = StrictJointEvaluator()
        with self.assertRaises(ValueError):
            evaluator.merge(evaluator)

    def test_cpu_cli_shards_and_refusal_to_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            np.savez(folder / "a.npz", pred_target=[6], pred_state=[3], gt_target=[6], gt_state=[3])
            np.savez(folder / "b.npz", pred_target=[6], pred_state=[0], gt_target=[6], gt_state=[3])
            output = folder / "scores.json"
            command = [sys.executable, str(ROOT / "script/eval_strict_joint.py"),
                       "--predictions", str(folder), "--output", str(output), "--split", "test"]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["strict_joint_micro"]["tp"], 1)
            self.assertEqual(result["strict_joint_micro"]["fp"], 1)
            self.assertEqual(result["strict_joint_micro"]["fn"], 1)
            self.assertAlmostEqual(result["strict_joint_f1"], 0.5)
            self.assertEqual(len(result["provenance"]["sources"]), 2)
            original = output.read_bytes()
            retry = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(retry.returncode, 0)
            self.assertEqual(original, output.read_bytes())

    def test_cache_missing_arrays_and_duplicate_file_rejected(self):
        spec = importlib.util.spec_from_file_location("strict_cache_entry", ROOT / "script/eval_strict_joint.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "a.npz"
            np.savez(path, pred_target=[6])
            with self.assertRaises(ValueError):
                module.evaluate_cache([path])
            np.savez(path, pred_target=[6], pred_state=[3], gt_target=[6], gt_state=[3])
            with self.assertRaises(ValueError):
                module.evaluate_cache([path, path])
            with self.assertRaises(ValueError):
                module.evaluate_cache([])


class ModelBatchAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest(f"Optional model adapter needs PyTorch: {error}")
        cls.torch = torch
        from strict_joint_evaluator import StrictJointModelEvaluator
        cls.evaluator_type = StrictJointModelEvaluator

    def make(self):
        return self.evaluator_type({"pixel_target_names": TARGET_NAMES, "state_names": STATE_NAMES,
                                    "num_states": 7, "num_pixel_targets": 16})

    def logits(self, labels, classes):
        values = self.torch.tensor(labels, dtype=self.torch.long)
        return self.torch.nn.functional.one_hot(values, classes).permute(0, 3, 1, 2).float()

    def test_adapter_default_raw_and_explicit_refined(self):
        torch = self.torch
        evaluator = self.make()
        predictions = {"target_logits": self.logits([[[6]]], 16),
                       "state_logits": self.logits([[[0]]], 7),
                       "refined_state_logits": self.logits([[[3]]], 7)}
        targets = {"target_gt": torch.tensor([[[6]]]), "change_state": torch.tensor([[[3]]])}
        evaluator.update(predictions, targets)
        result = evaluator.compute()
        self.assertEqual(result["strict_joint_f1"], 0)
        self.assertEqual(result["prediction_policy"]["state_key"], "state_logits")
        evaluator.state_head = "refined"
        evaluator.reset()
        evaluator.update(predictions, targets)
        self.assertEqual(evaluator.compute()["strict_joint_f1"], 1)

    def test_adapter_nearest_gt_to_actual_prediction_grid(self):
        torch = self.torch
        evaluator = self.make()
        predictions = {"target_logits": self.logits([[[6, 6], [6, 6]]], 16),
                       "state_logits": self.logits([[[0, 3], [1, 2]]], 7)}
        targets = {"target_gt": torch.full((1, 4, 4), 6),
                   "state_gt": torch.tensor([[[0, 6, 3, 6], [6, 6, 6, 6],
                                              [1, 6, 2, 6], [6, 6, 6, 6]]])}
        evaluator.update(predictions, targets)
        result = evaluator.compute()
        self.assertEqual(result["strict_joint_f1"], 1)
        self.assertEqual(result["valid_pixels"], 4)
        self.assertEqual(result["prediction_policy"]["grids"],
                         [{"input_gt_hw": [4, 4], "evaluation_hw": [2, 2], "samples": 1}])

    def test_adapter_empty_reset_and_format(self):
        evaluator = self.make()
        result = evaluator.compute()
        self.assertIn("n/a", evaluator.format_results(result))
        self.assertIsNone(result["strict_joint_f1"])
        evaluator.reset()
        self.assertEqual(evaluator.compute()["prediction_policy"]["samples"], 0)

    def test_adapter_requires_complete_finite_heads_and_gt(self):
        torch = self.torch
        evaluator = self.make()
        predictions = {"target_logits": self.logits([[[6]]], 16),
                       "state_logits": self.logits([[[3]]], 7)}
        targets = {"target_gt": torch.tensor([[[6]]]), "change_state": torch.tensor([[[3]]])}
        with self.assertRaises(ValueError):
            evaluator.update({"target_logits": predictions["target_logits"]}, targets)
        with self.assertRaises(ValueError):
            evaluator.update(predictions, {"target_gt": targets["target_gt"]})
        broken = {**predictions, "state_logits": predictions["state_logits"] * float("nan")}
        with self.assertRaises(ValueError):
            evaluator.update(broken, targets)
        self.assertEqual(evaluator.compute()["valid_pixels"], 0)

    def test_adapter_wrong_target_does_not_filter_state(self):
        torch = self.torch
        evaluator = self.make()
        predictions = {"target_logits": self.logits([[[15, 12]]], 16),
                       "state_logits": self.logits([[[3, 1]]], 7)}
        targets = {"target_gt": torch.tensor([[[6, 6]]]), "change_state": torch.tensor([[[3, 1]]])}
        evaluator.update(predictions, targets)
        building = evaluator.compute()["per_target"]["6"]
        self.assertEqual(building["strict_joint_f1"], 0)
        self.assertEqual(building["target_localization_f1"], 0)
        self.assertEqual(building["state_on_gt_target"]["micro"]["f1"], 1)
        self.assertEqual(building["state_on_gt_target"]["pixels"], 2)


if __name__ == "__main__":
    unittest.main()
