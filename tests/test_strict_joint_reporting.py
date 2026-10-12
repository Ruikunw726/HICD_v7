"""Presentation regression: four decimals, no changes to metric precision."""
import copy
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'evaluation'))
from strict_joint_protocol import evaluate_strict_joint
from strict_joint_reporting import format_strict_results, format_epoch_summary, format_csv_row


class StrictReportingTests(unittest.TestCase):
    def result(self):
        return evaluate_strict_joint(
            np.array([6, 6, 6, 6, 6, 15, 6]), np.array([0, 1, 6, 3, 5, 0, 0]),
            np.array([6, 6, 6, 15, 12, 15, 255]), np.array([3, 1, 0, 6, 5, 6, 1]))

    def test_all_displayed_ratios_are_four_decimals(self):
        result = self.result()
        before = copy.deepcopy(result)
        text = format_strict_results(result)
        self.assertIn('micro F1=0.2222 macro F1=0.1667', text)
        self.assertIn('T1: 0.2500/0.2000', text)
        self.assertIn('0.7500', text)
        for number in re.findall(r'\d+\.(\d+)', text):
            self.assertEqual(len(number), 4)
        self.assertEqual(before, result)
        self.assertEqual(result['strict_joint_f1'], 2 / 9)

    def test_omit_empty_targets_keep_false_positive_only(self):
        result = evaluate_strict_joint(np.array([14]), np.array([3]), np.array([15]), np.array([6]))
        text = format_strict_results(result)
        self.assertIn('Crater', text)
        self.assertIn('0.0000', text)
        self.assertIn('n/a', text)
        self.assertNotIn('Building', text)
        self.assertNotIn('TP/FP/FN', text)

    def test_epoch_and_csv_presentation_preserve_integers_and_inputs(self):
        artifact = {'epoch': 2, 'train_loss': 3.123456, 'validation_loss': 2.654321,
                    'best_score': 2 / 9, 'validation': self.result()}
        text = format_epoch_summary(artifact)
        self.assertIn('Epoch 2 | train loss=3.1235 val loss=2.6543 best joint F1=0.2222', text)
        row = {'score': 2 / 9, 'fp': 4, 'no_score': None, 'is_best': True}
        self.assertEqual(format_csv_row(row), {'score': '0.2222', 'fp': 4, 'no_score': None, 'is_best': True})
        self.assertEqual(row['score'], 2 / 9)

    def test_completed_report_publish_is_deduplicated_and_read_only(self):
        spec = importlib.util.spec_from_file_location('epoch_reporter', ROOT / 'script/report_strict_epochs.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / 'metrics').mkdir()
            artifact = {'epoch': 1, 'train_loss': 3.123456, 'validation_loss': 2.654321,
                        'best_score': 2 / 9, 'validation': self.result()}
            source = folder / 'metrics/epoch_001.json'
            source.write_text(json.dumps(artifact), encoding='utf-8')
            original = source.read_bytes()
            # An unfinished next epoch must never be printed.
            (folder / 'metrics/epoch_002.json.tmp').write_text('{', encoding='utf-8')
            output = folder / 'evaluation_summary.log'
            self.assertEqual(module.append_completed_epochs(folder, output), [1])
            initial = output.read_bytes()
            self.assertEqual(module.append_completed_epochs(folder, output), [])
            self.assertEqual(initial, output.read_bytes())
            artifact['epoch'] = 2
            (folder / 'metrics/epoch_002.json').write_text(json.dumps(artifact), encoding='utf-8')
            self.assertEqual(module.append_completed_epochs(folder, output), [2])
            self.assertEqual(output.read_text(encoding='utf-8').count('Epoch 2 |'), 1)
            self.assertEqual(original, source.read_bytes())


if __name__ == '__main__':
    unittest.main()
