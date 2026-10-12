from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.strict_joint_protocol import StrictJointEvaluator
from evaluation.farmland_semantics import decode_farmland_state, recompute_report
from evaluation.farmland_semantics import FarmlandNoChangeModelEvaluator


class FarmlandSemanticsTests(unittest.TestCase):
    def report(self, pt, ps, gt, gs):
        return StrictJointEvaluator().update(pt, ps, gt, gs).compute()

    def test_wrong_targets_and_background_fp_remain_errors(self):
        pt=np.array([0,0,6,0]);ps=np.array([2,1,0,6])
        gt=np.array([0,6,0,15]);gs=np.array([0,1,0,6])
        raw=self.report(pt,ps,gt,gs)
        fixed=recompute_report(raw)
        self.assertEqual(fixed['strict_joint_micro']['tp'],1)
        self.assertEqual(fixed['strict_joint_micro']['fp'],3)
        self.assertEqual(fixed['strict_joint_micro']['fn'],2)
        self.assertEqual(fixed['per_target']['6']['strict_joint_micro']['tp'],0)
        self.assertFalse(fixed['prediction_policy']['ground_truth_used_for_decoding'])

    def test_random_count_replay_equals_pixel_decode_including_diagnostics(self):
        rng=np.random.default_rng(42)
        gt=rng.integers(0,16,3000);gs=rng.integers(0,7,3000);gs[gt==0]=0
        gt[::17]=255;gs[::23]=255
        pt=rng.integers(0,16,3000);ps=rng.integers(0,7,3000)
        raw=self.report(pt,ps,gt,gs)
        expected=self.report(pt,decode_farmland_state(pt,ps),gt,gs)
        actual=recompute_report(raw)
        for field in ('strict_joint_micro','strict_joint_macro_f1','per_target','per_target_state','tasks','joint_confusion'):
            self.assertEqual(actual[field],expected[field])
        self.assertEqual(recompute_report(actual),actual)
        self.assertEqual(raw,self.report(pt,ps,gt,gs))

    def test_gt_definition_conflict_is_rejected_not_relabelled(self):
        raw=self.report(np.array([0]),np.array([1]),np.array([0]),np.array([1]))
        with self.assertRaisesRegex(ValueError,'GT contains'):
            recompute_report(raw)

    def test_model_adapter_and_torch_prediction_decode(self):
        import torch
        from evaluation.strict_joint_protocol import TARGET_NAMES, STATE_NAMES
        config={'pixel_target_names':TARGET_NAMES,'state_names':STATE_NAMES,'num_pixel_targets':16,'num_states':7}
        evaluator=FarmlandNoChangeModelEvaluator(config)
        pt=torch.tensor([[[0,6]]]);ps=torch.tensor([[[2,1]]])
        predictions={'target_logits':torch.nn.functional.one_hot(pt,16).permute(0,3,1,2).float(),
                     'state_logits':torch.nn.functional.one_hot(ps,7).permute(0,3,1,2).float()}
        evaluator.update(predictions,{'target_gt':pt,'state_gt':torch.tensor([[[0,1]]])})
        self.assertEqual(evaluator.compute()['strict_joint_f1'],1)
        self.assertEqual(decode_farmland_state(pt,ps).tolist(),[[[0,1]]])
        self.assertEqual(ps.tolist(),[[[2,1]]])
        with self.assertRaises(ValueError):decode_farmland_state(pt.float(),ps)


if __name__=='__main__':unittest.main()
