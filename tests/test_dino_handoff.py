"""Condition gate regression: no launch while baseline train/test is unfinished."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dino_handoff',ROOT/'script/check_dino_handoff.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)


class HandoffTests(unittest.TestCase):
    def fixture(self,root,status):
        run=root/'baselines_compare/runs_joint/bisrnet_t1t3_20261011_164416'
        run.mkdir(parents=True);run.with_name(run.name+'.status.txt').write_text(status)
        return run

    def test_training_finish_alone_does_not_allow_launch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p,'TRAINING_FINISHED x python_exit=0 tee_exit=0\nTEST_STARTED x')
            self.assertEqual(gate.check(p)['state'],'WAIT')

    def test_failed_test_blocks_launch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p,'TRAINING_FINISHED x python_exit=0 tee_exit=0\nTEST_FINISHED x python_exit=1 tee_exit=0')
            self.assertEqual(gate.check(p)['state'],'FAILED')

    def test_full_test_and_100_epochs_required(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);run=self.fixture(p,'TRAINING_FINISHED x python_exit=0 tee_exit=0\nTEST_FINISHED x python_exit=0 tee_exit=0')
            (run/'manifest.json').write_text(json.dumps({'run_identity':'fixture'}))
            (run/'metrics').mkdir()
            for i in range(100):(run/'metrics'/f'epoch_{i+1:03d}.json').write_text('{}')
            out=p/'baselines_compare/results_joint'/f'{run.name}_test'
            out.mkdir(parents=True)
            v={'split':'test','producer_manifest':{'run_identity':'fixture'},'metrics':{'valid_pixels':13077*4096,'ignored_pixels':0}}
            (out/'test_strict.json').write_text(json.dumps(v))
            self.assertEqual(gate.check(p)['state'],'READY')
            v['metrics']['valid_pixels']-=1;(out/'test_strict.json').write_text(json.dumps(v))
            with self.assertRaises(ValueError):gate.check(p)
