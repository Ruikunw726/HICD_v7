"""Full test inference from a verified epoch, raw64 + Farmland no-change decoding.

The default checkpoint/epoch preserve the original epoch50 invocation.
"""
import argparse
import copy
import csv
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import build_hicd_v7
from datasets import create_dataloader
from train_full_v7 import TrainerV7
from evaluation.strict_joint_evaluator import StrictJointModelEvaluator
from evaluation.strict_joint_protocol import StrictJointEvaluator
from evaluation.farmland_semantics import decode_farmland_state, recompute_report, POLICY_VERSION


def stamp():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def save_json(path, value):
    p = Path(path)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(p)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


class RawRefinementBypass(torch.nn.Module):
    def forward(self, state_logits_pixel, **kwargs):
        return state_logits_pixel, state_logits_pixel.new_zeros(()), []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--checkpoint-name', default='checkpoint_epoch_50.pth')
    parser.add_argument('--expected-epoch', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261009)
    parser.add_argument('--examples', type=int, default=6)
    opts = parser.parse_args()
    run, output = Path(opts.run_dir).resolve(), Path(opts.output_dir).resolve()
    assert opts.expected_epoch > 0 and 0 <= opts.examples
    checkpoint = (run / opts.checkpoint_name).resolve()
    assert checkpoint.parent == run, 'Checkpoint must be inside the selected run'
    lock = (run / '.strict_test.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'samples').mkdir()
    began = time.time()
    try:
        save_json(output / 'status.json', {'stage': 'loading', 'started_at': stamp(), 'pid': __import__('os').getpid()})
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert ckpt['epoch'] + 1 == opts.expected_epoch, 'Checkpoint epoch does not match requested epoch'
        manifest = json.loads((run / 'manifest.json').read_text())
        assert ckpt['run_identity'] == manifest['run_identity']
        assert ckpt['protocol_version'] == 'hicd.strict_joint.v1'
        if manifest.get('data_split_identity'):
            assert ckpt['data_split_identity'] == manifest['data_split_identity']
            assert ckpt['prediction_semantics_version'] == POLICY_VERSION
        config = copy.deepcopy(ckpt['dataset_config'])
        args = ckpt['args']
        cfg_path = run / 'snapshot' / f'{args.dataset}.yaml'
        assert config == yaml.safe_load(cfg_path.read_text())
        assert set(config['label_decode_map'].values()) <= {0,1,2,3,5,6,12,13,15,255}
        helper = TrainerV7.__new__(TrainerV7)
        helper.args, helper.dataset_config = args, config
        for name, expected in manifest['file_sha256'].items():
            if (name.startswith('/mnt/d/') or '/models/' in name or '/datasets/' in name
                    or name.endswith('/strict_joint_protocol.py')):
                assert sha(name) == expected, f'Training input changed: {name}'
        test_names = Path(args.data_dir) / 'test/names.txt'
        assert sha(test_names) == manifest['file_sha256'][str(test_names)]
        assert torch.cuda.is_available(), 'CUDA required'
        model = build_hicd_v7(helper._build_model_config())
        weight_key = 'loss_fn.state_ce.weight'
        if weight_key in ckpt['model_state_dict']:
            model.loss_fn.state_ce.weight = ckpt['model_state_dict'][weight_key].clone()
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        model.config['_current_epoch'] = opts.expected_epoch - 1
        model = model.to('cuda').eval()
        checkpoint_hash = sha(checkpoint)
        del ckpt
        loader = create_dataloader(data_dir=args.data_dir, config=config, split='test',
                                   batch_size=opts.batch_size, num_workers=opts.workers,
                                   img_size=args.img_size, boxes_dir=args.boxes_dir)
        count = len(loader.dataset)
        assert count == manifest['list_counts']['test'], 'Incomplete test split resolution'
        selected = sorted(np.random.default_rng(opts.seed).choice(count, size=opts.examples, replace=False).tolist())
        train_names = set((Path(args.data_dir) / 'train/names.txt').read_text().splitlines())
        val_names = set((Path(args.data_dir) / 'val/names.txt').read_text().splitlines())
        sampling = [{'dataset_index': i, 'sample_name': loader.dataset.samples[i]['base_name'],
                     'selection': 'uniform without replacement before inference; no quality/content filtering'} for i in selected]
        save_json(output / 'sampling.json', {'seed': opts.seed, 'population': count, 'selected': sampling})
        # Verify raw predictions do not depend on optional inference-only refinement.
        first = loader.dataset[0]
        one, two = first['img_t1'][None].cuda(), first['img_t2'][None].cuda()
        with torch.inference_mode():
            original_predictions, _, _ = model(one, two)
            model.instance_head = RawRefinementBypass().eval()
            bypass_predictions, _, _ = model(one, two)
            for key in ('target_logits', 'state_logits'):
                assert torch.allclose(original_predictions[key], bypass_predictions[key], rtol=1e-5, atol=1e-6)
                assert torch.equal(original_predictions[key].argmax(1), bypass_predictions[key].argmax(1))
        del one, two, original_predictions, bypass_predictions
        torch.cuda.reset_peak_memory_stats()
        raw_evaluator = StrictJointModelEvaluator(config, state_head='raw')
        groups = {'train_name_overlap': StrictJointEvaluator(), 'no_train_name_overlap': StrictJointEvaluator()}
        group_counts = dict.fromkeys(groups, 0)
        sample_details = []
        processed = 0
        save_json(output / 'manifest.json', {
            'run_identity': manifest['run_identity'], 'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_hash,
            'epoch':opts.expected_epoch, 'split':'test', 'expected_samples':count, 'config':config,
            'data_split_identity':manifest.get('data_split_identity'),
            'checkpoint_selection_criterion':manifest['best_selection_criterion'],
            'training_manifest_sha256':sha(run/'manifest.json'),
            'protocol_version':'hicd.strict_joint.v1', 'semantic_policy_version':POLICY_VERSION,
            'precision':'float32; autocast disabled', 'state_head':'raw state_logits',
            'GT_alignment':'nearest scale0.25 from256 to64; no pred-based mask',
            'batch_size':opts.batch_size,'workers':opts.workers,'seed':opts.seed,
            'refinement_bypassed':True,'raw_output_equivalence_checked':True,
            'selection':sampling,
            'code_sha256':{str(Path(__file__)):sha(__file__),
                           str(ROOT/'evaluation/farmland_semantics.py'):sha(ROOT/'evaluation/farmland_semantics.py'),
                           str(ROOT/'evaluation/strict_joint_evaluator.py'):sha(ROOT/'evaluation/strict_joint_evaluator.py'),
                           str(ROOT/'evaluation/strict_joint_protocol.py'):sha(ROOT/'evaluation/strict_joint_protocol.py')},
            'resources_at_start':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader']).decode(),
        })
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                predictions, _, _ = model(batch['img_t1'].cuda(non_blocking=True), batch['img_t2'].cuda(non_blocking=True))
                gt_t = F.interpolate(batch['label'][:,None].float(), scale_factor=.25, mode='nearest')[:,0].long()
                gt_s = F.interpolate(batch['change_state'][:,None].float(), scale_factor=.25, mode='nearest')[:,0].long()
                raw_evaluator.update(predictions, {'target_gt':gt_t,'state_gt':gt_s})
                pt = predictions['target_logits'].argmax(1).cpu().numpy()
                raw_ps = predictions['state_logits'].argmax(1).cpu().numpy()
                ps = decode_farmland_state(pt,raw_ps)
                gt,gs = gt_t.numpy(),gt_s.numpy()
                for j,name in enumerate(batch['base_name']):
                    index = processed+j
                    group = 'train_name_overlap' if name in train_names else 'no_train_name_overlap'
                    groups[group].update(pt[j],raw_ps[j],gt[j],gs[j]);group_counts[group]+=1
                    if index in selected:
                        number = selected.index(index)+1
                        npz_name = f'sample_{number:02d}.npz'
                        np.savez_compressed(output/'samples'/npz_name,
                            img_t1=(batch['img_t1'][j].numpy().transpose(1,2,0)*255).round().clip(0,255).astype(np.uint8),
                            img_t2=(batch['img_t2'][j].numpy().transpose(1,2,0)*255).round().clip(0,255).astype(np.uint8),
                            gt_target=gt[j],gt_state=gs[j],pred_target=pt[j],pred_state=ps[j],raw_pred_state=raw_ps[j])
                        single = StrictJointEvaluator().update(pt[j],ps[j],gt[j],gs[j]).compute()
                        sample_details.append({'sample_number':number,'dataset_index':index,'sample_name':name,
                            'data_file':npz_name,'train_name_overlap':name in train_names,'val_name_overlap':name in val_names,
                            'source_paths':loader.dataset.samples[index], 'metrics':single})
                processed += len(batch['base_name'])
                if batch_index%25==0 or processed==count:
                    progress={'stage':'running','processed_samples':processed,'expected_samples':count,
                              'elapsed_s':time.time()-began,'at':stamp()}
                    save_json(output/'status.json',progress)
                    print(f'TEST_PROGRESS {processed}/{count} elapsed={progress["elapsed_s"]:.1f}s',flush=True)
        assert processed==count and len(sample_details)==opts.examples
        raw = raw_evaluator.compute()
        decoded = recompute_report(raw)
        decoded['prediction_policy']['new_model_inference']=True
        save_json(output/'raw_test_metrics.json',raw)
        save_json(output/'test_metrics.json',{'split':'test','epoch':opts.expected_epoch,'samples':count,
            'checkpoint_sha256':checkpoint_hash,'run_identity':manifest['run_identity'],
            'data_split_identity':manifest.get('data_split_identity'),
            'new_full_test_inference':True,'raw':raw,'decoded':decoded,
            'same_name_overlap':{'train':group_counts['train_name_overlap'],
                'val':sum(sample['base_name'] in val_names for sample in loader.dataset.samples),
                'note':'No samples dropped; name-disjoint is not proof of scene independence'},
            'groups':{key:{'samples':group_counts[key],'decoded':recompute_report(value.compute())} for key,value in groups.items()},
            'elapsed_s':time.time()-began,'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20})
        save_json(output/'sample_details.json', sorted(sample_details,key=lambda x:x['sample_number']))
        with (output/'per_target.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['target','joint_P','joint_R','joint_F1','TP','FP','FN','support','localization_F1','state_on_GT_micro_F1'])
            for t in decoded['per_target'].values():
                m=t['strict_joint_micro'];writer.writerow([t['target_name'],m['precision'],m['recall'],m['f1'],m['tp'],m['fp'],m['fn'],m['support'],t['target_localization_f1'],t['state_on_gt_target']['micro']['f1']])
        with (output/'per_target_state.csv').open('w',newline='') as stream:
            fields=['target_name','state_name','precision','recall','f1','tp','fp','fn','support','predicted_support']
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(decoded['per_target_state'])
        save_json(output/'status.json',{'stage':'complete','finished_at':stamp(),'samples':processed,
            'elapsed_s':time.time()-began,'decoded_micro':decoded['strict_joint_f1'],'decoded_macro':decoded['strict_joint_macro_f1']})
        print('FULL_TEST_COMPLETE',decoded['strict_joint_f1'],decoded['strict_joint_macro_f1'],flush=True)
    except Exception:
        (output/'failure.txt').write_text(traceback.format_exc())
        save_json(output/'status.json',{'stage':'failed','at':stamp(),'elapsed_s':time.time()-began})
        raise


if __name__=='__main__':main()
