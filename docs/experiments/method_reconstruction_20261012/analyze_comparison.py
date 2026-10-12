"""Freeze a CPU-only validation comparison; never changes experiment inputs."""
import csv
import hashlib
import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
WORKSPACE = ROOT.parents[1]
H = ROOT / 'checkpoints/v12_strict_joint_t123_disjoint_e100_20261009'
B = WORKSPACE / 'baselines_compare/runs_joint/bisrnet_t1t3_20261011_164416'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix(v):
    q = [[0] * 106 for _ in range(106)]
    for e in v['joint_confusion']['nonzero_entries']:
        q[e['gt_joint_id']][e['pred_joint_id']] += e['count']
    tp = [q[i][i] for i in range(105)]
    fp = [sum(q[j][i] for j in range(106)) - tp[i] for i in range(105)]
    fn = [sum(q[i]) - tp[i] for i in range(105)]
    f1 = 2 * sum(tp) / (2 * sum(tp) + sum(fp) + sum(fn))
    active = [2 * tp[i] / (2 * tp[i] + fp[i] + fn[i])
              for i in range(105) if 2 * tp[i] + fp[i] + fn[i]]
    assert math.isclose(f1, v['strict_joint_f1'], abs_tol=1e-12)
    assert math.isclose(sum(active) / len(active), v['strict_joint_macro_f1'], abs_tol=1e-12)
    assert sum(map(sum, q)) == v['valid_pixels']
    assert v['valid_pixels'] + v['ignored_pixels'] == 13077 * 64 * 64
    assert len(active) == v['evaluable_joint_classes']
    return q


def summary(v):
    q = matrix(v)
    errors = dict(background_misses=0, wrong_target=0, same_target_wrong_state=0, background_false_positives=0)
    mobile = dict(background_misses=0, wrong_target=0, same_target_wrong_state=0)
    for g in range(106):
        for p, n in enumerate(q[g]):
            if g < 105:
                key = ('background_misses' if p == 105 else 'wrong_target' if g // 7 != p // 7
                       else 'same_target_wrong_state' if g != p else None)
                if key:
                    errors[key] += n
                    if g // 7 in [12, 13]:
                        mobile[key] += n
            elif p < 105:
                errors['background_false_positives'] += n
    assert sum(errors[k] for k in ['background_misses','wrong_target','same_target_wrong_state']) == v['strict_joint_micro']['fn']
    gt = [s for s in v['per_target_state'] if s['support']]
    return dict(micro=v['strict_joint_micro'], macro=v['strict_joint_macro_f1'],
        active_joint_classes=v['evaluable_joint_classes'], gt_supported_classes=len(gt),
        gt_supported_macro_DIAGNOSTIC_ONLY=sum(s['f1'] for s in gt)/len(gt),
        tasks={k: v['tasks'][k]['strict_joint_f1'] for k in ['T1','T2','T3']},
        targets={p['target_name']: {k:p[k] for k in ['strict_joint_micro','strict_joint_f1',
            'target_localization_f1','state_on_gt_target']} for p in v['per_target'].values()
            if p['target_id'] in [0,1,2,3,5,6,12,13]}, errors=errors,mobile_errors=mobile,
        rare_states=[s for s in v['per_target_state'] if s['target_id'] in [12,13] and s['state_id']==5])


hm,bm = read(H/'manifest.json'),read(B/'manifest.json')
for k in ['protocol_version','data_split_identity','prediction_semantics_version']:
    assert hm[k] == bm[k]
for k,v in bm['file_sha256'].items():
    if k.startswith('/mnt/d/'):
        assert hm['file_sha256'][k] == v
assert bm['reference_mismatches'] == []

sources = {
    'HICD44':H/'metrics/epoch_044.json', 'HICD90':H/'metrics/epoch_090.json',
    'HICD_macro56':H/'metrics/epoch_056.json',
    'BiSRNet44':B/'metrics/epoch_044.json', 'BiSRNet45':B/'metrics/epoch_045.json',
}
reports = {name:read(path)['validation' if name.startswith('HICD') else 'metrics'] for name,path in sources.items()}
qs = {name:matrix(v) for name,v in reports.items()}
reference_rows = list(map(sum,qs['HICD90']))
assert all(list(map(sum,q)) == reference_rows for q in qs.values())

# Recompute each baseline epoch available at inspection; hold the user-named44 comparison fixed.
baseline_paths = sorted((B/'metrics').glob('epoch_*.json'))
baseline_records = [read(p) for p in baseline_paths]
for r in baseline_records:
    matrix(r['metrics'])
assert [r['epoch'] for r in baseline_records] == list(range(1,len(baseline_records)+1))

code_hashes = {}
for name in ['models/backbone/dinov3.py','models/HICD_v7.py','models/modules/tfm.py',
             'models/modules/label_structure.py','models/losses/v7_loss.py',
             'datasets/dataset_v7.py','script/train_full_v7.py','script/train_strict_v7.py']:
    actual = sha(ROOT/name)
    expected = hm['file_sha256'].get('/mnt/f/mambacd/home/HICD_v7/'+name)
    code_hashes[name] = dict(sha256=actual, training_start_sha256=expected, startup_match=actual==expected)
    if name != 'script/train_strict_v7.py':
        assert actual == expected

result = dict(observed_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
    scope='Diagnostic and proposed reconstruction only; no model/config/metric/GPU mutation',
    hicd_run=H.name,baseline_run=B.name,baseline_epochs_observed=len(baseline_records),
    data_split_identity=hm['data_split_identity'],protocol=hm['protocol_version'],
    prediction_semantics=hm['prediction_semantics_version'],
    same_valid_GT_marginals_all_compared_reports=True,
    baseline_reports_independently_recomputed=len(baseline_records),
    compared={name:summary(v) for name,v in reports.items()},
    source_files={name:dict(path=str(path),sha256=sha(path)) for name,path in sources.items()},
    source_code=code_hashes,
    caveats=['No baseline test at this snapshot; never compare baseline val with HICD test.',
             'Different backbone/freeze/LR/loss: aligned evaluation is not a causal ablation.',
             'Macro GT-supported diagnostic cannot replace established all-active-combination macro.'])
(OUT/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT/'target_comparison.csv').open('w',newline='',encoding='utf-8-sig') as stream:
    writer=csv.writer(stream);writer.writerow(['model','target','precision','recall','joint_f1','localization_f1'])
    for name,v in reports.items():
        for target in v['per_target'].values():
            if target['target_id'] in [0,1,2,3,5,6,12,13]:
                s=target['strict_joint_micro'];writer.writerow([name,target['target_name'],s['precision'],s['recall'],s['f1'],target['target_localization_f1']])
with (OUT/'target_state_comparison.csv').open('w',newline='',encoding='utf-8-sig') as stream:
    writer=csv.writer(stream);writer.writerow(['model','target','state','TP','FP','FN','support','precision','recall','f1'])
    for name,v in reports.items():
        for s in v['per_target_state']:
            writer.writerow([name,s['target_name'],s['state_name'],s['tp'],s['fp'],s['fn'],s['support'],s['precision'],s['recall'],s['f1']])
print(json.dumps({name:{k:val[k] for k in ['micro','macro','tasks','errors','mobile_errors','gt_supported_macro_DIAGNOSTIC_ONLY']}
                  for name,val in result['compared'].items()},ensure_ascii=False))
print('COMPARISON_AUDIT_PASS',len(baseline_records),'baseline epochs; shared GT pixel marginals match')
