"""One-shot precondition check; does not start, stop or modify any experiment."""
import json
from pathlib import Path


def check(root):
    run = root/'baselines_compare/runs_joint/bisrnet_t1t3_20261011_164416'
    status_path = run.with_name(run.name+'.status.txt')
    status = status_path.read_text() if status_path.exists() else ''
    if ('USER_TERMINATED' in status or 'failed' in status.lower() or
            any('FINISHED' in line and 'python_exit=0 tee_exit=0' not in line
                for line in status.splitlines())):
        return {'state':'FAILED','reason':'Baseline interrupted or ended unsuccessfully; do not start a replacement automatically'}
    if not all(any(line.startswith(stage) and 'python_exit=0 tee_exit=0' in line for line in status.splitlines())
               for stage in ['TRAINING_FINISHED','TEST_FINISHED']):
        return {'state':'WAIT','reason':'Baseline training and full test must both finish successfully'}
    path=root/'baselines_compare/results_joint'/f'{run.name}_test/test_strict.json'
    if not path.is_file():
        return {'state':'WAIT','reason':'Baseline test artifact not yet published'}
    result=json.loads(path.read_text())
    manifest=json.loads((run/'manifest.json').read_text())
    producer=result['producer_manifest']
    if result['split']!='test' or producer['run_identity']!=manifest['run_identity']:
        raise ValueError('Baseline test identity mismatch')
    m=result['metrics']
    if m['valid_pixels']+m['ignored_pixels']!=13077*64*64:
        raise ValueError('Baseline test pixel/sample coverage mismatch')
    if len(list((run/'metrics').glob('epoch_*.json')))!=100:
        raise ValueError('Expected all 100 baseline validation epochs')
    return {'state':'READY','reason':'Baseline100 and complete test succeeded','test_file':str(path)}


if __name__=='__main__':
    root=Path(__file__).resolve().parents[3]
    value=check(root);print(json.dumps(value))
    raise SystemExit({'READY':0,'WAIT':75,'FAILED':2}[value['state']])
