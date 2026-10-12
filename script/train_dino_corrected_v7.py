"""Fresh strict trainer with an explicit versioned DINO adapter, same other mechanics."""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import train_strict_v7 as legacy
from models.dino_corrected_model import build_dino_corrected_hicd
from models.backbone.dinov3_corrected import INTEGRATION_VERSION

original_prepare = legacy.prepare_run


def prepare_run(args):
    import yaml
    cfg = yaml.safe_load((ROOT/args.config_dir/f'{args.dataset}.yaml').read_text())
    if cfg.get('dino_integration_version') != INTEGRATION_VERSION:
        raise ValueError('Missing corrected DINO version')
    original_prepare(args)
    run = ROOT/args.save_dir/args.exp_name
    manifest = json.loads((run/'manifest.json').read_text())
    for source in [Path(__file__),ROOT/'launch_dino_corrected.sh',ROOT/'script/check_dino_handoff.py',
                   ROOT/'script/check_dino_ready.py',ROOT/'script/verify_dino_integration.py',
                   ROOT/'script/verify_dino_corrected_real_cpu.py']:
        manifest['file_sha256'][str(source.resolve())] = legacy.sha256(source)
        dest = run/'snapshot/source'/source.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    provenance = ROOT/'docs/experiments/dino_integration_20261012/weight_provenance.json'
    proof = json.loads(provenance.read_text())
    if not proof['all_model_tensors_equal']:
        raise RuntimeError('DINO pretrained provenance verification did not pass')
    if legacy.sha256(args.pretrained_weight_path) != proof['local_sha256']:
        raise RuntimeError('Local pretrained weight differs from verified provenance')
    manifest['file_sha256'][str(provenance)] = legacy.sha256(provenance)
    shutil.copy2(provenance,run/'snapshot/weight_provenance.json')
    manifest['dino_integration'] = dict(version=INTEGRATION_VERSION,
        input='RGB [0,1] normalized inside DINO only; LVD1689M ImageNet mean/std',
        feature_collection='timm forward_intermediates after-block [2,5,8,11], norm=True',
        legacy_experiment_preserved=True,weight_provenance=proof)
    import hashlib
    manifest['run_identity'] = hashlib.sha256(json.dumps(
        {'args':manifest['args'],'hashes':manifest['file_sha256']},sort_keys=True).encode()).hexdigest()
    legacy.atomic_json(run/'manifest.json',manifest)
    print(f"CORRECTED_FRESH_RUN_PREPARED identity={manifest['run_identity']} integration={INTEGRATION_VERSION}",flush=True)


if __name__ == '__main__':
    legacy.build_hicd_v7 = build_dino_corrected_hicd
    legacy.prepare_run = prepare_run
    legacy.main()
