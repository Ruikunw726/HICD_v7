"""Verify frozen CPU acceptance and proposed launch inputs before any GPU use."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
directory=ROOT/'docs/experiments/dino_integration_20261012'
if not (directory/'READY.md').is_file():
    raise SystemExit('Corrected DINO acceptance has not been published')
r=json.loads((directory/'ready_manifest.json').read_text())
for name,digest in r['source_sha256'].items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Prepared training input changed; needs review: '+name)
for name in ['cpu_acceptance.json','real_cpu_smoke.json']:
    assert json.loads((directory/name).read_text())['passed']
assert json.loads((directory/'weight_provenance.json').read_text())['all_model_tensors_equal']
print('CORRECTED_DINO_READY_INPUTS_VERIFIED')
