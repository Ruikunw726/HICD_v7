"""User-scoped restoration: original membership, no renaming/content dedup/shuffle."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from datasets.dataset_v7 import FullImageDataset


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def lines(path):return [v.strip() for v in Path(path).read_text().splitlines() if v.strip()]
def parents(values):return {re.sub(r'_\d{2}_\d{2}$','',v) for v in values}
def overlaps(values):return {a+'__'+b:len(set(values[a])&set(values[b])) for a,b in [('train','val'),('train','test'),('val','test')]}
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--audit-dir',required=True)
    args=parser.parse_args();source=Path(args.source).resolve();output=Path(args.output).resolve();audit=Path(args.audit_dir).resolve()
    assert output.parent==source.parent and output!=source and not output.exists()
    audit.mkdir(parents=True,exist_ok=False);(audit/'before').mkdir()
    config=yaml.safe_load((ROOT/'datasets/configs/0617final_t123_trainfix.yaml').read_text())
    old={s:lines(source/s/'names.txt') for s in ('train','val','test')}
    original={s:[n for n in lines(source/s/'names.txt.bak') if not n.startswith('synth_')] for s in old}
    assert set(old['train'])==set(original['train'])
    restored={'train':old['train'],'val':original['val'],'test':original['test']}
    for s in restored:
        assert len(restored[s])==len(set(restored[s])) and set(restored[s])<=set(old[s])
        shutil.copy2(source/s/'names.txt',audit/'before'/f'{s}_names.txt')
        shutil.copy2(source/s/'names.txt.bak',audit/'before'/f'{s}_names.txt.bak')
    assert all(v==0 for v in overlaps(restored).values())
    assert all(v==0 for v in overlaps({s:parents(v) for s,v in restored.items()}).values())
    records={}
    for s in restored:
        data=FullImageDataset(str(source),config,split=s,img_size=256)
        index={v['base_name']:v for v in data.samples}
        assert len(index)==len(old[s])
        records[s]=[index[name] for name in restored[s]]
        assert all(all(r.get(key) for key in ('pre_path','post_path','label_path')) for r in records[s])
        print('RESTORED_RESOLVED',s,len(records[s]),flush=True)
    # No image files are copied or deleted. View is unavailable until final manifest exists.
    output.mkdir();name_hash={};catalog_hash={}
    for s in restored:
        (output/s).mkdir()
        namefile=output/s/'names.txt'
        if s=='train':shutil.copy2(source/s/'names.txt',namefile)
        else:shutil.copy2(source/s/'names.txt.bak',namefile)
        assert lines(namefile)==restored[s]
        name_hash[s]=sha(namefile)
        catalog=output/s/'resolved_samples.json'
        dump(catalog,{'schema_version':'hicd.resolved_split.v1','split':s,'names_sha256':name_hash[s],
                      'samples':records[s]})
        catalog_hash[s]=sha(catalog)
    shutil.copy2(source/'scene_state_info.json',output/'scene_state_info.json')
    manifest={'version':'hicd.original_real_membership_restored.v1',
        'created_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'source':str(source),'user_scope':'No renamed samples; restore original real-only membership without further dedup or shuffle',
        'before_counts':{s:len(old[s]) for s in old},'published_counts':{s:len(restored[s]) for s in restored},
        'removed_injected_counts':{s:len(set(old[s])-set(restored[s])) for s in old},
        'names_sha256':name_hash,'catalog_sha256':catalog_hash,'scene_metadata_sha256':sha(output/'scene_state_info.json'),
        'checks':{'cross_names':overlaps(restored),'cross_parent_tiles':overlaps({s:parents(v) for s,v in restored.items()}),
                  'all_members_resolved':True,'train_names_byte_identical':sha(source/'train/names.txt')==name_hash['train'],
                  'val_test_original_backup_byte_identical':all(sha(source/s/'names.txt.bak')==name_hash[s] for s in ('val','test'))},
        'parent_definition':'Remove only terminal _NN_NN tile coordinates; preserve long parent identifiers/dates',
        'city_prefix_overlap':overlaps({s:{'_'.join(n.split('_')[:2]) for n in restored[s]} for s in restored}),
        'city_level_holdout':False,'content_hash_dedup_performed':False,
        'scope':['T1','T2','T3'],'old_live_names_images_results_unchanged':True,
        'old_models_require_retraining_for_formal_new_split_comparison':True,
        'source_paths_pinned':True,'source_images_not_copied_or_made_immutable':True}
    manifest['split_identity']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    dump(output/'split_manifest.json',manifest);dump(audit/'summary.json',dict(manifest,output=str(output)))
    dump(audit/'removed_members.json',{s:sorted(set(old[s])-set(restored[s])) for s in old})
    print('PUBLISHED_RESTORED_ORIGINAL_SPLIT',manifest['published_counts'],manifest['split_identity'],flush=True)


if __name__=='__main__':main()
