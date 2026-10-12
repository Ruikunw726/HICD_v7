from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from datasets.dataset_v7 import FullImageDataset


class ResolvedCatalogTests(unittest.TestCase):
    def make(self,folder):
        view=folder/'view';(view/'test').mkdir(parents=True)
        name_file=view/'test/names.txt';name_file.write_text('sample_A\n')
        source=folder/'source';source.mkdir()
        record={'base_name':'sample_A','scene':'','inst_key':None,
                'pre_path':str(source/'pre.tif'),'post_path':str(source/'post.tif'),'label_path':str(source/'label.tif')}
        for key in ('pre_path','post_path','label_path'):Path(record[key]).write_bytes(b'fixture')
        nh=hashlib.sha256(name_file.read_bytes()).hexdigest()
        cat={'schema_version':'hicd.resolved_split.v1','split':'test','names_sha256':nh,'samples':[record]}
        cp=view/'test/resolved_samples.json';cp.write_text(json.dumps(cat))
        manifest={'catalog_sha256':{'test':hashlib.sha256(cp.read_bytes()).hexdigest()},'names_sha256':{'test':nh}}
        (view/'split_manifest.json').write_text(json.dumps(manifest))
        obj=FullImageDataset.__new__(FullImageDataset);obj.data_dir=str(view);obj.split='test'
        return obj,name_file,cp

    def test_exact_paths_without_sibling_search(self):
        with tempfile.TemporaryDirectory() as d:
            obj,names,cp=self.make(Path(d))
            sample=obj._collect_samples_from_names(str(names))[0]
            self.assertEqual(sample['base_name'],'sample_A')
            self.assertTrue(sample['pre_path'].endswith('pre.tif'))

    def test_changed_names_rejected_not_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            obj,names,cp=self.make(Path(d));names.write_text('other_name\n')
            with self.assertRaisesRegex(ValueError,'names changed'):obj._collect_samples_from_names(str(names))

    def test_changed_catalog_rejected_not_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            obj,names,cp=self.make(Path(d));cp.write_text('{}')
            with self.assertRaisesRegex(ValueError,'catalog changed'):obj._collect_samples_from_names(str(names))


if __name__=='__main__':unittest.main()
