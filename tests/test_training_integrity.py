"""Regression checks for audit A01-A06: parameter updates and supervision integrity."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'script'))
from datasets import dataset_v7 as data
from models.HICD_v7 import HICD_v7
from models.modules.label_structure import GNNLabelStructure, LabelStructure, HierarchicalConditionDecoder
from models.backbone.dinov3 import DINOv3Backbone
from train_full_v7 import TrainerV7


def dataset(raw, size, augment=True, crop=None):
    obj = data.FullImageDataset.__new__(data.FullImageDataset)
    obj.samples = [dict(pre_path='pre', post_path='post', label_path='label', scene='', base_name='fixture')]
    obj.img_size, obj.augment, obj.crop_sampler = size, augment, crop
    obj._img_cache, obj._cache_max = {}, 0
    obj.state_decode_map = {int(v): int(v)+20 for v in np.unique(raw)}
    obj.label_decode_map = {int(v): int(v) for v in np.unique(raw)}
    obj.config = {'pixel_target_names': ['Background']}
    obj.instances_json, obj.boxes_dir = None, None
    return obj


class TrainingIntegrityTests(unittest.TestCase):
    def sample(self, obj, raw):
        image = np.repeat(raw[None, :, :], 3, axis=0).astype(np.uint8)
        with ExitStack() as stack:
            stack.enter_context(patch.object(data, 'read_image', return_value=image.copy()))
            stack.enter_context(patch.object(data, 'read_label', return_value=raw.copy()))
            for name in ('color_jitter', 'random_erasing', 'gaussian_noise', 'random_blur'):
                stack.enter_context(patch.object(data, name, side_effect=lambda value, **kw: value))
            stack.enter_context(patch.object(data, 'copy_paste_both_labels',
                               side_effect=lambda a,b,c,d,**kw:(a,b,c,d,None)))
            return obj[0]

    def test_target_state_and_images_share_geometry_for_100_seeds(self):
        raw = np.arange(16).reshape(4,4).astype(np.uint8)
        for seed in range(100):
            np.random.seed(seed)
            result = self.sample(dataset(raw, 4), raw)
            np.testing.assert_array_equal(result['label'], result['change_state'] - 20)
            np.testing.assert_array_equal(result['label'], (result['img_t1'][0] * 255).round().long())
            np.testing.assert_array_equal(result['label'], (result['img_t2'][0] * 255).round().long())

    def test_crop_bottom_right_matches_both_labels_and_images(self):
        raw = np.arange(16).reshape(4,4).astype(np.uint8)
        crop = SimpleNamespace(get_crop_center=lambda *args:(3,3))
        obj = dataset(raw, 2, crop=crop)
        with patch.object(data, 'random_augment', side_effect=lambda a,b,c,d,e:(a,b,c,d,e)):
            result = self.sample(obj, raw)
        expected = raw[2:,2:]
        np.testing.assert_array_equal(result['label'], expected)
        np.testing.assert_array_equal(result['change_state'], expected + 20)
        np.testing.assert_array_equal((result['img_t1'][0] * 255).round(), expected)

    def test_unknown_raw_is_ignore_not_farmland(self):
        obj = dataset(np.array([[0]],dtype=np.uint8),1)
        self.assertEqual(obj._decode_label(np.array([[201]]))[0,0],255)
        self.assertEqual(obj._decode_state(np.array([[201]]))[0,0],255)

    def test_label_read_failure_is_visible(self):
        obj = dataset(np.ones((4,4),dtype=np.uint8),4)
        with patch.object(data,'read_image',return_value=np.zeros((3,4,4),dtype=np.uint8)), \
             patch.object(data,'read_label',side_effect=OSError('broken label')):
            with self.assertRaisesRegex(OSError,'broken label'):
                obj[0]

    def test_paired_copy_paste_retains_replaced_and_never_uses_farmland(self):
        labels = np.full((24,24),15,dtype=np.int64)
        labels[:8,:8] = 12
        labels[18:,18:] = 0
        states = np.full((24,24),6,dtype=np.int64); states[:8,:8] = 5
        first = np.zeros((3,24,24),dtype=np.float32);first[:,:8,:8] = 0.25
        second = np.zeros_like(first);second[:,:8,:8] = 0.75
        np.random.seed(1)
        a,b,c,d,overrides = data.copy_paste_both_labels(first.copy(),second.copy(),labels.copy(),labels.copy(),
            p=1,target_classes=[12],max_paste=3,state_map=states,background_class=15,legal_states={12:{5}})
        changed = overrides>=0
        self.assertTrue(changed.any())
        self.assertTrue((labels[changed]==15).all())
        self.assertTrue((overrides[changed]==5).all())
        self.assertTrue((a[:,changed]==0.25).all())
        self.assertTrue((b[:,changed]==0.75).all())
        self.assertTrue((c[18:,18:]==0).all())
        self.assertTrue((d[changed]==12).all())

    def test_copy_paste_requires_source_state(self):
        images = np.zeros((3,24,24));labels = np.full((24,24),12)
        output=data.copy_paste_both_labels(images,images,labels,labels,p=1,target_classes=[12])
        self.assertIsNone(output[-1])

    def model(self, enabled=True):
        obj = HICD_v7.__new__(HICD_v7);nn.Module.__init__(obj)
        for name in ('backbone','nfa_t1','nfa_t2','tfm','instance_head','label_structure','semantic_head'):
            setattr(obj,name,nn.Linear(2,2))
        obj.backbone.weight.requires_grad_(False)
        obj.use_freq_branch=enabled
        if enabled:
            obj.freq_branch=nn.Linear(2,2);obj.freq_fusion=nn.Linear(2,2)
        return obj

    def test_frequency_and_frozen_parameters_registered_once_and_update(self):
        model=self.model();groups=model.get_param_groups(lr=0.1)
        registered=[p for group in groups for p in group['params']]
        self.assertEqual(len(registered),len(list(model.parameters())))
        optimizer=torch.optim.SGD(groups)
        before=model.freq_branch.weight.detach().clone()
        loss=model.freq_fusion(model.freq_branch(torch.ones(1,2))).sum()
        loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
        self.assertFalse(torch.equal(before,model.freq_branch.weight))
        self.assertIsNone(model.freq_branch.weight.grad)
        model.backbone.weight.requires_grad_(True)
        model.assert_optimizer_coverage(optimizer.param_groups)

    def test_optimizer_coverage_rejects_missing_and_duplicate(self):
        model=self.model();groups=model.get_param_groups()
        with self.assertRaises(RuntimeError):model.assert_optimizer_coverage(groups[:-1])
        with self.assertRaises(RuntimeError):model.assert_optimizer_coverage(groups+[groups[-1]])
        self.model(False).get_param_groups()

    def test_log_prior_does_not_promote_negative_invalid_logit(self):
        target=torch.tensor([[[[10.]], [[-10.]]]])
        state=torch.tensor([[[[-1.]], [[-2.]]]],requires_grad=True)
        validity=torch.tensor([[1.,0.],[1.,0.]],requires_grad=True)
        for klass in (LabelStructure,GNNLabelStructure):
            obj=klass.__new__(klass);nn.Module.__init__(obj)
            obj.get_validity_score=lambda:validity
            weighted,_=obj.mask_invalid_logits(target,state)
            self.assertEqual(weighted.argmax(1).item(),0)
            self.assertLess(weighted[0,1,0,0].item(),state[0,1,0,0].item())
            weighted.sum().backward(retain_graph=True)
        self.assertTrue(torch.isfinite(state.grad).all())

    def test_unfreeze_zero_is_noop(self):
        obj=DINOv3Backbone.__new__(DINOv3Backbone);nn.Module.__init__(obj)
        obj.test_layer=nn.Linear(2,2)
        for parameter in obj.parameters():parameter.requires_grad_(False)
        obj.unfreeze(0)
        self.assertFalse(any(p.requires_grad for p in obj.parameters()))
        with self.assertRaises(ValueError):obj.unfreeze(-0.1)

    def test_actual_decoder_applies_probability_prior_for_negative_logits(self):
        decoder=HierarchicalConditionDecoder(1,2,2,hidden_dim=2)
        class Constant(nn.Module):
            def forward(self,value):
                return torch.tensor([-1.,-2.]).reshape(1,2,1,1).expand(value.shape[0],2,*value.shape[2:])
        decoder.target_condition_proj=Constant()
        structure=SimpleNamespace(get_validity_score=lambda:torch.tensor([[1.,0.],[1.,0.]]))
        state,_=decoder(torch.zeros(1,1,2,2),torch.zeros(1,2,2,2),structure)
        self.assertTrue((state.argmax(1)==0).all())
        self.assertTrue((state[:,1]<-2).all())

    def test_t123_config_excludes_all_t4_supervision(self):
        config=yaml.safe_load((ROOT/'datasets/configs/0617final_t123_trainfix.yaml').read_text())
        allowed={0,1,2,3,5,6,12,13,15,255}
        self.assertTrue(set(config['label_decode_map'].values()).issubset(allowed))
        for raw,target in config['label_decode_map'].items():
            if target==255:self.assertEqual(config['state_decode_map'][raw],255)
        for target in (4,7,8,9,10,11,14):self.assertEqual(config['pixel_class_weights'][target],0)
        self.assertEqual(config['training_tasks'],['T1','T2','T3'])

    def test_accumulation_flushes_short_window_and_clears_gradients(self):
        class Model(nn.Module):
            def __init__(self):super().__init__();self.w=nn.Parameter(torch.tensor(0.))
            def forward(self,a,b,targets):return {},self.w*targets['coefficient'],{}
        evaluator=SimpleNamespace(reset=lambda:None,update=lambda *a:None,compute=lambda:{},format_results=lambda a:'')
        obj=TrainerV7.__new__(TrainerV7);obj.model=Model();obj.device=torch.device('cpu')
        obj.args=SimpleNamespace(grad_accum=2,use_amp=False,max_epochs=1)
        obj.optimizer=torch.optim.SGD(obj.model.parameters(),lr=0.1)
        obj.evaluator=evaluator
        obj.train_loader=[dict(img_t1=torch.zeros(1),img_t2=torch.zeros(1),coefficient=v) for v in (1.,3.,5.)]
        obj._prepare_targets=lambda batch:{'coefficient':batch['coefficient']}
        TrainerV7.train_epoch(obj,0)
        self.assertAlmostEqual(obj.model.w.item(),-0.7,places=6)
        self.assertIsNone(obj.model.w.grad)


if __name__=='__main__':unittest.main()
