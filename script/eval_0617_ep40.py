import os, sys, yaml, torch, numpy as np
sys.path.insert(0, '.')
os.chdir("/mnt/f/mambacd/home/HICD_v7")

import torch.nn as nn
import torch.nn.functional as F
from models import build_hicd_v7
from datasets.dataset_v7 import FullImageDataset
from evaluation.icd_eval_v6 import ICDEvaluatorV6

with open('datasets/configs/0617final.yaml', 'r', encoding='utf-8') as f:
    dc = yaml.safe_load(f)

mc = dict(dc)
mc['backbone'] = {'type': 'dinov3', 'variant': 'dinov3_base', 'pretrained': False, 'pretrained_path': None}
mc['hidden_dim'] = 128
mc['embed_dim'] = 64
mc['active_heads'] = dc.get('active_heads', {})

device = torch.device('cuda')
model = build_hicd_v7(mc).to(device)
ckpt = torch.load('checkpoints/v7_0617_dinov3_freeze20/checkpoint_epoch_40.pth', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()
print(f'Loaded epoch {ckpt.get("epoch", "?")}')

evaluator = ICDEvaluatorV6(dc)
scenes = ['Airports', 'Ports', 'Urban-Rural Areas']

for scene in scenes:
    ds = FullImageDataset(data_dir='/mnt/d/CD/0617final', config=dc, split='val', scenes=[scene], img_size=256)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)
    for batch in loader:
        t1 = batch['img_t1'].to(device)
        t2 = batch['img_t2'].to(device)
        targets = {}
        label = batch['label'].to(device)
        if label.dim() == 3:
            ls = label.unsqueeze(1).float()
            ls = F.interpolate(ls, scale_factor=0.25, mode='nearest').squeeze(1).long()
            targets['target_gt'] = ls
        cs = batch['change_state'].to(device)
        if cs.dim() == 3:
            cs_s = cs.unsqueeze(1).float()
            cs_s = F.interpolate(cs_s, scale_factor=0.25, mode='nearest').squeeze(1).long()
            targets['state_gt'] = cs_s
            targets['change_state'] = cs_s
        targets['instance_labels'] = [{'labels': torch.zeros(0, dtype=torch.long, device=device), 'boxes': torch.zeros(0, 4, device=device)} for _ in range(t1.shape[0])]
        with torch.no_grad():
            preds, loss, ldict = model(t1, t2, targets)
        evaluator.update(preds, targets, ldict)

results = evaluator.compute()
print(evaluator.format_results(results))