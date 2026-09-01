import os, sys, yaml, torch, numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import build_hicd_v7
from datasets import create_dataloader

with open('datasets/configs/second.yaml', 'r', encoding='utf-8') as f:
    dc = yaml.safe_load(f)
mc = dict(dc)
mc['backbone'] = {'variant': 'L1', 'pretrained': False, 'pretrained_path': None}
mc['hidden_dim'] = 128
mc['active_heads'] = dc.get('active_heads', {})
mc['pixel_class_weights'] = [2.931, 0.160, 0.061, 2.452, 0.090, 0.307]
device = torch.device('cuda')
model = build_hicd_v7(mc).to(device)
ckpt = torch.load('checkpoints/v7_second_fix/best_model.pth', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

val_loader = create_dataloader(data_dir='/mnt/c/SECOND_6class_v2', config=dc, split='val', batch_size=1, num_workers=0, img_size=1024)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
match_counts = {t: 0 for t in thresholds}
total_gt = 0

with torch.no_grad():
    for bi, batch in enumerate(val_loader):
        if bi >= 20: break
        img_t1 = batch['img_t1'].to(device)
        img_t2 = batch['img_t2'].to(device)
        feat_t1 = model.backbone(img_t1)
        feat_t2 = model.backbone(img_t2)
        nfa_t1 = model.nfa_t1(*feat_t1)
        nfa_t2 = model.nfa_t2(*feat_t2)
        fused, _, _ = model.tfm(nfa_t1, nfa_t2)
        pf = model.semantic_adapter(fused)
        hm, _ = model.semantic_head(pf)
        probs = torch.softmax(hm, dim=1)
        bprob = probs[0, 4].cpu().numpy()
        bmask = (bprob > 0.5).astype(np.uint8)
        labeled, nf = ndimage.label(bmask)
        pred_boxes = []
        for i in range(1, nf + 1):
            ys, xs = np.where(labeled == i)
            if len(ys) < 50: continue
            H, W = bmask.shape
            pred_boxes.append([(xs.min()+xs.max())/2/W, (ys.min()+ys.max())/2/H, (xs.max()-xs.min())/W, (ys.max()-ys.min())/H])

        gt_boxes = batch['instance_boxes'][0]
        if gt_boxes is None or len(gt_boxes) == 0 or len(pred_boxes) == 0:
            total_gt += len(gt_boxes) if gt_boxes is not None else 0
            continue
        gt_np = gt_boxes.numpy()
        total_gt += len(gt_np)
        for gi in range(len(gt_np)):
            gcx,gcy,gw,gh = gt_np[gi]
            gx1,gy1,gx2,gy2 = gcx-gw/2,gcy-gh/2,gcx+gw/2,gcy+gh/2
            ga = gw*gh
            for pi, pb in enumerate(pred_boxes):
                pcx,pcy,pw,ph = pb
                px1,py1,px2,py2 = pcx-pw/2,pcy-ph/2,pcx+pw/2,pcy+ph/2
                pa = pw*ph
                ix1,iy1 = max(gx1,px1),max(gy1,py1)
                ix2,iy2 = min(gx2,px2),min(gy2,py2)
                inter = max(0,ix2-ix1)*max(0,iy2-iy1)
                iou = inter/max(ga+pa-inter,1e-6)
                for t in thresholds:
                    if iou >= t:
                        match_counts[t] += 1
                        break

print(f'Total GT: {total_gt}')
for t in thresholds:
    recall = match_counts[t] / max(total_gt, 1)
    print(f'  IoU>={t:.1f}: matched={match_counts[t]:4d}  recall={recall:.3f}')
