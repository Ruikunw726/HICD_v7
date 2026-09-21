import os, sys, yaml, torch, numpy as np, glob
sys.path.insert(0, '/mnt/f/mambacd/home/HICD_v7')
os.chdir('/mnt/f/mambacd/home/HICD_v7')

import torch.nn.functional as F
from PIL import Image
from models import build_hicd_v7

with open('datasets/configs/0617final.yaml', 'r', encoding='utf-8') as f:
    dataset_config = yaml.safe_load(f)

model_config = dict(dataset_config)
model_config['backbone'] = {'type': 'dinov3', 'variant': 'dinov3_base', 'pretrained': False, 'pretrained_path': None}
model_config['hidden_dim'] = 128
model_config['embed_dim'] = 64
model_config['active_heads'] = dataset_config.get('active_heads', {})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_hicd_v7(model_config).to(device)
ckpt = torch.load('checkpoints/v7_0617_dinov3_freeze20/checkpoint_epoch_40.pth', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()
print('Loaded epoch', ckpt.get('epoch', '?'))

COLORS = np.array([[0,0,0],[34,139,34],[128,128,128],[192,192,192],[169,169,169],[255,0,0],[255,165,0],[139,69,19],[0,128,0],[0,0,255],[0,191,255],[65,105,225],[128,0,128],[255,255,0],[0,128,128],[255,0,255]], dtype=np.uint8)
STATE_COLORS = np.array([[200,200,200],[255,0,0],[255,165,0],[0,255,0],[0,0,255],[255,255,0],[0,0,0]], dtype=np.uint8)

out_dir = '/mnt/f/mambacd/home/HICD_v7/inference_results/v7_0617_ep40'
os.makedirs(out_dir, exist_ok=True)

scenes = ['Airports', 'Ports', 'Urban-Rural Areas']
for scene in scenes:
    img_dir = f'/mnt/d/CD/0617final/{scene}/test/image'
    lbl_dir = f'/mnt/d/CD/0617final/{scene}/test/label'
    pre_files = sorted(glob.glob(f'{img_dir}/*_pre_war.tif'))[:3]
    for t1_path in pre_files:
        name = os.path.basename(t1_path).replace('_pre_war.tif', '')
        t2_path = f'{img_dir}/{name}_post_war.tif'
        lp = f'{lbl_dir}/{name}_target.tif'
        if not os.path.exists(t2_path):
            print(f'Skip {name}: no post')
            continue
        if not os.path.exists(lp):
            print(f'Skip {name}: no label')
            continue
        t1 = np.array(Image.open(t1_path)).astype(np.float32) / 255.0
        t2 = np.array(Image.open(t2_path)).astype(np.float32) / 255.0
        gt = np.array(Image.open(lp)).astype(np.int64)
        H, W = t1.shape[:2]
        ph = (32 - H % 32) % 32
        pw = (32 - W % 32) % 32
        if ph > 0 or pw > 0:
            t1 = np.pad(t1, ((0,ph),(0,pw),(0,0)), mode='reflect')
            t2 = np.pad(t2, ((0,ph),(0,pw),(0,0)), mode='reflect')
        t1t = torch.from_numpy(t1).permute(2,0,1).unsqueeze(0).float().to(device)
        t2t = torch.from_numpy(t2).permute(2,0,1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            preds, _, _ = model(t1t, t2t, None)
            tl = preds['target_logits']
            sl = preds.get('state_logits', None)
            if tl.shape[2] != H or tl.shape[3] != W:
                tl = F.interpolate(tl, size=(H,W), mode='bilinear', align_corners=False)
            if sl is not None and (sl.shape[2] != H or sl.shape[3] != W):
                sl = F.interpolate(sl, size=(H,W), mode='bilinear', align_corners=False)
            pt = tl.argmax(1).squeeze(0).cpu().numpy()
            ps = sl.argmax(1).squeeze(0).cpu().numpy() if sl is not None else None
        t1v = (t1[:H,:W]*255).clip(0,255).astype(np.uint8)
        t2v = (t2[:H,:W]*255).clip(0,255).astype(np.uint8)
        gc = COLORS[gt[:H,:W].clip(0,15)]
        pc = COLORS[pt[:H,:W].clip(0,15)]
        tag = scene.replace(' ','_').replace('-','_')
        Image.fromarray(t1v).save(f'{out_dir}/{tag}_{name}_pre.png')
        Image.fromarray(t2v).save(f'{out_dir}/{tag}_{name}_post.png')
        Image.fromarray(gc).save(f'{out_dir}/{tag}_{name}_gt.png')
        Image.fromarray(pc).save(f'{out_dir}/{tag}_{name}_pred.png')
        if ps is not None:
            sc = STATE_COLORS[ps[:H,:W].clip(0,6)]
            Image.fromarray(sc).save(f'{out_dir}/{tag}_{name}_state.png')
        print(f'Saved: {tag}_{name}')
print('Done:', out_dir)