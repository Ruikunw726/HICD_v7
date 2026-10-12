"""One real CPU train update and validation batch before waiting for the GPU."""
import json
import sys
from pathlib import Path
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'script')]
from models.dino_corrected_model import build_dino_corrected_hicd
from datasets import create_dataloader
from evaluation.strict_joint_evaluator import StrictJointModelEvaluator
from evaluation.farmland_semantics import recompute_report
from torch.nn import functional as F


def main():
    torch.set_num_threads(2);torch.manual_seed(20261012)
    config=yaml.safe_load((ROOT/'datasets/configs/0617final_t123_dino_corrected_20261012.yaml').read_text())
    config.update(backbone=dict(type='dinov3',variant='dinov3_base',pretrained=True,
        pretrained_path=str(ROOT/'weights/model.safetensors')),hidden_dim=128,embed_dim=64,w_pixel=1.,w_state=2.,w_contrastive=.05)
    model=build_dino_corrected_hicd(config).cpu().train()
    model.config['_current_epoch']=0;model.loss_fn.set_phase(2,config)
    root='/mnt/d/CD/0617final_256_v2_disjoint_20261009'
    train=create_dataloader(data_dir=root,config=config,split='train',batch_size=2,num_workers=0,img_size=256,state_aware=True)
    batch=next(iter(train))
    gt={dest:F.interpolate(batch[src][:,None].float(),scale_factor=.25,mode='nearest')[:,0].long()
        for src,dest in [('label','target_gt'),('change_state','state_gt')]}
    optimizer=torch.optim.AdamW(model.get_param_groups(lr=1e-4,weight_decay=1e-4))
    qkv=model.backbone.blocks[-1].attn.qkv.weight
    before=qkv.detach().clone()
    predictions,loss,parts=model(batch['img_t1'],batch['img_t2'],gt)
    assert torch.isfinite(loss) and all(torch.isfinite(p).all() for p in predictions.values() if isinstance(p,torch.Tensor))
    loss.backward()
    assert qkv.grad is not None and torch.isfinite(qkv.grad).all() and qkv.grad.abs().sum()>0
    grad=float(qkv.grad.abs().sum())
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step();assert not torch.equal(before,qkv.detach())
    del before,predictions,optimizer
    val=create_dataloader(data_dir=root,config=config,split='val',batch_size=2,num_workers=0,img_size=256)
    vbatch=next(iter(val));model.eval()
    with torch.inference_mode():predictions,_,_=model(vbatch['img_t1'],vbatch['img_t2'])
    gt={dest:F.interpolate(vbatch[src][:,None].float(),scale_factor=.25,mode='nearest')[:,0].long()
        for src,dest in [('label','target_gt'),('change_state','state_gt')]}
    evaluator=StrictJointModelEvaluator(config,state_head='raw')
    evaluator.update(predictions,gt)
    report=recompute_report(evaluator.compute())
    assert report['valid_pixels']+report['ignored_pixels']==2*64*64
    receipt=dict(passed=True,device='CPU',real_train_samples=2,real_validation_samples=2,
        train_loss=float(loss),loss_parts=parts,last_block_qkv_grad_l1=grad,last_block_updated=True,
        outputs={k:list(predictions[k].shape) for k in ['target_logits','state_logits']},
        validation=report,actual_training_will_reinitialize=True,performance_claim=False)
    path=ROOT/'docs/experiments/dino_integration_20261012/real_cpu_smoke.json'
    path.write_text(json.dumps(receipt,indent=2,allow_nan=False))
    print('CORRECTED_REAL_CPU_SMOKE_PASS',float(loss),'last-block updated; val count8192; not a performance result',flush=True)


if __name__=='__main__':main()
