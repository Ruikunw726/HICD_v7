"""CPU pretrained interface/gradient verification; never uses occupied GPU."""
import hashlib
import json
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.backbone.dinov3_corrected import DINOv3CorrectedBackbone,INTEGRATION_VERSION


def main():
    torch.set_num_threads(2);torch.manual_seed(20261012)
    out=ROOT/'docs/experiments/dino_integration_20261012';out.mkdir(parents=True,exist_ok=True)
    weight=ROOT/'weights/model.safetensors'
    reference=ROOT/'weights/audit_20261012/reference_lvd1689m.safetensors'
    local_state=load_file(str(weight));reference_state=load_file(str(reference))
    model=DINOv3CorrectedBackbone(pretrained=True,pretrained_path=str(weight)).cpu().eval()
    expected=set(model.backbone.state_dict())
    unequal=[k for k in expected if k not in reference_state or k not in local_state
             or not torch.equal(local_state[k],reference_state[k])]
    if unequal:raise RuntimeError(f'Pretrained model tensor mismatch: {unequal}')
    model.backbone.load_state_dict({k:local_state[k] for k in expected},strict=True)
    proof=dict(all_model_tensors_equal=True,model_tensor_count=len(expected),
        local_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        source='timm/vit_base_patch16_dinov3.lvd1689m',revision='c6a5fb7d12bbd3cf3b0079253141c3332aaed7da',
        ignored_local_auxiliary_keys=sorted(set(local_state)-expected),
        mean=[.485,.456,.406],std=[.229,.224,.225],timm_version=__import__('timm').__version__)
    assert proof['reference_sha256']=='1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332'
    (out/'weight_provenance.json').write_text(json.dumps(proof,indent=2))
    del local_state,reference_state
    x=torch.rand(1,3,256,256);before=x.clone()
    capture={}
    hooks=[]
    for i in model.extract_blocks:
        hooks.append(model.blocks[i].register_forward_hook(lambda mod,args,y,index=i:capture.__setitem__(index,y.detach().clone())))
    with torch.no_grad():tokens=model.extract_tokens(x)
    for hook in hooks:hook.remove()
    for i,y in zip(model.extract_blocks,tokens):
        expected_tokens=model.norm(capture[i])[:,model.num_prefix_tokens:]
        torch.testing.assert_close(y,expected_tokens,rtol=1e-6,atol=1e-6)
    outputs=model(x)
    assert [tuple(y.shape) for y in outputs]==[(1,64,64,64),(1,128,32,32),(1,256,16,16),(1,512,8,8)]
    loss=sum(y.square().mean() for y in outputs);loss.backward()
    grads={n:float(p.grad.abs().sum()) for n,p in model.blocks[-1].named_parameters() if p.grad is not None}
    assert grads and all(torch.isfinite(p.grad).all() for p in model.blocks[-1].parameters() if p.grad is not None)
    assert grads['attn.qkv.weight']>0
    assert model.norm.weight.grad is not None and torch.isfinite(model.norm.weight.grad).all() and model.norm.weight.grad.abs().sum()>0
    assert torch.equal(x,before)
    fixed=torch.tensor([.485,.456,.406]).view(1,3,1,1)
    torch.testing.assert_close(model.normalize_input(fixed),torch.zeros_like(fixed))
    receipt=dict(passed=True,integration_version=INTEGRATION_VERSION,device='CPU',real_pretrained_weights=True,
        input_unchanged=True,normalization_zero_at_mean=True,after_block_reference_hooks_match=True,
        prefix_tokens=model.num_prefix_tokens,output_shapes=[list(y.shape) for y in outputs],
        last_block_grad_l1=grads,final_norm_grad_l1=float(model.norm.weight.grad.abs().sum()),
        weight_provenance=proof,performance_claim=False)
    (out/'cpu_acceptance.json').write_text(json.dumps(receipt,indent=2))
    print('DINO_CORRECTED_CPU_ACCEPTANCE_PASS',len(expected),'pretrained tensors identical; after-block hooks and last-block/final-norm gradient passed',flush=True)


if __name__=='__main__':main()
