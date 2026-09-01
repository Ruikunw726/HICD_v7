# HICD v7 — Hierarchical Instance Change Detection

Semantic Change Detection framework with dual-branch architecture and graph-structured label space, focused on war damage assessment.

## Architecture

T1/T2 images are processed through a DINOv3 backbone (frozen S0-S5, trainable S6-S11), producing multi-scale features that flow through NFA and TFM for temporal fusion. The fused features split into two branches:

**Pixel Branch** (Semantic):
- FPN + multi-scale fusion
- Target Head: 15-class target region segmentation
- State Decoder: 7-class state classification (target-conditioned via HierarchicalConditionDecoder)
- Change Head: cosine similarity between T1/T2 features

**Instance Branch**:
- Instance Refinement Module: RoI Align + cosine similarity with learnable state embeddings
- Proposals extracted from pixel branch predictions (connected components)
- Per-object state classification (no-change / damaged / reduced)

**Cross-Branch Interaction**:
- ChangeGuidedROI: change features guide instance RoI features
- SpatialModulation: instance heatmap modulates pixel features
- Consistency Loss: instance state should match dominant pixel state within each RoI

## Key Innovations

1. **Graph-Structured Label Space**: 15 target classes x 7 state classes with validity constraints. Not all (target, state) combinations are valid. Graph embeddings + contrastive learning encode label relationships.

2. **Hierarchical Condition Decoder**: State prediction conditioned on target prediction. Learnable target-to-state bias matrix + validity masking suppresses invalid combinations.

3. **DINOv3 Backbone**: Frozen early layers preserve general features, trainable later layers adapt to remote sensing domain. 86.4M total params, 43.3M trainable.

4. **Copy-Paste Augmentation**: Rare class instances are pasted onto training images to balance class distribution.

## Datasets

| Dataset | Path | Targets | States | Description |
|---------|------|---------|--------|-------------|
| 0617final | D:\CD17final | 15 | 7 | Custom war damage dataset, 3 scenes (airport/port/urban) |
| SECOND | C:\SECOND_6class_v2 | 6 | 3 | Semantic change detection benchmark |
| XBD | D:\CD½ | 1 | 5 | xBD building damage assessment |

## Training



## Evaluation Metrics

- **Target mIoU**: Mean IoU across 15 target classes
- **State mIoU**: Mean IoU across 7 state classes (non-background pixels only)
- **SeK (Separated Kappa)**: Average of Kappa for changed and unchanged groups separately. Standard SCD metric.
- **Per-class IoU**: Individual class IoU values

## Project Structure

HICD_v7/
  models/
    HICD_v7.py              - Main model
    backbone/dinov3.py      - DINOv3 backbone
    heads/semantic_head.py  - FPN + Target/State/Change heads
    heads/proposal_head.py  - Instance refinement module
    modules/nfa.py          - Neighbor Feature Aggregation
    modules/tfm.py          - Temporal Fusion Module
    modules/label_structure.py - Graph label space + HierarchicalConditionDecoder
    modules/interaction.py  - Cross-branch interaction
    losses/v7_loss.py       - Unified loss function
  datasets/
    dataset_v7.py           - Data loading + copy-paste augmentation
    configs/0617final.yaml  - Dataset config (label mapping, class weights)
  evaluation/
    icd_eval_v6.py          - Evaluator (mIoU + SeK + per-class)
  script/
    train_full_v7.py        - Training script
  weights/
    model.safetensors       - DINOv3-Base pretrained weights

## Dependencies

- Python 3.11 (conda env: mamba)
- PyTorch 2.x + CUDA 12.6
- DINOv3 pretrained weights (download to weights/model.safetensors)
