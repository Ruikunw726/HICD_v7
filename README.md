# HICD v7 — Hierarchical Instance Change Detection

基于 DINOv3 + 图结构标签空间的双分支语义变化检测框架, 专注于战争损伤评估场景。

## 核心创新

1. **DINOv3 骨干网络** — 用视觉基础模型 DINOv3-Base 替代 VSSM/LWGANet, 冻结前6层(S0-S5)保留通用特征, 训练后6层(S6-S11)适配遥感域, 86.4M参数
2. **图结构标签空间 (Graph-Structured Label Space)** — 15类目标 × 7类状态, 不是所有(target, state)组合都合法, 通过图结构嵌入 + validity masking 约束组合空间, 对比学习编码标签关系
3. **层级条件解码器 (HierarchicalConditionDecoder)** — 状态预测以目标预测为条件, 可学习 target→state 偏置矩阵, 无效组合的 logits 被 mask 抑制
4. **NFA + TFM 时序融合** — Neighbor Feature Aggregation 多尺度特征聚合, Temporal Fusion Module 保留 T1/T2 独立特征用于交互
5. **双分支交互** — ChangeGuidedROI 用变化特征引导实例 RoI, SpatialModulation 用实例热图反馈像素特征, Consistency Loss 约束实例-像素状态一致性
6. **Copy-Paste 增强** — 对稀有类别(Rare Classes)进行实例粘贴, 平衡类别分布, 粘贴后自动更新状态标签
7. **条件状态解码** — 区别于独立分类, 状态预测以目标预测为条件信号, 通过 einsum 实现 target→state 偏置, 有效降低非法组合预测概率
8. **Separated Kappa (SeK)** — SCD 标准指标, 分别计算变化/未变化组的 Kappa 系数后取平均, 避免未变化像素主导评估

## 数据集

### 0617final — 自建战争损伤数据集

| 场景 | Train | Val | Test | 说明 |
|------|-------|-----|------|------|
| Airports | ~2000 | ~500 | ~500 | 机场跑道/停机坪/塔台 |
| Ports | ~1500 | ~400 | ~400 | 港口码头/船坞/仓库 |
| Urban-Rural | ~2500 | ~600 | ~700 | 城市建筑/农田/公路 |

- **目标类别**: 15类 (Farmland, Runway, Taxiway, Apron, Bridge, Highway, Building, Shelter, Tower, Pier, Dock, Tank, Aircraft, Vessel, Crater)
- **变化状态**: 7类 (no-change, damaged, reduced, added, extended, replaced, none)
- **标注格式**: 像素级 TIF (raw label 0-68, 通过 decode_map 映射到 target + state)

### SECOND — 语义变化检测基准

| 类别 | Train | Val | Test |
|------|-------|-----|------|
| Total | 2077 | 445 | - |

- 6类像素标签, 3类状态 (no-change/increase/decrease)
- 已切分为 256×256 patch (带重叠)

### XBD — 建筑物损伤评估

- 仅建筑物目标, 5级损毁等级
- 35GB, 已切分为 256×256 patch

## 模型架构

`
T1 (Pre-war) ─┐
               ├─→ DINOv3 Backbone (frozen S0-S5, trainable S6-S11)
T2 (Post-war)─┘         │
                         ▼
                   NFA × 2 → TFM (fused + T1/T2 independent features)
                         │
          ┌──────────────┼──────────────┐
          ▼              │              ▼
   Instance Adapter      │       Semantic Adapter
          │              │              │
          ▼              │              ▼
  ┌──────────────┐       │     ┌──────────────────┐
  │  Instance     │◄──────┴────►│   Pixel Branch    │
  │  Branch       │  交互层     │                   │
  │               │             │ ├─Target Head     │ → 15类目标分割
  │ RoI Align     │             │ ├─State Decoder   │ → 7类状态(target-conditioned)
  │ + Cosine Sim  │             │ └─Change Head     │ → 变化/未变化
  │ + State Emb   │             │                   │
  └──────────────┘             └──────────────────┘
`

### 交互层细节

| 模块 | 功能 |
|------|------|
| ChangeGuidedROI | 变化特征引导实例 RoI 特征, 让实例分支关注变化区域 |
| SpatialModulation | 实例热图调制像素特征, 让像素分支获取实例级空间信息 |
| LabelStructure | 图结构嵌入 + validity masking + contrastive loss |
| Consistency Loss | 实例状态应与对应区域内像素状态一致 |

## 项目结构

`
HICD_v7/
├── models/
│   ├── HICD_v7.py                  # 主模型
│   ├── backbone/
│   │   ├── dinov3.py               # DINOv3 backbone (Safetensors 加载)
│   │   └── lwganet.py              # LWGANet backbone (备选)
│   ├── heads/
│   │   ├── semantic_head.py        # FPN + Target/State/Change 三头
│   │   ├── proposal_head.py        # Instance Refinement (RoI + Cosine Sim)
│   │   └── instance_head.py        # DETR-style 实例头 (legacy)
│   ├── modules/
│   │   ├── nfa.py                  # Neighbor Feature Aggregation
│   │   ├── tfm.py                  # Temporal Fusion Module
│   │   ├── label_structure.py      # 图标签空间 + HierarchicalConditionDecoder
│   │   ├── interaction.py          # Cross-branch interaction
│   │   └── clip_module.py          # CLIP 文本编码 (可选)
│   └── losses/
│       └── v7_loss.py              # 统一损失 (pixel + state + change + instance + contrastive)
├── datasets/
│   ├── dataset_v7.py               # 数据加载 + copy-paste + 增强
│   └── configs/
│       ├── 0617final.yaml          # 标签映射 + 类别权重 + 状态权重
│       ├── second.yaml             # SECOND 配置
│       └── xbd.yaml                # XBD 配置
├── evaluation/
│   ├── icd_eval_v6.py              # 评估器 (mIoU + SeK + per-class)
│   └── icd_protocol.py             # ICD 评估协议
├── script/
│   ├── train_full_v7.py            # 训练脚本 (tqdm + CSV 日志)
│   ├── tta_eval.py                 # TTA 推理
│   ├── cut_patches.py              # 数据切分工具
│   └── infer_v7.py                 # 推理脚本
└── weights/
    └── model.safetensors           # DINOv3-Base 预训练权重 (需单独下载)
`

## 快速开始

### 环境配置

`ash
conda create -n mamba python=3.11 -y && conda activate mamba
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install timm safetensors opencv-python-headless tqdm pyyaml
`

### 下载权重

`ash
# DINOv3-Base 预训练权重
# 从 Hugging Face 下载 dinov3_vitb14 的 model.safetensors
# 放到 weights/model.safetensors
`

### 训练

`ash
cd HICD_v7
export PYTHONPATH=F:\mambacd:
conda activate mamba

# 0617final 数据集
python -B script/train_full_v7.py \
  --dataset 0617final \
  --data_dir /mnt/d/CD/0617final \
  --backbone_type dinov3 \
  --backbone dinov3_base \
  --batch_size 2 \
  --img_size 256 \
  --max_epochs 50 \
  --learning_rate 1e-4 \
  --freeze_backbone_epochs 10 \
  --use_amp \
  --num_workers 2 \
  --exp_name v7_0617_run1

# SECOND 数据集
python -B script/train_full_v7.py \
  --dataset second \
  --data_dir /mnt/c/SECOND_6class_v2 \
  --backbone_type dinov3 \
  --backbone dinov3_base \
  --batch_size 2 \
  --img_size 256 \
  --max_epochs 50 \
  --learning_rate 1e-4 \
  --freeze_backbone_epochs 10 \
  --use_amp \
  --num_workers 2 \
  --exp_name v7_second_run1
`

### 评估指标

| 指标 | 说明 |
|------|------|
| Target mIoU | 15类目标的平均交并比 |
| State mIoU | 7类状态的平均交并比 (非背景像素) |
| SeK | Separated Kappa, SCD 标准指标 |
| Kappa Changed / Unchanged | 分组 Kappa |
| Per-class IoU | 每类目标单独 IoU |

### 输出说明

| 输出 | Shape | 说明 |
|------|-------|------|
| target_logits | (B, 15, H/4, W/4) | 目标类别 logits |
| state_logits | (B, 7, H/4, W/4) | 变化状态 logits |
| change_logits | (B, 2, H/4, W/4) | 变化/未变化 logits |
| inst_state_logits | (N, 7) | 实例级状态 logits |

## 标签映射 (0617final)

### 目标类别

| target_id | Target | Raw Label 范围 |
|-----------|--------|---------------|
| 0 | Farmland | 1-2 |
| 1 | Runway | 3-7 |
| 2 | Taxiway | 8-12 |
| 3 | Apron | 13-17 |
| 4 | Bridge | 18-22 |
| 5 | Highway | 23-27 |
| 6 | Building | 28-32 |
| 7 | Shelter | 33-37 |
| 8 | Tower | 38-42 |
| 9 | Pier | 43-47 |
| 10 | Dock | 48-52 |
| 11 | Tank | 53-57 |
| 12 | Aircraft | 57-61 |
| 13 | Vessel | 62-66 |
| 14 | Crater | 67-68 |

### 变化状态

| state_id | State | 说明 |
|----------|-------|------|
| 0 | no-change | 未变化 |
| 1 | damaged | 损伤 |
| 2 | reduced | 减少 |
| 3 | added | 新增 |
| 4 | extended | 扩展 |
| 5 | replaced | 替换 |
| 6 | none | 无状态 (背景/弹坑) |

## 损失函数

`
Total Loss = w_pixel × (Target CE + Dice)
           + w_state × (State CE + Dice) × 2.0
           + w_change × (Change CE + Dice)
           + w_consist × (Instance Consistency + Box Consistency)
           + w_contrastive × (Label Structure Contrastive Losses)
`

| 损失 | 权重 | 说明 |
|------|------|------|
| Target CE + Dice | 1.0 | 目标分割 (Focal + Dice) |
| State CE + Dice | 2.0 | 状态分类 (加权 CE, ignore bg=255) |
| Change CE + Dice | 1.0 | 变化检测 |
| Instance Consistency | 1.0 | 实例-像素状态一致性 |
| Box Consistency | 1.0 | bbox 内像素一致性 |
| Contrastive | 0.5 | 图结构嵌入对比学习 |

## 版本历史

| 版本 | 日期 | 主要改动 |
|------|------|----------|
| V1-V4 | 2026-07 | VSSM + CLIP, 实例级检测 (详见主仓库) |
| V5 | 2026-08-03 | 双分支解码器: 实例检测 + 语义分割, Task-Specific Adapters |
| V6 | 2026-08 | 移除 CLIP, 简化架构 |
| V7 | 2026-08-28 | DINOv3 backbone, 图结构标签空间, HierarchicalConditionDecoder, 双分支交互, copy-paste 增强, SeK 指标 |

## 依赖

- Python 3.11 (conda env: mamba)
- PyTorch 2.x + CUDA 12.6
- timm, safetensors, opencv-python-headless, tqdm, pyyaml

## 致谢

- [DINOv3](https://github.com/mini-DINO/DINOv3) — 视觉基础模型骨干网络
- [SegEarth-OV](https://github.com/Lsan17/SegEarth-OV) — 余弦相似度变化检测启发
- [HyperCLIP-CD](https://github.com/) — 图结构标签空间思路启发
