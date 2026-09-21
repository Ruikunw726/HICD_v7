# HICD v9 — Hierarchical Instance Change Detection

基于 DINOv3 + GNN 图结构标签空间的战争损毁语义变化检测。

## 核心创新

1. **DINOv3 Backbone** — DINOv3-Base，冻结S0-S5，训练S6-S11（86.4M参数）
2. **GNN图结构标签空间** — 16类目标 × 7类状态，2层GAT + validity MLP
3. **层级条件解码器** — 状态预测基于目标条件，soft validity mask
4. **Background Suppression Loss** — 背景抑制损失，解决背景假阳性问题
5. **BDD协议评估** — F_loc/F_clf/F_overall + SeK，对标WD-CD论文
6. **稀有类过采样** — Bridge/Shelter/Dock样本重复注入训练集
7. **频域分支** — Haar DWT多级分解，增强小目标检测
8. **自适应类权重** — 基于验证指标自动调整

## 数据集

### 0617final_256_v2 — 战争损毁数据集

| 配置 | 值 |
|------|-----|
| 训练样本 | 83,900 (含稀有类过采样) |
| 验证样本 | 30,127 |
| 测试样本 | 30,127 |
| 目标类别 | 16类 (14目标 + Tower忽略 + Background) |
| 状态类别 | 7类 (no-change/damaged/reduced/added/extended/replaced/none) |
| 图像尺寸 | 256×256 |

**目标类别**:
- 地表: Farmland, Runway, Taxiway, Apron, Bridge, Highway
- 建筑: Building, Shelter, Pier, Dock, Tank
- 移动: Aircraft, Vessel
- 其他: Crater, Background
- 忽略: Tower (样本过少)

**稀有类过采样** (v9新增):
| 类别 | 原始样本 | 过采样后 |
|------|---------|---------|
| Bridge | 35 | 1,855 (50x) |
| Shelter | 46 | 4,990 (50x) |
| Dock | 66 | 6,442 (50x) |

## 模型架构

```
T1/T2 → DINOv3 Backbone (冻结S0-S5, 训练S6-S11)
       → 4尺度特征 [P2, P3, P4, P5]
       → NFA + TFM (时序融合)
       → PixelStateHead
       |   ├── FPN + 多尺度融合
       |   ├── Target Head (16类)
       |   ├── HierarchicalConditionDecoder (7类状态)
       |   └── BackgroundSuppressionLoss
       |
       → FrequencyBranch (频域特征)
       → GNNLabelStructure (2层GAT, validity MLP)
```

## 损失函数

```
L_total = L_pixel + L_state + L_validity + L_bg_suppress

L_pixel = CE + 2×Dice + 0.5×BackgroundSuppression
L_state = CE + Dice (GNN动态mask)
L_bg_suppress: 惩罚背景像素被预测为前景
```

## 评估协议

### BDD协议 (对标WD-CD论文)

| 指标 | 公式 | 描述 |
|------|------|------|
| F_loc | target vs background F1 | 定位能力 |
| F_clf | ND vs D classification F1 | 状态分类能力 |
| F_overall | 0.3×F_loc + 0.7×F_clf | 综合指标 |
| SeK | exp(IOU2-1)×(ρ-η)/(1-η) | Separated Kappa |

**状态折叠**:
- ND (未损毁): no-change, added, extended
- D (损毁): damaged, reduced

### 对标WD-CD论文 (Table 4)

| 方法 | F_loc | F_clf | F_overall |
|------|-------|-------|-----------|
| DeepSiameseNet | 72.99 | 72.23 | 72.46 |
| ChangeFormer | 81.54 | 73.54 | 75.94 |
| ChangeMamba | 81.54 | 74.05 | 76.30 |
| CD-Lamba (SOTA) | 83.04 | 78.03 | **79.53** |
| **HICD v8** | 80.28 | 71.65 | 74.24 |
| **HICD v9** | 训练中 | - | - |

## 快速开始

```bash
# 环境
conda create -n mamba python=3.11 -y && conda activate mamba
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install timm safetensors opencv-python-headless tqdm pyyaml scipy

cd HICD_v9 && export PYTHONPATH=$(pwd):$PYTHONPATH

# 训练 (v9: 稀有类过采样)
bash launch_v9.sh

# 训练 (v8: 背景抑制)
bash launch_v8.sh

# 监控训练
screen -r hicd_v9
tail -f checkpoints/v9_rare_oversample/train.log
```

## 实验版本

| 版本 | 实验名 | 关键改进 | F_overall |
|------|--------|---------|-----------|
| v7 | v7_synth_dense | 高密度合成弹坑 | - |
| v8 | v8_bg_suppress | BackgroundSuppressionLoss | 78.49% |
| v9 | v9_rare_oversample | 稀有类过采样(Bridge/Shelter/Dock) | 训练中 |

## v8 → v9 改进

1. **稀有类问题诊断**
   - Bridge: train中0%样本 → 从源数据注入35个样本×50倍
   - Shelter: 46样本 → 4,990 (过采样)
   - Dock: 66样本 → 6,442 (过采样)

2. **数据加载修复**
   - 优先使用names.txt加载样本
   - 修复稀有类检测：使用原始label值(18-22/33-37/48-52)

3. **评估协议完善**
   - SeK修复：eta_hat=1.0时的nan问题
   - 新增Global SeK (所有像素)

## 项目结构

```
HICD_v9/
├── models/
│   ├── HICD_v7.py                 主模型
│   ├── backbone/dinov3.py         DINOv3 backbone
│   ├── losses/v7_loss.py          损失函数 (+BG suppression)
│   ├── instance_classifier.py     实例级分类器 (待训练)
│   └── modules/
│       ├── label_structure.py     GNN标签结构
│       ├── frequency_branch.py    频域分支
│       └── nfa.py, tfm.py        特征聚合/时序融合
├── datasets/
│   ├── dataset_v7.py              DataLoader (+稀有类过采样)
│   └── configs/0617final.yaml     数据集配置
├── evaluation/
│   ├── bdd_protocol.py            BDD协议评估
│   └── icd_eval_v6.py             评估器
├── script/
│   ├── train_full_v7.py           训练脚本
│   ├── train_instance.py          实例分类器训练
│   └── inject_from_source.py      稀有类注入
├── launch_v8.sh                   v8训练启动
├── launch_v9.sh                   v9训练启动
└── checkpoints/
    ├── v8_bg_suppress/            v8模型
    └── v9_rare_oversample/        v9模型 (训练中)
```

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| V1-V4 | 2026-07 | VSSM + CLIP, 实例检测 |
| V5-V6 | 2026-08 | 双分支解码器, 移除CLIP |
| V7.0-7.3 | 2026-08~09 | DINOv3, GNN标签空间, 频域分支 |
| V8.0 | 2026-09-18 | BDD协议评估, BackgroundSuppressionLoss |
| V8.1 | 2026-09-20 | SeK修复, Global SeK |
| V9.0 | 2026-09-20 | 稀有类过采样, 数据加载修复 |

## 关键发现

### 1. IoU vs BDD协议
- IoU只在GT范围内计算，忽略假阳性
- Building IoU=0.996 但 F_loc Precision只有0.227
- **根因**: `ignore_index=255`导致背景不参与loss

### 2. 背景假阳性问题
- 模型把1770万背景像素预测为Building
- **解决**: BackgroundSuppressionLoss (v8)
- F_overall从~70%提升到78.49%

### 3. 稀有类学习问题
- Bridge/Shelter/Dock在训练集中样本极少
- **解决**: 从源数据注入 + 50倍过采样 (v9)

### 4. 域差距分析
- 跨数据集(0.873) > 跨场景(0.771) > 跨城市(0.619)
- DINOv3: S5是通用性峰值，冻结S0-S5冻错了层
- 建议: 冻结深层(S6-S11)而非浅层

## 依赖

- Python 3.11 (conda env: mamba)
- PyTorch 2.x + CUDA 12.6
- timm, safetensors, opencv-python-headless, tqdm, pyyaml, scipy
- GDAL (用于读取TIF标签)

## 致谢

- [DINOv3](https://github.com/facebookresearch/dinov2) — 视觉基础模型
- [WD-CD](https://github.com/) — 战争损毁数据集与BDD协议
- [xView2/xBD](https://github.com/DIUx-xView) — 建筑损毁评估基准
