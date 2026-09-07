# HICD v7 — Hierarchical Instance Change Detection

基于 DINOv3 + GNN 图结构标签空间的战争损毁语义变化检测。

## 核心创新

1. **DINOv3 Backbone** — DINOv3-Base 替换 LWGANet，冻结 S0-S5，训练 S6-S11（86.4M参数）
2. **GNN 图结构标签空间** — 15类目标 × 7类状态，2层GAT + validity MLP，学习合法(target, state)组合
3. **层级条件解码器** — 状态预测基于目标条件，soft validity mask (0.1 + 0.9×validity)
4. **动态状态损失** — GNN validity 过滤非法 (target, state) 组合，逐像素动态权重
5. **NFA + TFM 时序融合** — 多尺度特征聚合，保留 T1/T2 独立特征
6. **频域分支（FrequencyBranch）** — 独立于backbone的频域特征提取，Haar DWT多级分解，增强小目标检测
7. **自适应类权重** — 基于验证IoU自动调整，解决类别不平衡问题
8. **Copy-Paste 增强** — 稀有类别实例粘贴，更新状态标签
9. **Separated Kappa (SeK)** — SCD标准指标，平衡变化/未变化评估

## 数据集

### 0617final — 战争损毁数据集

| 场景 | 训练集 | 验证集 | 测试集 | 描述 |
|------|--------|--------|--------|------|
| 机场 | ~770 | ~165 | ~165 | 跑道、停机坪、塔台 |
| 港口 | ~420 | ~89 | ~89 | 栈桥、船坞、仓库 |
| 城乡 | ~5590 | ~1199 | ~1199 | 建筑、农田、公路 |

- **目标类别**: 15类 (农田、跑道、滑行道、停机坪、桥梁、公路、建筑、掩体、塔台、栈桥、船坞、罐体、飞机、舰船、弹坑)
- **状态类别**: 7类 (无变化、损毁、缩减、新增、扩展、替换、无)
- **标注**: 像素级TIF (原始标签0-68，通过decode_map映射)

### SECOND — 语义变化检测基准
- 6类像素标签，3类状态，256×256 patches

### XBD — 建筑损毁评估
- 仅建筑，5个损毁等级，256×256 patches

## 模型架构

    T1/T2 --> DINOv3 Backbone (冻结 S0-S5, 训练 S6-S11)
           --> 4尺度特征 [P2, P3, P4, P5]
           --> NFA_t1 / NFA_t2
           --> TFM (时序融合)
           --> PixelStateHead
           |    |-- FPN + 多尺度融合
           |    |-- Target Head (15类)
           |    |-- HierarchicalConditionDecoder (7类)
           |    |-- target_logits, state_logits
           |
           +--> FrequencyBranch (独立频域分支)
           |    |-- Haar DWT 2级分解
           |    |-- HF分支 (边缘/纹理/小目标)
           |    |-- LF分支 (整体结构)
           |    |-- 时相差分
           |    |-- 增强P2层特征
           |
           --> VotingRefinementModule (推理时后处理)
           --> GNNLabelStructure (2层GAT, validity MLP)
           --> validity_matrix [15, 7]

## 损失函数

- **Target**: Focal CE + Dice, AdaptiveClassWeighter
- **State**: CE + Dice, GNN动态mask + 独立AdaptiveClassWeighter
- **GNN监督**: Validity BCE (对比损失)

注：精炼模块(VotingRefinementModule)仅在推理时使用，不参与训练损失计算。

## 快速开始

    conda create -n mamba python=3.11 -y && conda activate mamba
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    pip install timm safetensors opencv-python-headless tqdm pyyaml scipy

    cd HICD_v7 && export PYTHONPATH=$(pwd):$PYTHONPATH

    python -B script/train_full_v7.py \
      --dataset 0617final --data_dir /path/to/0617final \
      --backbone_type dinov3 --backbone dinov3_base \
      --batch_size 32 --img_size 256 --max_epochs 50 \
      --learning_rate 1e-4 --freeze_backbone_epochs 0 \
      --use_amp --num_workers 4 --exp_name v7_0617_run1

## 评估指标

| 指标 | 描述 |
|------|------|
| Target mIoU | 15类目标的平均IoU |
| State mIoU | 7类状态的平均IoU（非背景） |
| SeK | Separated Kappa = (Kappa_changed + Kappa_unchanged) / 2 |
| Kappa Changed | 变化状态的多类Kappa |
| Kappa Unchanged | 未变化状态的二值Kappa |

## 最新实验结果

### 0617final 数据集 (v7_0617_freq_branch, epoch 50)

| 指标 | 值 |
|------|-----|
| Target mIoU | 0.693 |
| State mIoU | 0.505 |
| SeK | 0.708 |

**各类别IoU:**

| 类别 | IoU | 类别 | IoU |
|------|-----|------|-----|
| 农田 | 96.1% | 建筑 | 99.1% |
| 跑道 | 93.8% | 掩体 | 56.1% |
| 滑行道 | 83.6% | 塔台 | 0% |
| 停机坪 | 91.9% | 栈桥 | 63.2% |
| 桥梁 | 82.5% | 船坞 | 41.7% |
| 公路 | 95.2% | 罐体 | 87.0% |
| 飞机 | 64.8% | 舰船 | 80.1% |
| 弹坑 | 0% | | |

## 项目结构

    HICD_v7/
      models/
        HICD_v7.py                 主模型
        backbone/dinov3.py         DINOv3 backbone
        heads/proposal_head.py     VotingRefinementModule (推理后处理)
        modules/label_structure.py GNN标签结构 + PixelStateHead + ConditionDecoder
        modules/nfa.py             邻域特征聚合
        modules/tfm.py             时序融合模块
        modules/frequency_branch.py 频域分支 (DWT + HF/LF编码)
        losses/v7_loss.py          统一损失函数 (自适应 + 动态状态)
      datasets/
        dataset_v7.py              DataLoader + Copy-Paste增强
        configs/                   各数据集YAML配置
      evaluation/
        icd_eval_v6.py             评估器 (mIoU + SeK)
      script/
        train_full_v7.py           训练脚本
        infer_v7.py                推理脚本

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| V1-V4 | 2026-07 | VSSM + CLIP, 实例检测 |
| V5 | 2026-08-03 | 双分支解码器, Task-Specific Adapters |
| V6 | 2026-08 | 移除CLIP, 简化架构 |
| V7.0 | 2026-08-28 | DINOv3 backbone, 图结构标签空间 |
| V7.1 | 2026-09-02 | VotingRefinementModule, 自适应权重 |
| V7.2 | 2026-09-03 | GNN标签空间, 移除change分支, 动态状态损失 |
| V7.3 | 2026-09-07 | 频域分支(FrequencyBranch), 精炼模块改为推理后处理, GNN validity约束 |

## 依赖

- Python 3.11 (conda env: mamba)
- PyTorch 2.x + CUDA 12.6
- timm, safetensors, opencv-python-headless, tqdm, pyyaml, scipy

## 致谢

- [DINOv3](https://github.com/mini-DINO/DINOv3) — 视觉基础模型backbone
- [SegEarth-OV](https://github.com/Lsan17/SegEarth-OV) — 余弦相似度变化检测启发
- [HyperCLIP-CD](https://github.com/) — 图结构标签空间启发
