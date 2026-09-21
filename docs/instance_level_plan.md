# 实例级损毁解译方案

> 日期: 2026-09-15
> 状态: 方案定稿，待实施
> 服务论文: P1 (第5章 实例级范式)
> 核心原则: **像素级与实例级完全解耦，各自独立训练，推理时组合**

---

## 一、四任务划分与类别分工

| 任务 | 类别 | 处理范式 | 说明 |
|------|------|----------|------|
| **T1 建筑损毁** | Building | 像素级 | 大面积，像素级已达 0.987 |
| **T2 地表变化** | Farmland, Runway, Taxiway, Apron, Highway, Bridge | 像素级 | 连续/线性目标 |
| **T3 移动目标** | Aircraft, Vessel | 实例级 | 离散，中等大小 |
| **T4 固定设施** | Crater, Tank, Pier, Dock, Shelter, Tower | 实例级 | 离散，小目标 |

**关键决策**: 像素级标注中**移除 T3/T4 类别**（映射为 ignore），两边各自安好。

### 像素级改动

| 位置 | 当前 | 改为 |
|------|------|------|
| `label_decode_map` | 33-42, 43-68 → 各类 | → 255 (ignore) |
| `num_pixel_targets` | 15 | **7** (Farmland~Bridge) |
| `target_names` | 15 类 | 7 类 |
| `pixel_class_weights` | 15 项 | 7 项 |
| 模型输出 | 15 类 target | 7 类 target |

**保留的 7 类**: Farmland, Runway, Taxiway, Apron, Bridge, Highway, Building

**移除的 8 类**: Shelter, Tower, Pier, Dock, Tank, Aircraft, Vessel, Crater → 归实例级

---

## 二、实例级统计

基于 `instance_dataset.json`（61,029 真实样本扫描）:

| 任务 | 类别 | 实例数 | 状态分布 | 优先级 |
|------|------|--------|----------|--------|
| **T3** | Aircraft | 1,021 | reduced 351, added 320, no-change 274, damaged 49, replaced 27 | P0 |
| **T3** | Vessel | 1,207 | reduced 547, added 291, no-change 210, replaced 86, damaged 73 | P0 |
| **T4** | Crater | 6,024 | added 6024（全部） | P0 |
| **T4** | Tank | 2,212 | no-change 2046, damaged 113, added 29, reduced 24 | P0 |
| **T4** | Pier | 239 | no-change 213, extended 12, damaged 8, reduced 6 | P1 |
| **T4** | Dock | 139 | no-change 123, reduced 10, damaged 4, added 2 | P1 |
| **T4** | Shelter | 107 | added 52, no-change 50, reduced 5 | P1 |
| **T4** | Tower | 13 | - | P2（过拟合尝试） |

**总计**: 10,949 实例，3,103 个样本

**关键观察**:
- Crater 全部是 added（弹坑是新增的）
- Tank 92% 是 no-change（状态不平衡）
- Aircraft/Vessel 状态分布较均匀
- Tower 仅 13 个，尝试过拟合

**数据集位置**: `D:/CD/0617final_256_v2/instance_dataset.json` (4MB)
**构建脚本**: `script/build_instance_dataset.py`

---

## 三、实例分类器设计

### 整体架构

```
输入: 256×256 图像 + 实例框
  │
  ├─ 裁剪实例区域（GT框 + 随机抖动 + 上下文填充）
  │
  ├─ 空间分支: 轻量 CNN
  ├─ 频域分支: DWT → HF/LF 特征
  │
  └─ 融合 → 分类头
       ├─ Target 分类（8类稀有类 + 背景）
       └─ State 分类（7类）
```

### 训练策略

```
训练数据:
  - 256×256 原图
  - instance_dataset.json 中的 GT 框
  - 裁剪时: GT框 + 随机抖动(±20%) + 上下文填充(30%)

推理流程:
  - 像素级预测 → 候选提取 → 裁剪 → 实例分类器
  - VotingRefinement 精炼
```

### 网络结构

```python
class InstanceClassifier(nn.Module):
    def __init__(self, num_classes=9, num_states=7):
        # 空间分支
        self.spatial_branch = nn.Sequential(
            ConvBlock(3, 32), ConvBlock(32, 64),
            ConvBlock(64, 128), ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d(1),
        )
        # 频域分支
        self.freq_branch = FrequencyBranch(in_channels=3, out_dim=64)
        # 分类头
        self.target_head = nn.Linear(256 + 64, num_classes)
        self.state_head = nn.Linear(256 + 64, num_states)
```

### 实例候选提取

```python
def extract_instance_candidates(target_logits, rare_classes, min_pixels=10):
    """
    从像素级预测中提取稀有类的实例候选。
    
    Args:
        target_logits: [B, 15, H/4, W/4]
        rare_classes: [14, 11, 13, 12, 9, 10, 7]  # Crater, Tank, Vessel, Aircraft, Pier, Dock, Shelter
        min_pixels: 最小像素数阈值
    
    Returns:
        candidates: list of (batch_idx, class_id, bbox, confidence)
    """
    target_pred = target_logits.argmax(dim=1)  # [B, H/4, W/4]
    candidates = []
    
    for b in range(target_pred.shape[0]):
        for cls_id in rare_classes:
            mask = (target_pred[b] == cls_id)
            if mask.sum() < min_pixels:
                continue
            
            # 连通域分析
            labeled, n = connected_components(mask.cpu().numpy())
            for i in range(1, n + 1):
                inst_mask = (labeled == i)
                if inst_mask.sum() < min_pixels:
                    continue
                
                # 计算 bbox
                rows = np.where(inst_mask.any(axis=1))[0]
                cols = np.where(inst_mask.any(axis=0))[0]
                y1, y2 = rows[0], rows[-1]
                x1, x2 = cols[0], cols[-1]
                
                candidates.append({
                    'batch_idx': b,
                    'class_id': cls_id,
                    'bbox': [x1, y1, x2, y2],
                    'pixels': inst_mask.sum(),
                })
    
    return candidates
```

### 实例分类头

```python
class RareClassInstanceHead(nn.Module):
    """
    稀有类实例分类头。
    输入: RoI Align 提取的实例特征
    输出: 实例级 target + state
    """
    def __init__(self, feat_dim=256, num_rare_classes=7, num_states=7, roi_size=7):
        super().__init__()
        self.roi_size = roi_size
        
        # 共享特征提取
        self.shared_fc = nn.Sequential(
            nn.Linear(feat_dim * roi_size * roi_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        
        # Target 分类（稀有类 + 背景）
        self.target_classifier = nn.Linear(256, num_rare_classes + 1)
        
        # State 分类
        self.state_classifier = nn.Linear(256, num_states)
    
    def forward(self, roi_features):
        """
        roi_features: [N, C, roi_size, roi_size]
        """
        x = roi_features.flatten(1)
        x = self.shared_fc(x)
        
        target_logits = self.target_classifier(x)
        state_logits = self.state_classifier(x)
        
        return target_logits, state_logits
```

### 像素-实例融合

```python
def fuse_pixel_instance(pixel_target_logits, instance_results, candidates):
    """
    融合像素级和实例级预测。
    
    策略: 稀有类用实例级结果覆盖像素级预测。
    """
    fused = pixel_target_logits.clone()
    
    for cand in candidates:
        if cand['confidence'] > 0.5:  # 实例级置信度阈值
            x1, y1, x2, y2 = cand['bbox']
            # 用实例级预测覆盖该区域
            fused[cand['batch_idx'], :, y1:y2+1, x1:x2+1] = \
                instance_to_pixel_logits(cand['instance_pred'])
    
    return fused
```

---

## 四、训练策略

### 核心原则：完全解耦

像素级和实例级**各自独立训练**，互不依赖。推理时组合。

```
像素级训练:
  输入: 256×256 图像
  标注: 只含 7 类（Farmland~Bridge），T3/T4 类别映射为 ignore
  模型: DINOv3 + PixelStateHead
  产出: 7 类 target + 7 类 state

实例级训练:
  输入: 256×256 图像 + 实例框
  标注: instance_dataset.json（T3/T4 的 8 类）
  模型: 独立的 InstanceClassifier（轻量CNN + 频域分支）
  产出: 8 类 target + 7 类 state

推理组合:
  像素级 → 密集图（T1/T2）
  实例级 → 实例检测结果（T3/T4）
  合并 → 最终制图
```

### 实例级训练细节

```
数据准备:
  - 从 256×256 原图裁剪实例区域
  - GT框 + 随机抖动(±20%) + 上下文填充(30%)
  - 统一 resize 到 64×64

增强:
  - 随机旋转 (0-360°)
  - 随机缩放 (0.5-2.0)
  - 水平/垂直翻转
  - 颜色抖动

采样:
  - 类平衡采样，每个 batch 确保各类都有
  - Tower 单独处理（13实例，重度增强）

损失:
  - Target: Focal CE + Dice
  - State: CE（加权，处理状态不平衡）
```

### 类平衡采样

```python
class InstanceBalancedSampler:
    """确保每个 batch 包含各稀有类的实例"""
    def __init__(self, instance_dataset, rare_classes, instances_per_class=4):
        self.dataset = instance_dataset
        self.rare_classes = rare_classes
        self.ipc = instances_per_class
        
        # 按类别索引实例
        self.class_indices = defaultdict(list)
        for idx, inst in enumerate(instance_dataset):
            self.class_indices[inst['class_id']].append(idx)
    
    def __iter__(self):
        batch = []
        for cls_id in self.rare_classes:
            indices = self.class_indices[cls_id]
            if len(indices) >= self.ipc:
                chosen = np.random.choice(indices, self.ipc, replace=False)
            else:
                chosen = np.random.choice(indices, self.ipc, replace=True)
            batch.extend(chosen)
        np.random.shuffle(batch)
        return iter(batch)
```

### Tower 过拟合策略

```python
# 13 个实例 → 重度增强 → 过拟合
tower_augmentation = Compose([
    RandomRotate(360),
    RandomScale(0.5, 2.0),
    HorizontalFlip(),
    VerticalFlip(),
    ColorJitter(0.3, 0.3, 0.3, 0.1),
])

# 小网络 + 强正则
tower_head = nn.Sequential(
    nn.Linear(feat_dim, 64),
    nn.ReLU(),
    nn.Dropout(0.5),  # 强 dropout
    nn.Linear(64, 2),  # tower / not-tower
)

# 或者用原型学习
tower_prototype = mean(features_of_13_towers)  # 预计算
# 推理: cosine_distance(feat, tower_prototype) < threshold
```

---

## 五、评估协议

### 实例级指标

| 指标 | 定义 | 用途 |
|------|------|------|
| Instance IoU | 预测实例与GT实例的IoU | 主指标 |
| Count Accuracy | 实例计数准确率 | 可清点性 |
| Per-class AP | 每类的 Average Precision | 综合 |
| State Accuracy | 实例级状态分类准确率 | 状态判定 |

### 像素-实例协同指标

| 指标 | 定义 |
|------|------|
| Rare-class IoU 提升 | 实例级融合后 vs 纯像素级 |
| 实例计数误差 | |预测数 - GT数| |
| 边界 F1 | 实例边界的精确率/召回率 |

### 四任务分解评估

| 任务 | 评估类别 | 指标 |
|------|----------|------|
| T1 建筑损毁 | Building | IoU, State IoU |
| T2 地表变化 | Farmland, Runway, Taxiway, Apron, Highway, Bridge | mIoU, SeK |
| T3 移动目标 | Aircraft, Vessel | Instance IoU, Count Acc |
| T4 固定设施 | Crater, Tank, Pier, Dock, Shelter | Instance IoU, Count Acc |

---

## 六、对比实验策略

### 与现有方法的对比难点

| 方法类型 | 输出格式 | 和我们的差异 |
|----------|----------|-------------|
| 二值 CD（BIT等） | changed / unchanged | 我们有 target+state |
| 语义 CD（ChangeFormer等） | from-class → to-class | 我们是 target×state |
| 损毁评估（xBD baseline） | 建筑+严重度 | 我们覆盖更多目标 |

### 分层对比设计

| 对比层 | 我们的方法 | 基线方法 | 数据集 | 指标 |
|--------|-----------|----------|--------|------|
| **二值 CD** | state≠no-change | BIT, ChangeFormer | 0617final | F1, IoU, SeK |
| **语义 CD** | target+state → from-to | IFN, SNUNet | SECOND | mIoU, SeK |
| **建筑损毁** | T1 子集 | xBD baseline | 0617final | F1_overall |
| **完整框架** | 全部输出 | 无直接基线 | 0617final | 我们的指标 |
| **四任务分解** | T1-T4 分别报告 | 无直接基线 | 0617final | 各任务指标 |

### P1 论文对比表设计

```
表1: 0617final 主表（我们的指标）
  - 完整模型
  - 像素-only 消融
  - 实例-only 消融
  - 去 GNN / 去频域消融

表2: 二值 CD 对比（映射到共同空间）
  - BIT, ChangeFormer, SNUNet
  - 我们的模型（二值化后）
  - 指标: F1, IoU, SeK

表3: SECOND 数据集对比（公开基准）
  - 直接用 SECOND 的 from-to 格式
  - 对比已发表的 SCD 方法
  - 最公平的对比

表4: 四任务分解评估（我们的特色）
  - T1/T2/T3/T4 分别报告
  - 展示各任务难度差异
  - 这是别人没有的分析
```

### 各任务对比可行性

| 任务 | 有直接基线吗 | 对比方式 |
|------|-------------|----------|
| T1 建筑损毁 | ✅ xBD baseline | 直接对比 F1 |
| T2 地表变化 | ✅ SCD 方法 | 映射到 from-to 对比 |
| T3 移动目标 | ❌ 没有 | 只报告我们的结果 |
| T4 固定设施 | ❌ 没有 | 只报告我们的结果 |

**T3/T4 是我们的贡献点**——别人没做过，不需要对比，只需展示我们能做到。

---

## 七、Tower 处理策略

### 问题

Tower 在像素级完全学不到（IoU=0，仅 13 个实例），但仍在消耗模型容量和梯度。

### 方案：像素级彻底移除，实例级单独处理

**像素级改动**:

| 位置 | 当前 | 改为 |
|------|------|------|
| `label_decode_map` | 38-42 → 8 (Tower) | 38-42 → 255 (ignore) |
| `num_pixel_targets` | 15 | 14 |
| `target_names` | 含 Tower | 移除 Tower |
| `pixel_class_weights` | 15 项 | 14 项（移除 Tower） |
| `state_ignore_targets` | [0, 8] | [0] |
| 模型输出 | 15 类 target | 14 类 target |

**实例级保留**:
- Tower 实例保留在 `instance_dataset.json`
- 单独训练 Tower 检测头（过拟合策略）
- 13 个实例 + 重度增强（旋转/缩放/翻转/颜色抖动）
- 小网络 + 强正则（dropout 0.5）
- 或原型学习（13 个实例的特征均值作为原型）

**效果**:
- 像素级: 容量集中在 14 个可学的类
- 实例级: 尝试用过拟合救回 Tower

---

## 八、实施计划

| 阶段 | 内容 | 时间 | 产出 |
|------|------|------|------|
| 1 | 实例数据集构建（从标注提取 T3/T4 实例） | 2天 | instance_dataset.json |
| 2 | 四任务 baseline 评估（当前模型） | 1天 | T1-T4 各任务指标 |
| 3 | 实例候选提取 + RoI Align 实现 | 2天 | 模块代码 |
| 4 | 实例分类头训练 | 2天 | 训练好的 head |
| 5 | 像素-实例融合 | 1天 | 融合推理代码 |
| 6 | 四任务对比评估 | 2天 | 对比表格 |
| 7 | 消融实验 | 2天 | 消融表格 |

**总计: ~2周**

### 优先级

```
P0 (立即):
  ├── 实例数据集构建
  └── 四任务 baseline 评估

P1 (短期):
  ├── 实例候选提取 + RoI Align
  └── 实例分类头训练

P2 (中期):
  ├── 像素-实例融合
  └── 对比 + 消融
```

---

## 七、与 P1 论文的关系

本方案直接支撑 P1 的核心创新点 N3：

| N3 要求 | 本方案提供 |
|---------|-----------|
| 像素变化场先验 | 已有（像素级分支） |
| 实例精炼 | ✅ 专门分类头替代简单投票 |
| 像素-实例协同 | ✅ 稀有类用实例级覆盖 |
| 实例可清点性 | ✅ 实例级输出 + 计数指标 |

**论文中的表述**:
> We propose a pixel-instance collaborative framework where pixel-level prediction provides dense change fields for large/continuous targets, while instance-level classification handles small/discrete targets that are challenging for pixel-level methods due to downsampling. A dedicated rare-class instance head with class-balanced training improves instance countability and state interpretation for long-tailed categories.

---

## 八、风险与缓解

| 风险 | 缓解 |
|------|------|
| 实例候选提取不准 | 用 GT 连通域训练，推理时用预测 |
| Tower 过拟合失败 | 接受放弃，报告为 limitation |
| 融合策略简单 | 先用覆盖，后续可试注意力融合 |
| 训练不稳定 | 两阶段训练，先冻结再微调 |
