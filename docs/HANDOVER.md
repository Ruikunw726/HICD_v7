# HICD 项目交接文档

**交接日期**: 2026-09-28
**交接给**: Codex

---

## 一、项目概述

**HICD** (Hierarchical Instance Change Detection) — 战争损毁语义变化检测

- **代码库**: `F:\mambacd\home\HICD_v7` (WSL: `/mnt/f/mambacd/home/HICD_v7`)
- **数据集**: `D:\CD\0617final_256_v2`
- **目标**: 方法论文，超过SOTA (CD-Lamba 79.53%)

---

## 二、当前状态 (截至2026-09-28)

### 训练状态

| 实验 | 状态 | F_overall | 说明 |
|------|------|-----------|------|
| v8_bg_suppress | ✅ 完成 | 78.49% | 有合成数据 |
| v10_real_only | ✅ 完成 | 77.48% | 纯真实数据 |
| **v11_no_t4** | ⏸️ 中断 | - | WSL挂了，待恢复 |

### 关键指标 (v10, 纯真实数据)

| 指标 | v10 | SOTA (CD-Lamba) | 差距 |
|------|-----|----------------|------|
| T1 F_overall | 0.7851 | 79.53% | -1% |
| T2 F1 | 0.5447 | - | |
| T3 F1 | 0.6732 | - | |
| T4 F1 | 0.4843 | - | |
| Global SeK | 0.1053 | - | |

---

## 三、核心决策

### 1. 任务分工
- **T1建筑 + T2地表 + T3移动** → 像素级
- **T4设施** (Bridge/Shelter/Pier/Dock/Tank/Crater) → 实例级

### 2. 数据策略
- **像素级**: 纯真实数据 (61,029样本)
- **实例级**: 可用合成+过采样
- **合成数据**: 只对T4有帮助，对T1-T3有拖累

### 3. 稀有类过采样 — 已放弃
- 50x/20x过采样都导致主类下降
- 结论：不是方法问题，是数据问题

### 4. T2/T4定义已更新 (2026-09-28)
- **T2**: Farmland, Runway, Taxiway, Apron, Highway (移除Bridge)
- **T4**: Bridge, Shelter, Pier, Dock, Tank, Crater (加入Bridge)
- 原因：Bridge F1=0拉低T2，应归T4

---

## 四、关键文件

### 核心代码
```
F:\mambacd\home\HICD_v7\
├── models/
│   ├── HICD_v7.py                 主模型
│   ├── instance_classifier.py     实例分类器 (待训练)
│   ├── backbone/dinov3.py         DINOv3 backbone
│   ├── losses/v7_loss.py          Loss (含BackgroundSuppression)
│   └── modules/label_structure.py GNN标签空间
├── datasets/
│   ├── dataset_v7.py              DataLoader
│   └── configs/0617final.yaml     配置 (已改: T4→ignore)
├── evaluation/
│   ├── bdd_protocol.py            BDD协议 (已改: T2/T4定义)
│   └── icd_eval_v6.py             评估器
├── script/
│   ├── train_full_v7.py           训练脚本
│   └── train_instance.py          实例级训练 (待用)
├── launch_v11.sh                  v11启动脚本
└── docs/
    ├── experiment_log.md          实验记录
    └── instance_level_plan.md     实例级方案
```

### 数据
- **主数据集**: `D:\CD\0617final_256_v2`
  - train: 61,029 / val: 13,418 / test: 13,418 (纯真实)
  - 备份: `names.txt.bak_synth` (含合成), `names.txt.bak_rare` (含注入)
- **实例数据**: `D:\CD\0617final_256_v2\instance_dataset.json`
  - T3: Aircraft(1021), Vessel(1207)
  - T4: Bridge(133), Shelter(107), Pier(239), Dock(139), Tank(2212), Crater(6024)

---

## 五、待办事项

### 优先级P0
1. **恢复WSL，继续v11训练** — 移除T4类，看T1-T3能否提升
2. **对比FC-EF** — 多任务vs多任务（公平对比）

### 优先级P1
3. **实例级训练** — T4稀有类，可用合成+过采样
4. **方法改进超过SOTA** — 当前78.49% vs 79.53%

### 优先级P2
5. GitHub推送 (代码已commit `2834e23`)
6. 论文写作

---

## 六、重要发现

### 1. IoU vs BDD协议
- IoU忽略假阳性，BDD协议更准确
- Building IoU=0.996 但 F_loc Precision只有0.227

### 2. 合成数据影响
- 合成弹坑对T4有帮助，对T1-T3有拖累
- 文件名: `synth_crater_*` (10,746样本)

### 3. SeK算法
- 训练: 背景参与loss (BackgroundSuppressionLoss)
- 评估: 排除背景，只在目标像素计算
- 已修复 `eta_hat=1.0` 时返回nan的bug

### 4. 标签编码
- 原始标签值: Bridge=18-22, Shelter=33-37, Dock=48-52
- 解码后class_id: Bridge=4, Shelter=7, Dock=10
- **扫描时用原始值，不是解码后ID**

---

## 七、运行命令

### WSL启动
```bash
wsl
cd /mnt/f/mambacd/home/HICD_v7
export PYTHONPATH=$(pwd)
```

### 训练
```bash
bash launch_v11.sh           # v11: 移除T4
screen -r hicd_v11           # 查看训练
tail -f checkpoints/v11_no_t4/train.log
```

### 评估
```bash
/root/miniconda/envs/mamba/bin/python script/eval_v9_quick.py
```

---

## 八、注意事项

1. **WSL环境**: Python在 `/root/miniconda/envs/mamba/bin/python`
2. **训练用screen**: 断开不影响训练
3. **日志优化**: tqdm输出到stderr，`2>/dev/null`不进log
4. **数据划分**: 61,029/13,418/13,418 是最终版本，勿混淆历史数字

---

## 九、论文定位

**策略**: 强调多任务端到端优势
- 统一模型 vs 多个单任务模型
- T3/T4大幅提升
- 对比: 让FC-EF也做多任务训练（公平对比）

**核心卖点**:
1. 多任务统一框架
2. GNN标签空间
3. BackgroundSuppressionLoss
4. T3/T4性能优势

---

**祝顺利！** 🚀

---

## 十、baselines_compare (另一会话工作)

**路径**: F:\mambacd\baselines_compare

### 当前状态 (2026-09-18)

| 模型 | T1 | T2 | T3 | T4 |
|------|----|----|----|-----|
| FC-EF (单任务) | 0.89 | 0.65 | 0.45 | 0.16 |
| FC-Siam-diff | 0.42 (35/50) | 排队 | - | - |

### 重要决策
1. **已决定改多任务训练** (MULTITASK_PLAN.md)
   - 原因: 分任务对基线不公平（类空间更小）
   - 计划: FC-EF-multi, FC-Siam-multi
   - 状态: 等Siam跑完后实施

2. **评估协议**: 与HICD一致 (dd_protocol.py)

### 关键文件
- PROTOCOL.md — 对比协议说明
- MULTITASK_PLAN.md — 多任务训练计划
- 	rain_fcef.py — FC-EF训练
- eval_fcef.py — 评估脚本
- 	ask_labels.py — 四任务标签定义

### 待办
1. 等FC-Siam T1-T4跑完
2. 实现	rain_multitask_fcef.py
3. 训多任务FC-EF/Siam
4. 与HICD对比

### 模型仓库
- FC-EF/Siam: epos/FCEF_daudt
- SNUNet, STANet, BIT, ChangeFormer, ChangeMamba等已克隆

---

## 十一、导师沟通与论文进展

### 论文定位 (已确认)
- **发方法论文**，不是数据集论文（WD-CD已被Li Dongjing发过）
- **策略**: 强调多任务端到端优势
- **目标**: 超过SOTA (CD-Lamba 79.53%)

### 关键文献

| 论文 | 作用 | 文件 |
|------|------|------|
| **WD-CD** (Li Dongjing) | 数据集来源，BDD协议 | 用户提供PDF |
| **CD-Lamba** | SOTA对比 (79.53%) | - |
| **FC-EF** (Daudt 2018) | 古早baseline | baselines_compare |

### 已读/精读文档
- docs/DESIGN_ISPRS2024_精读.md — ISPRS论文精读
- docs/PhD_thesis_framework.md — 博士论文框架
- docs/paper_order_tasks.md — 论文任务顺序

### 周报/日报
- **日报**: E:\docs\changedetection\日报\daily_report_YYYYMMDD.md
- **周报**: 每周四写
- **最近日报**: daily_report_20260918.md

---

## 十二、工作习惯

1. **每天写日报** — 记录进展、问题、明日计划
2. **周四写周报**
3. **实验记录** — docs/experiment_log.md
4. **训练用screen** — 断开不影响
5. **评估用BDD协议** — F_loc/F_clf/F_overall + SeK

---

## 十三、下一步建议

### 短期 (本周)
1. WSL恢复后继续v11训练
2. 看T1-T3能否超过FC-EF单任务
3. 准备多任务对比实验

### 中期 (2周内)
4. 实例级训练 (T4稀有类)
5. 方法改进 (F_clf是瓶颈)
6. 论文框架细化

### 长期
7. 超过SOTA
8. 完成论文写作
