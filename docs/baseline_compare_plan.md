# CNN / Transformer / Mamba 四任务对比实验计划

> 日期: 2026-09-16
> 状态: 计划定稿，待实施
> 依据: WD-CD BDD 协议 + 0617final 四任务划分 + 用户确认「每任务单独训练」
> 相关: `evaluation/bdd_protocol.py`, `script/eval_four_tasks.py`, `docs/instance_level_plan.md`

---

## 一、WDCD / BDD 协议结论

本地代码与用户确认共同指向：

| 问题 | 结论 | 证据 |
|------|------|------|
| 是否一次训全类别？ | **否** | 用户确认；T1 建筑损毁与 BDA 协议一致 |
| 是否每任务单独训？ | **是** | 建筑损毁用 BDD F_loc / F_clf；四任务各自独立指标 |
| 主指标 | F_overall = 0.3·F_loc + 0.7·F_clf | `script/eval_bdd_protocol.py` |
| 状态折叠 | ND = {no-change, added, extended}；D = {damaged, reduced} | 同上 |
| 对照数字（仅 T1 建筑） | DeepSiameseNet 72.46%；CD-Lamba 79.53% | 同上 |

**训练设定对齐 WDCD**：每个任务独立训练一个模型，不共享跨任务输出头；其它任务类别像素一律 `ignore (255)`。

---

## 二、四任务定义（与现有评估一致）

| 任务 | 名称 | 类别 ID | 类别名 | 输出范式 |
|------|------|---------|--------|----------|
| **T1** | 建筑损毁 | 6 | Building | 双头: loc(2) + clf(ND/D) |
| **T2** | 地表覆盖变化 | 0,1,2,3,4,5 | Farmland, Runway, Taxiway, Apron, Bridge, Highway | 双头: multi-loc + state |
| **T3** | 移动目标变化 | 12,13 | Aircraft, Vessel | 双头: multi-loc + state |
| **T4** | 设施变化 | 7,9,10,11,14 | Shelter, Pier, Dock, Tank, Crater | 双头: multi-loc + state |

说明：
- Tower(8) 已在当前配置中 ignore，不进 T4。
- 「其它任务类别」在训练标签中写成 255。
- 状态共 7 类；评估时再按 BDD 折叠成 ND/D，或报全状态 SeK / mIoU。

### 标签重映射（每任务）

```
原始 target ∈ 本任务类 → 重映射为任务内 0..K-1
原始 target ∉ 本任务类 → 255 (ignore)
原始 target = Background(15) → Background(最后一个类，或 0 用于二值 loc)
state 原样保留；非本任务像素上 state 不计 loss
```

T1 额外折叠（与 BDD 对齐）：

```
loc: 0=bg, 1=Building
clf: 0=ND (no-change/added/extended), 1=D (damaged/reduced)
      其它 state (replaced/none) 在 clf 上 ignore
```

---

## 三、模型矩阵（每类 2 个，共 6 个）

| 家族 | 模型 | 角色 | 官方仓库 | 本地状态 |
|------|------|------|----------|----------|
| **CNN** | **SNUNet-CD** | 语义/多类变化经典 CNN | justchenhao/SNUNet-CD | 待克隆 |
| **CNN** | **FCCDN** | 面向建筑损毁的 CNN（托 T1） | NWPU-INGP/FCCDN 或等价实现 | 待克隆；备选 STANet |
| **Transformer** | **ChangeFormer** | 通用 transformer SCD | wenhwu/ChangeFormer | 待克隆 |
| **Transformer** | **BIT** | Bitemporal transformer CD | justchenhao/BIT_CD | 待克隆 |
| **Mamba** | **ChangeMamba (ST/MambaSCD)** | 时空 SSM 语义变化 | ChenHongruixuan/ChangeMamba | **已有** `F:\mambacd\ChangeMamba`，含 0617 + BDA |
| **Mamba** | **MambaBDA (同仓)** | ChangeMamba 建筑损毁双头 | 同上 `tasks/bda.py` | **已有**，最贴 T1 |

备选（若 FCCDN 适配成本过高）：
- CNN2 → **STANet** 或 **DSIFN**
- Mamba2 若需独立仓 → **CDMamba** / **MambaBCD**（ChangeMamba 已覆盖则不必再克隆）

### 任务 × 模型适配策略

| 模型原生形态 | 用到的任务 | 适配方式 |
|--------------|------------|----------|
| MambaBDA（loc 2 类 + damage 5 类） | **T1 首选** | 0617 建筑子集 + ND/D 折叠（可先 5 级再折叠） |
| MambaSCD（多类语义变化） | T2 / T3 / T4 | 标签重映射为任务内类 + ignore |
| SNUNet / ChangeFormer / BIT（多类/二值） | T2–T4 主用；T1 可改双头 | 统一加 task adapter：输出 `K_task+1` 类或 loc+clf |
| FCCDN（建筑损毁） | **T1 专用** | 同 BDA 双头协议 |

**公平性约束**（每组对比尽量相同）：
- 同一 `0617final_256_v2` train/val/test 划分
- 同一输入尺寸 256×256（部分实现原生 256，FCCDN/Mamba 可按官方 crop；统一改为 256）
- 同一训练预算：**50 epoch 或 ~40k iter**，取先到者
- 同一优化器族：AdamW，lr=1e-4，weight_decay=1e-4 或按官方默认（记录在案）
- AMP + batch 8–16（视显存）
- 类不平衡：任务内类权可选；**主表不加 Copy-Paste/GNN/频域**，只保留官方损失 + 可选 Lovasz（与 ChangeMamba 一致时统一打开）

---

## 四、统一评估协议

对每个「模型 × 任务」在 **test split** 上导出 `pred_target / pred_state`，再跑同一套评估。

### 指标

| 指标 | 适用 | 来源 |
|------|------|------|
| **F_loc / F_clf / F_overall** | 全部任务（T1 主表） | `bdd_protocol.evaluate_bdd_protocol` |
| **SeK** | 变化/未变化分离 | `bdd_protocol.compute_sek` |
| **Target mIoU** | T2–T4 | 任务内类别 IoU 平均 |
| **State mIoU** | T2–T4 | 非 ignore 像素上 7 类状态 |
| **F1_changed** | 全部 | state ∉ {no-change, none} |

主文建议两张表：

```
表 A — T1 建筑损毁（BDD，对标 WD-CD）
  Method | Family | F_loc | F_clf | F_overall
  DeepSiameseNet (ref) | - | - | 72.46
  CD-Lamba (ref)       | - | - | 79.53
  SNUNet-CD            | CNN | ...
  FCCDN                | CNN | ...
  BIT                  | TF  | ...
  ChangeFormer         | TF  | ...
  MambaBDA             | Mamba | ...
  ChangeMamba-SCD*     | Mamba | ...
  Ours (HICD v7)       | Ours | ...

表 B — T2–T4（多任务分解）
  Method | T2 mIoU/SeK | T3 mIoU/SeK | T4 mIoU/SeK
```

*ChangeMamba-SCD 也可映射到 T1 作「单模型跨任务」对照，但主设定仍是分任务训练。

---

## 五、数据与输出目录

| 项 | 路径 |
|----|------|
| 数据 | `D:/CD/0617final_256_v2`（Windows）/ `/mnt/d/CD/0617final_256_v2`（WSL） |
| 配置 | 复用 `HICD_v7/datasets/configs/0617final.yaml` 的 decode_map / 划分 |
| 新建工作区 | `F:\mambacd\baselines_compare\` |
| 模型源码 | `baselines_compare/repos/<model>/` |
| 任务标签缓存 | `baselines_compare/labels/task_{T1..T4}/` |
| 权重 | `baselines_compare/runs/<model>__<task>/` |
| 评估 JSON | `baselines_compare/results/<model>__<task>.json` |
| 汇总表 | `baselines_compare/results/summary.md` |

ChangeMamba 可直接用现有 `F:\mambacd\ChangeMamba`，数据 list 已有 `data_lists/0617/`。

---

## 六、实施阶段（不急，按批推进）

| 阶段 | 内容 | 预估 | 产出 |
|------|------|------|------|
| **P0** | 任务标签重映射脚本 + 与现有 `eval_four_tasks` 数字自检 | 1 天 | `make_task_labels.py`，抽样可视化 |
| **P1** | **Mamba 双模型先跑通**（本地已有 ChangeMamba：BDA→T1，SCD→T2–T4） | 2–3 天 | 4–5 个 run，摸清显存/时长 |
| **P2** | 克隆并适配 **ChangeFormer + BIT**（Transformer） | 3–4 天 | T1–T4 各 1 run |
| **P3** | 克隆并适配 **SNUNet + FCCDN**（CNN） | 3–4 天 | T1–T4 各 1 run |
| **P4** | 全量重跑 + 汇总 + 论文表 | 2–3 天 | summary + 表 A/B |

总矩阵（完整）：**6 模型 × 4 任务 = 24 次训练**。  
P1 摸底可只跑：Mamba×4 + 每家族 1 个代表 × T1 = **约 6–8 次**，确认协议后再铺满。

### 建议启动顺序

```
1. make_task_labels（ignore 策略，对齐 instance_level_plan）
2. ChangeMamba-BDA → T1（最快拿到 F_overall，对标 72/79）
3. ChangeMamba-SCD → T2, T3, T4
4. ChangeFormer / BIT
5. SNUNet / FCCDN
6. HICD v7 分任务版（对照：全类联合 vs 分任务，可作消融）
```

HICD 本体建议加一行：
- **Ours-joint**：现有全 15 类联合训练，在 test 上按任务 mask 评估（已有 `eval_four_tasks.py`）
- **Ours-task-specific**（可选）：与基线同设定分任务训练，证明框架优势不是「看全图类别」

---

## 七、环境注意

| 项 | 说明 |
|----|------|
| 训练环境 | 历史脚本走 WSL Ubuntu + conda `mamba`（`/root/miniconda/envs/mamba/bin/python`） |
| ChangeMamba 依赖 | Linux + CUDA；需编译 `kernels/selective_scan` |
| 预训练 | ChangeMamba: `pretrained_weight/vssm_tiny_224_0229.pth`（本地已有） |
| CNN/Transformer | ImageNet 预训练 ResNet/Swin/等，按各仓 README |
| 当前会话 | 本机 `nvidia-smi`/WSL 暂不可用时，训练需在可用 GPU 的 WSL/服务器上执行；计划与代码可先在 Windows 侧写好 |

---

## 八、风险

| 风险 | 缓解 |
|------|------|
| 官方实现与 0617 标签格式不齐 | 统一中间格式：`pre.png / post.png / label.png` 或现有 TIF + list；先写 adapter |
| Mamba kernel 编译失败 | 用 ChangeMamba 已验证环境；或退回 `mamba-ssm` 官方包 |
| 24 次训练周期过长 | P1 只跑 6–8 次摸底；主表优先 T1 + 每家族一模型 |
| 类极不平衡（Tower 已 ignore；Crater/Tank 尾部） | 任务内类权 + Lovasz（与官方一致时）；主表不混用 Copy-Paste |
| 二值/语义输出不可比 | 全部经同一 `bdd_protocol` + 四任务 mask；不比原始论文数字（除 T1 文献值） |

---

## 九、待你确认后的执行项

1. 认可本计划后，从 **P0 标签脚本 + P1 ChangeMamba** 开工。
2. FCCDN 若找不到维护良好的官方仓，自动换成 **STANet**（会在结果表备注）。
3. 是否需要 **Ours-task-specific（HICD 分任务）** 进主表：默认做，因为要和「WDCD 单独训练」设定对齐。

---

## 十、快速对照：仓库现状

| 仓库 | 路径 | 与本计划关系 |
|------|------|--------------|
| HICD_v7 | `F:\mambacd\home\HICD_v7` | 主方法 + 统一评估 |
| ChangeMamba | `F:\mambacd\ChangeMamba` | Mamba 基线，已适配 0617（`tasks/change_0617.py`）+ BDA |
| changedetection (v5) | `F:\mambacd\changedetection` | 旧版 HICD/Mamba 骨架，可参考数据加载 |
