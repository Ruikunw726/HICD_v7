# HICD 实验记录
## v10_real_only (纯真实数据) — 已完成
**日期**: 2026-09-23 | **状态**: 已完成 (50轮) | **结论**: 纯真实数据表现优秀，T1-T3接近v8，T4意外学到，SeK大幅改善

### BDD协议 (T1建筑)
| 指标 | v10 | v8 | CD-Lamba |
|------|-----|-----|----------|
| F_overall | **0.7748** | 0.7849 | 79.53% |
| F_loc | 0.8269 | 0.8266 | 83.04% |
| F_clf | 0.7525 | 0.7506 | 78.03% |
| Global SeK | **0.1053** | 0.01 | - |

### 四任务F1 (best)
| 任务 | v10 | v8 | 评价 |
|------|-----|-----|------|
| T1 建筑 | **0.7851** | 0.7849 | 持平 |
| T2 地表 | **0.5447** | 0.5318 | 略好 |
| T3 移动 | **0.6732** | 0.6697 | 持平 |
| T4 设施 | **0.4843** | 0.5119 | 接近 |

### 各类别F1
Runway 0.97, Farmland 0.91, Taxiway 0.89, Tank 0.85, Apron 0.84, Building 0.82, Vessel 0.80, Aircraft 0.70, Highway 0.69, Shelter 0.69, Dock 0.66, Pier 0.64, Crater 0.06, Bridge 0.00

### SeK修复效果
Global SeK: 0.01 → **0.11** | T1 SeK: 0.05 → **0.21**

### 数据配置
纯真实61,029样本 | 无合成 | 无注入 |

---

## 数据集说明

**主数据集**: `/mnt/d/CD/0617final_256_v2`

### 数据来源

| 来源 | 文件名格式 | 数量 | 说明 |
|------|-----------|------|------|
| 真实数据 | `Al-Fallujah_*`, `Kyiv_*`, `Kherson_*`, etc. | 61,029 | 战争损毁真实标注 |
| **合成弹坑** | `synth_crater_{id}_{variant}` | 10,746 | v7_synth_dense实验生成 |

**注意**: 合成数据文件名以`synth_crater_`开头，与真实数据命名不同。

---

## v9_rare_oversample (50x boost) — 已完成

**日期**: 2026-09-20
**状态**: 已完成 (50轮)
**结论**: 稀有类学到了，但主类被挤压，整体F_overall略低于v8

### 最终结果 (Epoch 40-50)

**四任务F1对比**:
| 任务 | v9 (50x) | v8 (best) | 差距 | 评价 |
|------|----------|-----------|------|------|
| **T1 建筑** | **0.7604** | 0.7849 | -2.5% | 轻微下降 |
| **T2 地表** | **0.4829** | 0.5318 | -4.9% | 略有下降 |
| **T3 移动** | **0.5735** | 0.6697 | -9.6% | ↓ 明显下降 |
| **T4 设施** | **0.5053** | 0.5119 | -0.7% | 基本持平 |

**BDD协议 (T1建筑)**:
| 指标 | v9 (50x) | v8 (best) | 差距 |
|------|----------|-----------|------|
| **F_overall** | **0.7604** | 0.7849 | -2.5% |
| F_loc | 0.7927 | ~0.83 | -4% |
| F_clf | 0.7224 | ~0.75 | -3% |

### 数据配置 (83,900训练样本)

| 来源 | 样本数 | 占比 | 文件名格式 |
|------|--------|------|-----------|
| 真实数据 | ~61,029 | 72.7% | `Al-Fallujah_*`, `Kyiv_*`, etc. |
| **合成弹坑** | **10,746** | **12.8%** | `synth_crater_*` |
| 稀有类注入 | 7,200 | 8.6% | 重复注入真实样本 |
| 历史注入 | 4,925 | 5.9% | 重复注入真实样本 |

**合成弹坑数据**:
- 文件名: `synth_crater_00000_0`, `synth_crater_00001_0`, etc.
- 数量: 10,746样本
- 来源: v7_synth_dense实验
- 用途: 增强Crater类样本

### 稀有类注入详情

| 类别 | 源样本 | 过采样 | 注入后 | 标签原始值 |
|------|--------|--------|--------|-----------|
| Bridge | 35 | 50x | 1,855 | 18-22 |
| Shelter | 46 | 50x | 4,990 | 33-37 |
| Dock | 66 | 50x | 6,442 | 48-52 |

### 各类别F1对比

| 类别 | v9 (ep40) | v8 | 变化 | 评价 |
|------|-----------|-----|------|------|
| **Bridge** | 0.5721 | 0.0000 | +57% | ✓ 学到了 |
| **Shelter** | 0.8661 | 0.0000 | +87% | ✓ 学到了 |
| **Dock** | 0.9248 | 0.0044 | +92% | ✓ 学到了 |
| Building | 0.7798 | 0.8266 | -5% | ↓ 轻微下降 |
| Aircraft | 0.5089 | 0.7926 | -28% | ✗ 下降严重 |
| Crater | 0.0526 | 0.0738 | -3% | ✗ Precision仅0.03 |
| Highway | 0.5622 | 0.6731 | -16% | ↓ 下降 |
| Tower | 0.0000 | 0.0000 | - | 无样本 |

### 发现

1. **稀有类过采样有效**：Bridge/Shelter/Dock从0学到57-92%
2. **50x boost太激进**：主类被挤压
   - T1建筑: -2.5% (可接受)
   - T2地表: -4.9% (略降)
   - **T3移动: -9.6%** (Aircraft -28%, 最严重)
   - T4设施: -0.7% (稀有类在这里，反而持平)
3. **T3移动下降最严重**：Aircraft被稀有类干扰
4. **T4设施反而没降**：稀有类(Bridge/Shelter/Dock)在这里，学到东西弥补了
5. **Crater问题**：Precision只有0.03（合成数据假阳性多）
6. **收敛在ep30左右**：F_overall=0.7435，后续波动

### 教训

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 主类下降 | 50x过采样挤压 | 下次用20x |
| Aircraft低 | 被稀有类干扰 | 降低boost |
| Crater假阳性 | 合成数据质量问题 | 待查 |
| log太长 | 进度条每步输出 | miniters=100 |

---

## 实验结论 (2026-09-23)

### 稀有类过采样 — 失败，已放弃

| 实验 | 策略 | F_overall (ep10) | vs v8 |
|------|------|------------------|-------|
| **v8** | 无过采样 | **0.7424** | baseline |
| v9 | 50x过采样 | 0.6449 | -9.8% ✗ |
| v9_v2 | 20x过采样 | 0.6991 | -4.3% ✗ |

**结论**：
1. **稀有类过采样是负增益** — 让Bridge/Shelter/Dock学到东西，但主类下降更多
2. **得不偿失** — T1/T2/T3/T4全部下降
3. **决定**：放弃过采样，回到v8配置

**稀有类问题的正确解法**（待探索）：
- 实例级方法单独处理T4设施类
- 或者接受稀有类F1=0（样本确实太少）

---

## v9_rare_oversample_v2 (20x boost) — 已停止

**日期**: 2026-09-22
**状态**: 已停止 (ep12，结论明确)
**改动**: boost从50x降到20x

### 配置
- 稀有类过采样: 20x (原来是50x)
- 合成弹坑: 保留 10,746样本 (`synth_crater_*`)
- 真实数据: 61,029样本
- 稀有类注入: 7,200样本 (20x过采样)

### 预期
- T1建筑: 保持78-80%
- T2地表: 保持50-55%
- T3移动: 恢复到65-70% (重点)
- T4设施: 45-55% (稀有类学一点)
- F_overall: 76-78%

---

## v8_bg_suppress (已完成)

**日期**: 2026-09-18
**结果**: F_overall=78.49% (best)

### 实验目的
解决背景假阳性问题(IoU=0.996但F_loc Precision只有0.227)

### 关键改进
- BackgroundSuppressionLoss (w=0.5)
- Background从ignore改为target=15
- BDD协议评估

### 数据配置
- 训练样本: 71,775
- 真实数据: 61,029 (85%)
- 合成弹坑: 10,746 (15%) — `synth_crater_*`
- 无稀有类注入

### 四任务结果

| 任务 | F1 | SeK |
|------|-----|-----|
| T1 建筑 | 0.7849 | 0.0494 |
| T2 地表 | 0.5318 | -0.0002 |
| T3 移动 | 0.6697 | 0.0571 |
| T4 设施 | 0.5119 | 0.0263 |

### 对标WD-CD

| 方法 | F_loc | F_clf | F_overall |
|------|-------|-------|-----------|
| DeepSiameseNet | 72.99 | 72.23 | 72.46 |
| CD-Lamba (SOTA) | 83.04 | 78.03 | 79.53 |
| **v8 (ours)** | 80.28 | 71.65 | **78.49** |

---

## v7_synth_dense (已完成)

**日期**: 2026-09-17
**数据**: 71,775样本 = 61,029真实 + 10,746合成弹坑(`synth_crater_*`)

### 实验目的
高密度合成弹坑增强

### 结果
- 弹坑IoU: 0.050 → 0.407
- 像素比例: 0.046% → 0.890%

---

## 关键发现

### 1. IoU vs BDD协议
- IoU只在GT范围内计算，忽略假阳性
- Building IoU=0.996但F_loc Precision只有0.227
- 根因: `ignore_index=255`导致背景不参与loss

### 2. 稀有类学习问题
- Bridge/Shelter/Dock原始样本极少
- 50x过采样能学到，但会挤压主类
- **建议**: 20x过采样

### 3. 合成数据影响
- 合成弹坑文件名: `synth_crater_*`
- 数量: 10,746样本 (12-15%)
- Crater类Precision低(0.03)，可能是合成数据质量问题

### 4. 标签编码
- 原始标签值: Bridge=18-22, Shelter=33-37, Dock=48-52
- 解码后class_id: Bridge=4, Shelter=7, Dock=10
- 扫描时需要用原始值

### 5. SeK计算
- 训练: 背景参与loss (BackgroundSuppressionLoss)
- 评估: 排除背景，只在目标像素上计算SeK
- 否则背景像素稀释SeK

---

## v11_no_t4 — 2026-10-08 原实验续训（历史记录，已停止）

**授权与范围**：用户在总览会话明确授权继续 v11；本轮只恢复原实验，不更改 Bridge/T3/T4 decode、loss、模型、评估或采样定义。原日志的 F_overall 与 BDD 汇总差异仍登记为待核查，非本次修正。

**恢复来源**：`checkpoints/v11_no_t4/checkpoint_epoch_32.pth`，保存 epoch=31（零基）；SHA256 `466d352445a31882b9bb2f428d717f18cd5fda7f7ba9f66fe089cf29275145c2`。CPU 全量读取及模型/optimizer张量有限检查通过；恢复既有动态 CE weight buffer 后 strict=True，模型键完全兼容。恢复349项 AdamW state、5组学习率、scheduler last_epoch=32、AMP scale=32768/growth_tracker=782，日志实际确认 `RESUME_OK epoch=33/50`。best_loss 保留历史最低1.9521795181734931，best按最低验证损失选择。

**协议与数据**：直接复用 checkpoint 参数及与历史日志完全相等的 `datasets/configs/0617final.yaml`（SHA256 `829184d8624902b4b666c405eb7be53c665f8c3c63dd98f938438c1f149799ac`）；batch32、img256、AMP、workers4、state_aware、目标50轮。实际 train61029/val13418；test名单13418，train加权6741，与历史日志一致。Bridge仍为18–22→4，不将本次结果重新命名为严格“全部T4移除”。

**持久进程**：Ubuntu-26.04，screen `571355.hicd_v11_resume`（Detached），训练主PID `571365`；2026-10-08 11:05:07 +08:00 启动。首次screen留下死socket且未初始化worker，仅清理自己的死socket后重试；现入口等待worker状态并检查Detached，避免仅凭socket宣称成功。

**早期健康检查**：2026-10-08 11:07:49 +08:00，第33/50轮已推进至少41/1907 batch，运行平均loss3.0771；连续batch推进，未见Traceback/OOM/nan/inf；主进程存活、CUDA正常，显存约12.4GiB。这不是完整轮次验证结果，也不是test结果，剩余训练尚未完成。

**代码和保护**：新增 `launch_v11_resume.sh`、`script/resume_v11.py`，旧 `launch_v11.sh` 转接安全入口；拒绝重复训练/screen，flock保护，缺失resume即失败，保留stderr。bash语法、Python编译、缺失检查点及实际重复启动保护验证通过；本轮未修改 train_full_v7.py、模型/损失、YAML、数据或评估实现。内层HEAD `2834e232512b4bb6e2df2a7716cb0ae9d41e5304`；既有工作区修改/删除保留并快照。

**审计/日志**：`checkpoints/v11_no_t4/resume_20261008_110218/` 保存恢复前CSV/log/best/epoch32、配置/名单/hash、有效源码、Git diff/status、manifest、command.txt、python.pid、status.txt、early_health.json；scene_state_info.json也快照并记录hash。主train.log和CSV追加，CSV待完整轮次验证后新增；独立 `train_resume.log` 保存本次stdout/stderr。最终launcher初始化保护另存 `launcher_with_initialization_guard.sh`，与启动时source快照区分。

**精确命令（工作目录HICD_v7）**：
```bash
/root/miniconda/envs/mamba/bin/python -u script/resume_v11.py --mode run --checkpoint /mnt/f/mambacd/home/HICD_v7/checkpoints/v11_no_t4/checkpoint_epoch_32.pth --audit-dir /mnt/f/mambacd/home/HICD_v7/checkpoints/v11_no_t4/resume_20261008_110218
```

**限制**：旧checkpoint未保存RNG、sampler、自适应权重/running_iou等；原phase2 reset保留，不制造或宣称严格连续性。当前hash提供本次可追溯状态，不能证明历史所有图像文件和代码未变。尚无第33轮完整验证或新test成绩；未创建自动监控任务。

**回复前复核**：2026-10-08 11:09:30 +08:00，已连续推进至第33轮至少96/1907 batch，运行平均loss2.9541，无非有限loss/上述致命异常；证据 `resume_20261008_110218/early_health_followup.json`。

---

## v12_strict_joint_20261008 — 严格目标+原始状态的新训练（16:23停止，保留缺陷版本记录）

**最新授权**：用户明确说明v11已终止并要求按最新标准重新训练，本次替代v11续训。启动前未发现旧v11或其他训练进程/screen，旧文件保持原样；没有全局WSL重启或结束无关进程。

**初始化/预算**：全新任务模型、AdamW、CosineAnnealingLR、AMP，epoch0、无resume/旧任务checkpoint，仅正常加载原DINOv3 pretrained backbone。50轮、batch32、img256、LR1e-4、workers4、AMP、state_aware、freeze_backbone_epochs=0，其余参数与v11审计manifest相同（仅实验名与resume不同）。代码/数据配置/损失/采样未顺便修改。

**评估**：04公共协议 `hicd.strict_joint.v1`，raw state_logits；GT nearest scale0.25，256输入对应64×64评估网格。每目标主P/R/F1需target+具体state全对，replaced保留，错目标/错状态/pred_none/有效背景前景FP计错，GT ignore统一，无GT且无预测项为N/A（JSON null、CSV空）。报告全前景/任务 strict joint micro+macro、各目标TP/FP/FN/support、定位和有效GT目标区域状态诊断、105条target-state与联合混淆计数。原IoU反馈mask只独立服务自适应loss及warmup，非strict分数。

**范围**：有效GT目标为Farmland/Runway/Taxiway/Apron/Bridge/Highway/Building/Aircraft/Vessel；T4当前仅Bridge GT，Shelter/Pier/Dock/Tank/Crater和Tower仍ignore，未恢复其数据或改split。那些通道在其他有效区域上的FP仍计错。不能称完整T4六类、实例检测或已解决split独立性问题。

**验证**：shell/Python语法通过；27项core+adapter测试全过无skip；TrainingEvaluator的IoU反馈与旧helper小数组对照通过。独立smoke执行2个合成训练batch+1真实val batch（2样本），实际模型/target准备/strict计数/打印/反馈/JSON路径成功，train_updates=2、val_updates=1；仅管线检查，不作模型成绩。正式进程重新初始化，日志明确 `FRESH_START epoch=1/50 task_checkpoint=None optimizer_states=0 scheduler_last_epoch=0`。

**运行**：2026-10-08 16:04:47 +08:00启动，screen `452.hicd_v12_strict_joint_20261008`，训练主PID459。16:06:57 +08:00已到epoch1至少44/1907 batch，运行平均loss10.6561，连续推进且有限，未见Traceback/OOM/nan/inf；显存约12123MiB。仅早期健康验收，未完成首轮全量验证/50轮训练，无新test数字。

**产物/选择**：新 `script/train_strict_v7.py` + `launch_v12_strict.sh`；strict CSV无旧F_overall/T1_f1列，每轮train+validation完整JSON写metrics/epoch_NNN.json。best按验证全前景strict_joint_f1 micro最大；macro并报。latest每轮原子保存、best更新、每10轮编号；checkpoint保存协议/config/args/run_identity、optimizer/scheduler/AMP、RNG及自适应loss状态。未保存worker RNG/批次中途状态；本次新训练入口拒绝resume，后续恢复须验证协议与身份。

**证据**：实验目录 `checkpoints/v12_strict_joint_20261008/` 含manifest、snapshot、Git HEAD/status/diff、resources、27项验收说明、preflight_smoke.json/log、early_health.json、command.txt/python.pid/status.txt/train.log；run_identity=`994b373cb5cbac1fd3a88febbcd613b1b7b722cc3d0a6a3a821c903088ee3bce`。精确配置/名单/scene metadata与有效源码快照/hash均保存，预训练权重hash引用原文件。启动锁/运行锁/重复任务拒绝验证通过，错误输出保留；未创建自动监控。

**回复前复核**：2026-10-08 16:09:31 +08:00，epoch1至少124/1907 batch、运行平均loss9.8819，有限且无上述异常；补充 `early_health_followup.json`。

---

## v12_strict_joint_t123_trainfix_20261008 — 审计修复后、仅T1–T3（2026-10-09已完成50轮）

**授权/来源**：用户要求复核05记忆的10/08代码清查并按需修复，随后明确“现在不训练T4，只训练1-3”。原v12命中A01频域未优化、A02几何错位，不能以健康loss作为正确学习证明。16:23:52精确结束旧主PID459及其4个DataLoader子进程，16:23:53旧status返回143/tee0；旧run停在epoch1/704batch，无完整checkpoint，日志/快照全部保留，不用作修复后成绩。

**修复**：频域/融合参数入optimizer，全部488参数张量各注册一次（暂冻结者预注册）；target/state/双图一次共享几何、同步crop+nearest；paired copy-paste只占背景15、保留源状态/双时相内容而不随机造状态；软约束用log概率先验，避免负logit乘小数反升；补累积尾窗和零解冻边界，标签读取/shape失败显式报错、未映射raw忽略。A03仅大图crop触发，A06当前默认参数不触发，历史发生比例/收益未测。实例A07–A11及基线A12–A16本轮未修复/训练。

**范围**：独立配置 `datasets/configs/0617final_t123_trainfix.yaml`，T1 Building；T2 Farmland/Runway/Taxiway/Apron/Highway；T3 Aircraft/Vessel。全部T4（包括Bridge）及Tower的target/state raw映射255，目标权重0；原配置/名单/split/图像保留。16/7输出通道保留用于协议兼容和误报计数，T4预测错误仍计FP，没有T4正样本/状态监督或实例训练。Farmland state loss原ignore选择不变。

**初始化/预算**：50轮、batch32、img256、LR1e-4、AMP、workers4、state-aware；DINO原预训练backbone，任务/optimizer/scheduler/AMP全新，epoch0且无resume。严格协议仍hicd.strict_joint.v1、raw state_logits、nearest scale0.25→64网格；best仍全前景strict_joint micro最大，macro同报。模型修复/类别范围变化与旧run严格隔离。

**验证**：40项测试（27strict+13完整性）无失败；100种子坐标对齐、右下裁剪、Farmland不被copy-paste覆盖、source replaced状态保留、实际decoder负logit、频域更新/zero_grad、尾窗数值、零比例和T123-only配置均覆盖；语法及diff空白检查通过。GPU smoke改为4个真实增强train+2个真实val：目标状态配对source-decodable，T4 train/val GT support0/0，30个freq_branch/freq_fusion参数收到有限非零梯度且更新。仅通路验收，正式训练重新初始化，不将smoke分数当成绩。

**启动/健康**：2026-10-08 16:25:28 +08:00，screen `1709.hicd_v12_strict_joint_t123_trainfix_20261008`，主PID1716。16:27:54 +08:00已到epoch1至少103/1907batch，运行平均loss9.4367，有限且连续推进，无Traceback/OOM/nan/inf；约12371MiB显存。未完成首轮完整验证/50轮，无新test或修复收益结论。

**证据**：`checkpoints/v12_strict_joint_t123_trainfix_20261008/`含manifest、snapshot、审计原记录/源码/config/三名单/metadata副本、acceptance_tests、preflight_smoke、early_health、command/status/PID；run_identity=`fd1ae754c259bb7bd3b550c4cdc1b3ac04b9932fbe2a723c7a52459d206a59b0`；配置SHA256=`c4beb1e57eaf3db996ee2ba0e37eb1341b6ffcf346a3d68d973bc3b8338e0ad9`。未来恢复必须匹配新6组/全参数优化器与配置身份，不能直接用旧v11/v12 optimizer。旧快照保留；无自动监控、推送或改动公共协议。

**回复前复核**：2026-10-08 16:33:29 +08:00，epoch1至少453/1907batch、运行平均loss7.9583，有限且无上述异常，PID1716/screen存活；early_health_followup.json。loss变化不是受控修复增益或完整验证成绩。

### 2026-10-09 完成验收与验证结果

11:32:30 +08:00正常结束，python_exit0/tee_exit0，50个epoch JSON与50行CSV逐行global micro匹配、50轮loss有限；每轮val13418、T4 GT support0，无训练进程/screen。仅验证集，未做test推理。ZIP/pickle指令元数据核验best_model显示第7轮，latest/epoch50显示50，协议/identity/score与JSON一致；没有执行pickle或全tensor加载。

| 指标 | 官方micro-best：epoch7 | epoch50 |
|---|---:|---:|
| strict_joint_f1 micro | 0.7190867 | 0.6261505 |
| joint类别macro-F1 | 0.4024206 | 0.5204461 |
| T1 micro | 0.7067725 | 0.8001874 |
| T2 micro | 0.7277867 | 0.5225332 |
| T3 micro | 0.5933199 | 0.7069399 |

第50轮目标joint F1：Farmland0.4492、Runway0.9690、Taxiway0.9022、Apron0.8595、Highway0.6921、Building0.8002、Aircraft0.6371、Vessel0.7269。完整P/R/TP/FP/FN/support和定位/状态诊断见epoch_050.json及results_analysis_20261009.json。

主要发现：Farm占前景GT50.1213%、GT仅no-change，loc0.9396但GT目标区state微F1/准确率0.4826；joint从epoch7的0.7160降至0.4492，其余七类都提高。配置state_ignore_targets=[0]导致Farm直接状态CE/Dice不监督，而严格主指标要求其state正确；图validity辅助监督仍在，此不一致是漂移原因候选，尚未做因果消融。T1/公路的低召回、Aircraft replaced F1约0.22%（715px）、Vessel replaced约37.25%（8460px）是下一层瓶颈。不能仅根据micro下降称整体退化：macro、其余类别和loss均有进展。

单轮峰值：T1 epoch36=0.8045918，T3 epoch32=0.7213168，macro epoch32=0.5233032；非同一模型best，且32/36没有独立编号权重，不能拼接或替换官方best7。当前config/loss/strict核心hash与启动manifest相符；trainer启动snapshot hash匹配，当前trainer后来有报告格式改动（diff仅见数值/CSV/打印格式化），本run以snapshot与原始metrics为准。下一步先对齐Farm直接监督，再做T1–T3受控改进/按预设best7补test；不改变本run既定选择规则，不直接与旧BDD/外部0.7851比较。本轮无重训、配置/代码修改或长test。

### 2026-10-09 用户明确农田固定no-change：独立回顾复算

用户说明本项目Farm无论损毁均按不变，GT确实仅state0。撤回上段优先恢复Farm直接状态监督的建议：保留Farm state_ignore，预测target=Farm时固定state0，使输出符合项目语义；其他类仍要求原始具体状态正确。规则只依据预测target，错目标、背景FP与ignore不被消除。

新增evaluation/farmland_semantics.py的opt-in规则/适配器，预测语义版本hicd.predicted_farmland_nochange.v1；strict匹配公式仍v1，原core/raw结果不改。4项测试通过，含随机3000像素直接解码vs计数复算（各状态诊断也一致）、错目标/背景FP、GT冲突拒绝及模型batch/Torch接口。

第50轮val复算：micro0.6261505→0.8786285，macro0.5204461→0.6185618，Farm joint0.4492441→0.9395602，T2 micro0.5225332→0.9272358；T1=0.8001874、T3=0.7069399和其他目标joint均不变。50轮回顾峰值46轮0.8789693（无保存权重），已保存权重里50轮最优；原raw-best第7轮在该解码下micro0.8218201，best_model文件仍7轮，不覆盖。

产物farmland_nochange_review_20261009/summary.json及50个decoded val报告，带源hash/identity/预测策略标识。仅对已保存完整计数的精确复算，无新模型推理/test/重训；提升来自预测语义规则，不是新学习或受控方法增益。T1–T3/T4-ignore范围、原数据/CSV/JSON/checkpoint不变。后续test采用新规则时显式声明并以50轮现有权重为可用候选，保留原raw结果对照。

### 2026-10-09 第50轮完整test与随机可视化（已完成）

用户明确授权epoch50跑test及随机GT/Pred图。加载checkpoint_epoch50（epoch49零基）strict=True、身份/config/source验证通过；恢复动态CE buffer，float32 raw state、nearest64、预测Farm固定state0。已有BiSRNet基线训练保留并继续运行；测试batch8/workers2，无训练或参数更新。推理投票后处理跳过前验证raw logits/argmax等价，非refined计分。

15:27:11 +08:00，完整13418张test处理完成，耗时约801.54秒。全前景joint P0.9517749/R0.8289321/F10.8861163，macro0.6827043；T1 micro0.8074207、T2 0.9293423、T3 0.6953959，T4 N/A且GT/pred支持0。完整计数TP12667376/FP641838/FN2614184/GT15281560，raw未解码micro0.6255719另存。不是val复算替代test。

目标joint F1：Farmland0.9438241、Runway0.9324503、Taxiway0.8900159、Apron0.8541477、Highway0.7610734、Building0.8074207、Aircraft0.7044548、Vessel0.6924259。稀有状态：Aircraft replaced0.2055974（1452px支持、R0.13154），Vessel replaced0.3831930（17537px支持、R0.26470）。

原split同名重叠仍在：train/test197、val/test341；没有重划或丢样本。另报无train同名13221张micro0.8859314、T3 0.6453032；有train同名197张T3 0.8784704。不能仅因整体micro差异小就说稀有类泛化不受影响，也未核实场景级独立。

seed20261009推理前固定均匀抽6张，不挑好图，indices3972/5557/6343/8283/8665/12239；6个NPZ源数组、单图PNG/SVG/PDF与总览完成视觉QA。含sample2全黑背景（foreground F1 N/A）和sample4 Taxiway→Runway（joint F1=0），没有换掉失败/空样本。指标grid64，显示nearest；目标/状态各用一致颜色，GT-ignore斜线，错误分类展示；图片无额外contrast/crop，不声称输出256细节。

新目录test_epoch50_farmland_nochange_20261009/包含完整metrics、各目标/target-state CSV、分组诊断、checkpoint hash/manifest、采样与source图数组、status和figures。计数/分组加和、F1公式、13418完整数量与6图对应已验证。脚本eval_epoch50_strict_test.py及render_test_comparisons.py可复用；不覆盖原训练/val/checkpoint，不修改公共协议/数据或停止基线。

### 2026-10-09 撤回为T4过拟合保留的跨split共享（数据恢复完成，尚未新训练）

用户要求目前只训练T1–T3，样本泄漏改回，并明确没有改名样本。采用原names.txt.bak真实名单：train备份64481去3452合成前缀后61029，与当前train一致；val/test原备份各13077，撤回后来各加入341条。没有内容去重/随机洗牌/城市级重划；最初CPU内容审计在用户澄清后取消，未采用其未完成结果。

发布独立view `D:/CD/0617final_256_v2_disjoint_20261009`，train61029/val13077/test13077；same-name及去末尾_NN_NN后的母样本tile交集全部0，解析pre路径交集0。train names字节不变，val/test与原备份字节一致；原root names/图片和旧run保留，不让已有任务未来静默切到新test。paths catalog固定实际pre/post/label，schema/name/catalog哈希和范围验证失败即报错、不向兄弟目录fallback。

identity=`9f66845372df4c8730a543ac2899575b8ba40eff31b676355b183991e09db05b`；manifest在新view/split_manifest.json，恢复前名单/撤回清单和验收在datasets/split_audits/original_restored_20261009/。51项CPU测试、语法/bash-n/diff检查通过；new loader实载三split全部数量，9个真实样本shape/T123/Farm0校验通过。城市前缀共享74沿用原设计，未宣称城市/地理footprint独立或完成内容hash审计。

新配置0617final_t123_disjoint_20261009.yaml与launch_v12_strict.sh默认采用新版，T4仍ignore、Farm固定no-change；训练计分显式应用既定Farm规则，旧配置raw结果不变；新manifest/checkpoint记split与semantic identity，测试数量/config从快照读取。没有启动新训练，旧88.61%test及其权重/图为旧划分历史，添加superseded标识不改原指标。正式比较须重新训练并让基线同步新data/catalog/reference；旧BiSRNet状态已由其他工作记为用户停止/取消自动test，本会话未停止它。

**最终推广**：确认当前无训练/测试Python、用户已停旧基线后，原D:/CD/0617final_256_v2的val/test名单也原子恢复为13077，train61029字节不变；旧名单在audit/before，新rootnames与原目录现在逐字节一致。canonical_transition记录此更新，优先于最初为保护运行基线而仅发布新view的状态。原图片、权重、旧指标均保留；旧reference/hash不可擅改，未来基线需切换新数据身份。无新GPU任务。

---

## v12_strict_joint_t123_disjoint_e100_20261009 — 恢复原划分后的100轮（已完成，2026-10-11）

**授权**：用户明确要求数据修复后开始新训练，预算100轮。数据61029/13077/13077、三split同名及母样本tiles互斥；版本identity9f66845372df4c8730a543ac2899575b8ba40eff31b676355b183991e09db05b，T1–T3-only、T4/Tower ignore、Farm预测固定不变，不再用旧共享val/test。

**初始化与预算**：DINO原预训练backbone，任务权重/AdamW/scheduler/AMP全新，epoch0且无resume。batch32/img256/LR1e-4/AMP/workers4/state-aware/freeze0不改；cosine T_max100。primary best_model按val joint micro最大，另保留best_macro按joint类别macro，latest每轮/每10轮编号；checkpoint保存数据/预测策略身份和macro最佳score/epoch。后期可能改善召回/稀有状态但不保证加轮次收益，旧46–50解码micro已近平台；划分及LR周期也变，不能当旧50预算的单因素对照。

**启动/验收**：51项CPU检查、语法通过；独立GPU4 real train+2 real val通路通过，freq30参数有finite非零梯度并更新。val两张仅background导致前景n/a，未伪造分数。正式进程重新初始化，FRESH_START epoch1/100/optimizer0/scheduler0，日志declared固定Farm+新split+T_max100。2026-10-09 17:11:55 +08:00启动，screen157237.hicd_v12_strict_joint_t123_disjoint_e100_20261009，主PID157244；17:16:40到epoch1至少480/1907batch、平均loss7.6386，finite且无OOM/Traceback/nan/inf，screen/进程存活。仅初始健康，不是完整epoch成绩。

**产物**：run_identity d1584ee10d43b2df6e2b2e6922535cb2451a9f44941bedd8bcb23dba4ea93d75；新目录含manifest、含catalog/split manifest的snapshot/hash、preflight_smoke、status/command/PID、early_health及training_receipt。旧权重/结果不覆盖，当前后台目标100轮；没有新test或定时自动任务。

回复前第二次health已推进超过1000/1907batch，相比480继续增长，loss有限无上述异常，early_health_followup.json保存精确时间/批次/loss；尚未完成首轮验证。

### 2026-10-10 可信度核查（进行中43轮验证快照）

已完成43个epoch artifact、每轮val13077；训练仍运行，未改参数/数据。初读42轮时micro-best epoch40=.8810079，最终43轮完成后更新best43=.8831862（macro=.5678663，T1=.8086347/T2=.9297337/T3=.6436613）；macro-best仍epoch39=.6025228。live names/catalog hash与启动一致、same-name/pre源路径跨split0；独立按联合矩阵重新计算43轮micro/macro精确一致，像素计数闭合，正确背景不加前景TP，GT不用于预测解码。

必须区分train与val：43轮Aircraft train=.8493258、val=.6673715；Vessel train=.8884361、val=.6310640。Farm占val前景50.8218%，no-change占84.7535%，因此micro高不代表全部小state高；损毁/替换状态仍弱。val既选best又反馈adaptive loss权重，是开发集；100轮的新test尚未跑，恢复test也是此前已评估test的子集，非全新外部盲测。原城市前缀共享74，未做地理独立性/多seed/同协议基线和因果消融。

本次评估认为定义范围内指标与已知漏样修复可信，泛化/全状态能力结论需上述后续证据。credibility_review_20261010.json保存快照与边界；无停训/改代码或新GPU评估。

### 2026-10-11 100轮已完成：结果核查与归档

用户要求查看当前结果并做好实验记录。status记录2026-10-11 06:13:10 +08:00正常结束（python_exit=0、tee_exit=0）；总历时37小时1分15秒。100轮train/val完整；本轮无HICD训练/测试进程，未启动新GPU任务。

| 验证严格联合指标 | 第50轮 | micro-best第90轮 | macro-best第56轮 | 最终第100轮 |
|---|---:|---:|---:|---:|
| micro-F1 | 87.9534% | 89.5640% | 88.7501% | 89.4294% |
| macro-F1 | 59.2885% | 57.8201% | 62.2611% | 57.8068% |
| T1 micro | 79.7360% | 82.5335% | 81.8210% | 82.1282% |
| T2 micro | 92.9319% | 93.9570% | 93.1016% | 93.9459% |
| T3 micro | 64.6415% | 64.7825% | 63.8118% | 64.4563% |

主best_model.pth读回completed_epoch90（原存epoch89零基），micro P95.4282%/R84.3789%；独立best_macro.pth读回56（原epoch55），没有更改原主选模规则。latest与checkpoint_epoch100均100；编号50也正确。5份checkpoint run/split/协议/预测语义身份与原manifest一致、scheduler T_max100及轮次正确。

同一run前50轮micro峰值49轮88.5657%→全程90轮89.5640%，提高0.9984个百分点；macro峰值39轮60.2523%→56轮62.2611%，提高2.0088个百分点。50→100主要改善T1/T2及整体召回，T3未明显提高；最后10轮micro89.3132%–89.5267%近平台。本比较是100轮cosine计划内前后观察，不是独立50/100预算的受控重复实验。

稀有replaced：Aircraft GT694px，第56/90/100轮F1=22.38/12.32/7.22%；Vessel GT5957px=21.44/4.83/4.39%。100轮train T3 micro90.93%，val64.46%，泛化与小状态仍是瓶颈；没有用高micro宣称全面损毁识别或SOTA。

macro解释：GT支持组合固定26；既定规则将GT无支持但有误预测的组合也以0纳入，第56/90/100轮参与类别数27/29/29。仅GT支持组合的诊断macro64.6557/64.4916/64.4768%另列，不能替换官方62.2611/57.8201/57.8068%或据此改best。后续基线统一类别、N/A与背景政策。

独立复算200份train/val联合矩阵的micro/macro与原JSON精确一致，计数闭合、农田非0状态行/列为0、loss有限、指定致命日志标记无命中。train drop_last=True，每轮计分61024而非名单61029；val每轮完整13077。live数据manifest登记文件SHA仍匹配启动。未做重新推理/大checkpoint内容SHA或全参数扫描。

完整完成记录：`docs/experiments/v12_t123_disjoint_e100_completion_20261011.md`。机器证据、100轮全精度history及50/56/90/100轮全部联合类别CSV：`checkpoints/v12_strict_joint_t123_disjoint_e100_20261009/completion_review_20261011/`。配置/身份/原日志和权重保留，公共/其他专题记忆未改。

本实验新test仍未评估；val用于选模及adaptive loss反馈，属于开发集。后续建议在固定test13077上按既定micro-best90及预先声明的补充macro-best56各评一次并另记，不按test选checkpoint。旧划分test88.61/68.27是历史，不能与当前val直接比较；城市/内容独立性、单次运行、基线/消融等限制仍在。


### 2026-10-11 第90轮完整测试集已完成

用户明确指定第90轮权重跑test并记录。加载当前run的best_model.pth（内部epoch89），strict=True，run/split/config/源码输入hash和实际轮次校验通过；checkpoint SHA256 `dd8b2300c7f2f6208bea3af19be7bb8869d15f3e379a5c05b7b0695ced5be98e`。采用固定第90轮，不根据test另选模型；没有评估第56轮或更新训练参数。

2026-10-11T16:41:46.834415+08:00完整13,077张test完成，耗时544.65s；float32/raw state/nearest64、batch8/workers2、T1–T3-only及Farm预测固定不变。train/test与val/test同名交集均0，未删样本。本run新test状态由未评估更新为已完成，之前记录保留为历史。

test strict joint P=95.5402%、R=85.0587%、micro-F1=89.9953%、macro-F1=62.2192%；T1=83.1818%/T2=93.8324%/T3=62.5648%。TP=12809293/FP=597933/FN=2250058；T4返回N/A，其GT忽略，不作能力结论。相同90轮val micro89.5640%/macro57.8201%，val是选模/adaptive反馈开发集，测试另列。

目标及所有状态P/R/F1/support见完整报告与CSV；replaced：Aircraft F1=33.1322%（GT1431px）、Vessel=20.1583%（GT15034px）。官方test macro有效组合28、GT支持组合25，继续把无GT但有误预测的组合按0纳入，不更换公式。仅GT支持macro诊断=69.6855%，不能替换正式分数。

独立矩阵复算raw/decoded micro/macro/P/R和全部目标/状态计数精确一致，样本/像素闭合、组别全量一致、Farm合并与身份核验通过；证据verification.json。产物`checkpoints/v12_strict_joint_t123_disjoint_e100_20261009/test_epoch90_20261011/`、目录旁log和`docs/experiments/v12_t123_epoch90_test_20261011.md`。examples0，未额外渲染图片；原权重/训练数据/指标不覆盖，公共及其他专题记忆未改。

边界：恢复test是此前已看过旧test的子集，非新外部盲测；城市/内容独立性、单次运行、同协议基线/消融仍待证据。不能用旧划分50轮test与本结果归因预算或模块增益。下一步以固定协议结果作为本run test记录，保留各小状态诊断。

### 2026-10-12 方法重构提案：完成诊断，未改模型

用户要求方法重构并明确选择先诊断/方案、不改模型。对正在训练的新版Bi-SRNet做只读val复核：已完成45轮；44轮val micro88.5944%、macro43.2430%、T1/T2/T3 81.3342/93.5127/59.0247%。HICD44为88.4808/56.3079/80.6106/93.3330/62.3969%；HICD micro-best90为89.5640/57.8201/82.5335/93.9570/64.7825%。不存在44轮所有指标/类别全面领先，基线test尚未完成；不混用val/test。

数据身份/strict/Farm语义、8项共同数据hash及比较报告完整GT行边际相同，基线45轮矩阵独立复算通过。基线44 Recall89.7755%明显高于HICD44 83.2472%，Precision87.4440%低于94.4166%；前景→背景1180783 vs2243839，背景→前景1568654 vs522896。HICD的主要行为短板是保守漏检；小目标具体状态另有困难，原因尚待控制实验。

只读代码发现与原run SHA相符的两个P0接入点：DINO输入未做官方要求的权重匹配mean/std；中间特征在block前抽取，末block/final norm未进入返回特征。另列TFM早混合/同patch金字塔/复杂loss控制/旧Farm图合法表等候选。CPU检查GNN56/90的有支持非Farm组合均通过门限，未把动态门控当成已证实退化原因。

完整提案和全精度证据在`docs/experiments/method_reconstruction_20261012/`。建议先核查与分别控制归一化/after-block接入，随后固定简单监督对照，再逐项验证pre/post方向保留、RGB细节、目标条件状态和图/频域。多seed、固定选模/test和合法输出约束的公平控制写入方案。所有新结构/阈值/选模口径都是候选，未启动或宣称有效。原模型/配置/权重/指标、共享数据及运行基线保留；本轮仅本线记录和分析产物。

### 2026-10-12 DINO接入已校正并验收，安排基线后接续100轮

最新用户授权覆盖接入修正及基线完成后的从头重训。独立hicd.dinov3_input_features.v1 adapter在DINO内部按LVD均值/标准差归一化，保留DWT原RGB；使用真实timm Eva after-block2/5/8/11 normalized特征接口，末block和final norm进入输出/梯度。旧DINO/core/配置/数据/评估保持，新训练通过显式新builder/入口，不热切换旧实验或运行基线。

本地权重与固定revision的timm官方LVD对照实际162个模型张量全部相等，额外24个历史bias键不被当前模型使用；原文件未替换，来源/hash见weight_provenance。四层接口独立hook、输入不变、Normalize、输出shape、末层/finalnorm非零有限梯度通过；真实CPU2train一次更新+2val strict8192px闭合，val仅bg，非性能结果。54项CPU回归、语法通过；GPU预检等资源空闲再做。

待启动run：v13_dino_corrected_t123_disjoint_e100_20261012，原恢复划分、T1–T3-only/T4ignore/Farm不变、原结构其余部分/损失/采样/batch32/LR1e-4/AdamW/cosine100/AMP/冻结/micro主best+macro次best不改，不resume。配置新增的唯一解析差异为dino版本。本次两处接入一并校正，是合并接入对照，不声称能独立分离每处修正的因果收益。

本聊天每小时接续automation已创建ACTIVE（id=dino），仅在Bi-SRNet bisrnet_t1t3_20261011_164416的100轮训练与完整test均正常结束、GPU/进程空闲后调用launch_dino_corrected.sh；先CPU回归/接口核验及GPU实样smoke再正式后台启动。重复启动/部分目录/失败拒绝；无变化静默，成功健康或失败报告后暂停接续。check-only当前WAIT，正式run未启动。

权威准备记录：docs/experiments/dino_integration_20261012/READY.md及17源码/config hash的ready_manifest.json、CPU验收。用户估计明早不是硬启动时刻，本地自动接续需要应用/电脑运行。原训练/测试权重结果与基线保持；未来实际启动及完整成绩另记。
