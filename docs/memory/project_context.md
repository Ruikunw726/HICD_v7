# 项目背景与状态快照

## 2026-10-08：新严格联合指标实验已从头启动

用户要求每个目标类别/任务按目标与原始状态完全正确计算P/R/F1，并告知v11已终止、授权重新训练。01已核查没有旧v11训练Python或screen；旧权重、日志保留。04已完成严格联合流式模块/训练包装器/协议文档，27项CPU测试及独立随机/反例复核通过。01已实际启动独立新实验 `v12_strict_joint_20261008`。

- 新协议：`hicd.strict_joint.v1`；第1/50轮从零任务权重、优化器、scheduler与AMP开始，仅正常加载DINOv3预训练骨干。日志标记 `task_checkpoint=None optimizer_states=0 scheduler_last_epoch=0`，未从v11恢复。
- 2026-10-08 16:06:57 +08:00 主会话读取的健康快照：第1轮44/1907 batch，运行平均loss10.6561，loss有限，fatal_patterns空。screen=`452.hicd_v12_strict_joint_20261008`，Python PID=459。此为观测快照，不代表后续实时状态。
- 新目录：`checkpoints/v12_strict_joint_20261008/`；train.log、128列严格train_log.csv、manifest.json、配置/名单/源码snapshot、preflight_smoke.json、early_health.json，以及每轮metrics。best按验证集全前景严格joint micro-F1最大值选择。
- 实际隔离smoke做两步合成训练和一个真实验证batch（2张样本），严格评估/旧IoU自适应反馈/序列化通过，输出105个target-state明细及联合混淆计数；正式进程重新初始化，smoke结果不作为完整模型成绩。
- 执行会话回报前复核于2026-10-08 16:09:31 +08:00已到第1轮124/1907 batch，运行平均loss9.8819、loss有限且无致命异常，证据为新目录 `early_health_followup.json`；本轮严格评估实现和新训练启动均已验收。
- 尚未完成第1轮完整验证或50轮训练。实际像素配置仍仅对Bridge提供T4 GT，其余多数设施ignore；新严格计分不会恢复缺失/忽略的GT。

新主类别指标为strict_joint P/R/F1，target-only定位和GT目标区域内state诊断并列，逐target-state明细用于定位问题。禁止ND/D合并和0.3/0.7作为新主F1；旧v11中的BDD字段属于历史协议。新实验从任务权重/优化器epoch0开始，沿用预训练骨干、原模型/损失/数据/采样/实际标签映射和raw状态64×64评估口径。Bridge/T4 ignore等独立问题未隐式修正。

## 2026-10-08：v11 已实际恢复训练

用户在总览会话授权继续 v11，执行由“训练恢复与方法改进”会话承担，另经只读子代理复核入口。模型/数据/标签/评估沿用原 v11 实际设置。

- 从 `checkpoints/v11_no_t4/checkpoint_epoch_32.pth` 恢复，进入第 **33/50** 轮；严格模型键加载通过，349 项优化器状态、scheduler 和 AMP scaler 恢复核验通过。
- 持久会话：`571355.hicd_v11_resume`；训练 Python PID：`571365`。2026-10-08 11:07:49 +08:00 的早期健康快照已观测到 **41/1907** batch，运行平均 loss **3.0771**，未见 fatal patterns。进程和 batch 是该时点快照，不能作为后续实时状态。
- 执行会话完成回报前于 **2026-10-08 11:09:30 +08:00** 再次核实到 **96/1907** batch，运行平均 loss **2.9541**，未见 Traceback、OOM 或非有限 loss；证据为审计目录 `early_health_followup.json`。本轮恢复任务已验收，训练继续在后台执行，50轮尚未完成。
- 本次审计与备份目录：`checkpoints/v11_no_t4/resume_20261008_110218/`，包括 `manifest.json`、`early_health.json`、`command.txt`、`status.txt`、原日志/CSV/best/epoch32 权重与配置/名单快照；本次日志为 `train_resume.log`。
- `launch_v11.sh` 已转接 `launch_v11_resume.sh`，配合 `script/resume_v11.py` 做精确恢复、重复启动保护、错误日志保留；原 CSV 继续追加。首次 screen 初始化失败后仅处理本次失效 socket，重试已进入训练，旧故障根因未确证。
- 旧 checkpoint 未保存完整自适应权重历史和 RNG，恢复不是逐位连续复现；原 phase 重置行为及研究协议保留。已知标签/数据/评估争议仍在案，此次续训不等于科学协议修正或论文结果验证。

依据：01 专题执行反馈及主会话读取的 `early_health.json`、`manifest.json`、`status.txt`；更晚进度查看 01 专题记忆与训练日志。目标仍为50轮，当前记录只确认恢复与早期健康。

> 下文保留首次整理时的证据等级。六个专题随后完成接手审计；最新补充见 [接收回执](intake_receipt_2026-10-02.md) 和各工作流记忆。特别注意：v11 epoch 32 的 F_overall 已核实为 Building 指标；跨 split 同名交集已实测；基线进度与最近报告日期已更新。这里的“待核查”如已被后续记录覆盖，以有来源、较新的核查为准。

## 项目定位（交接记录）

HICD = Hierarchical Instance Change Detection，战争损毁语义变化检测。当前重点是方法论文 P1，不是重新发表 WD-CD 数据集论文。多任务统一框架、GNN 标签空间、背景抑制损失、像素与实例协同是待实证支撑的论点。历史目标数字为 CD-Lamba 79.53%，指标与数据可比性尚须审查，不当成已核实可直接比较的 SOTA。

## 路径（2026-10-02 已核实存在）

| 内容 | 路径 |
|---|---|
| Codex 项目 HICD | F:/mambacd |
| 主代码，独立 Git 根 | F:/mambacd/home/HICD_v7 |
| WSL 主代码映射（交接记录） | /mnt/f/mambacd/home/HICD_v7 |
| 基线代码 | F:/mambacd/baselines_compare |
| 数据 | D:/CD/0617final_256_v2 |
| 实例索引 | D:/CD/0617final_256_v2/instance_dataset.json |
| 日报目录 | E:/docs/changedetection/日报 |
| WSL Python（待运行核实） | /root/miniconda/envs/mamba/bin/python |

## 任务分工（2026-09-28 决策）

- T1：Building，像素级。
- T2：Farmland、Runway、Taxiway、Apron、Highway，像素级。
- T3：Aircraft、Vessel，像素级。
- T4：Bridge、Shelter、Pier、Dock、Tank、Crater，实例级。
- Tower 不在交接所列当前 T4 名单中；旧方案包含它，属于历史冲突。

## 数据状态

2026-10-02 对当前三个 names.txt 的逐行读取：train 61,029，val 13,418，test 13,418；非空行数相同；三者均没有包含 `synth_crater_` 的行。此核查不证明文件解析完整、不重复、无泄漏或所有样本均真实。
交接称像素级用纯真实数据，实例级可合成与过采样。历史合成弹坑 10,746；`names.txt.bak_synth`、`names.txt.bak_rare` 的用途来自交接，尚未逐一核实。

实例计数（交接记录，非重新扫描）：Aircraft 1,021；Vessel 1,207；Bridge 133；Shelter 107；Pier 239；Dock 139；Tank 2,212；Crater 6,024。

## 实验历史与当前观察

| 实验 | 历史记录 | 证据等级 |
|---|---|---|
| v8_bg_suppress | 完成；F_overall 78.49%；真实+10,746 合成 | HANDOVER 与 experiment_log；split/汇总范围待追溯 |
| v10_real_only | 完成；摘要 F_overall 77.48%；T1 表写 0.7851；T2 0.5447；T3 0.6732；T4 0.4843；Global SeK 0.1053 | 交接记录，摘要与 T1 不能混用 |
| v11_no_t4 | 交接称 9 月 28 日因 WSL 中断 | 历史状态已过期，见下方核实 |
| FC-EF 单任务 | T1 0.89 / T2 0.65 / T3 0.45 / T4 0.16 | 9 月 18 日历史摘要；最新模型与 test 结果待核 |
| FC-Siam-diff | T1 0.42，35/50，其他任务排队 | 9 月 18 日历史摘要；运行状态待核 |

2026-10-02 已读 `checkpoints/v11_no_t4/train.log`：末行为 Epoch 32/50，F_overall 0.7668，F_loc 0.8188，F_clf 0.7446，SeK 0.1191；这些是日志原值，不宣称为 test 或当前 SOTA。
目录有 `checkpoint_epoch_32.pth`、`best_model.pth`，最后修改日期显示 2026-09-29；未看到 `checkpoint_epoch_latest.pth`。运行中与否尚待进程核查。

## 版本状态（2026-10-02）

外层 Git HEAD：`ecf8a0d`；内层 HICD_v7 Git HEAD：`2834e23`，与交接内层提交一致。两层均有大量既有工作区变更/删除和未跟踪内容；提交号不能代表当前全部实现。未经专项核实不能声称已全部提交或已推送。

