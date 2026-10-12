# 03 T4 实例分支接管记忆

建立：2026-10-02。负责人：HICD 项目“T4 实例分支”会话。来源：HANDOVER 第 3–6 节；docs/instance_level_plan.md（2026-09-15，存在过期分工）。

## 接管范围

- P1：Bridge、Shelter、Pier、Dock、Tank、Crater 的实例分类与状态识别，合成/采样方案、类别不平衡与评估。
- 审计 instance_dataset.json 的结构、数据 split 归属、稀有类与状态数目。
- 核查 InstanceClassifier、训练脚本、实例候选来源及与像素推理的组合。
- 明确 GT 框分类、候选检测、完整系统指标分别能够证明什么。

## 关键路径

`models/instance_classifier.py`、`script/train_instance.py`、`script/build_instance_dataset.py`、`script/add_bridge_instances.py`、`docs/instance_level_plan.md`；数据 `D:/CD/0617final_256_v2/instance_dataset.json`。

## 已知冲突

旧方案将 T3 也移到实例、保留 Tower；最新交接要求 T3 像素、Bridge 入 T4，不能照搬旧方案。当前像素拟去 T4，旧候选提取却需要 T4 像素预测，必须解决这一依赖。端到端表述与独立训练方案也待审。

## 本轮接手输出

只读核查少量源码与索引元数据，明确实现缺口；在本文件保存证据并向用户发接手清单。本轮不重建索引、不改模型/标签、不启动训练。

## 依赖

01 提供像素模型接口；04 提供标签与评估；02/01 协调训练资源；05 负责与论文论点对应。

## 接手核查记录

### 2026-10-02（Asia/Shanghai）：首次接手完成

会话：`01a0faa7-4660-7a83-9350-ef168cc89155`，注册标题“HICD｜T4 实例分支”。已读 AGENTS.md、公共记忆四份文件、threads.json、本工作流、HANDOVER 相关段落、instance_level_plan.md 及下述源码。核查方式为当前文件读取、JSON 解析与计数、names.txt 精确键匹配、少量路径存在检查和限定路径 Git 状态；未导入模型、运行训练或模型评估。

### 已核实：最新任务意图与实现尚未对齐

- 最新交接（2026-09-28，HANDOVER.md:43–58）要求 T1/T2/T3 像素，T4 = Bridge/Shelter/Pier/Dock/Tank/Crater；本工作线按此范围接手。9/15 旧计划中的 T3 实例化和 Tower 保留不作为当前执行方案。索引保留 T3 历史数据不等于授权把 T3 移出像素分支。
- `models/instance_classifier.py:22–26` 当前 `INSTANCE_CLASSES` 为 Shelter/Pier/Dock/Tank/Aircraft/Vessel/Crater 七类，`CLASS_ID_MAP` 没有原始 target ID 4（Bridge），因此加载时会过滤 Bridge（:59–62），仍接纳 Aircraft/Vessel。尚未形成当前要求的 T4 六类入口。
- 分类器是独立双时相裁剪网络：共享空间分支、低频/高频分支、时相差融合、target/state 两个分类头（:284–365）。默认 crop 64、七状态；频域实现是平均池化与插值残差（:246–253），不能仅凭注释称其为完整 Haar DWT。损失实为加权 CrossEntropyLoss（:375–390），注释“Focal CE”不代表已实现 focal。

### 已核实：索引及状态分布

来源：`D:/CD/0617final_256_v2/instance_dataset.json`，本次读取时大小 4,027,815 bytes，修改时间显示 2026-09-23 14:26:38（文件时间不等于生成来源证明）。根字段为 description/tasks/samples/statistics；tasks 已将 Bridge 列入 T4、保留 T3，未列 Tower。

直接遍历 samples 得到 3,146 个样本键、11,082 条实例；类别实数与 statistics 的 total_instances 一致。T4 共 8,854 条，T3 历史数据共 2,228 条。下表状态数为 JSON 的 statistics 字段值，本轮没有回扫原始标注验证其真实性。

| 类别 | 实例实数 | 索引记录的状态分布 |
|---|---:|---|
| Bridge | 133 | no-change 104；damaged 29 |
| Shelter | 107 | no-change 50；reduced 5；added 52 |
| Pier | 239 | no-change 213；damaged 8；reduced 6；extended 12 |
| Dock | 139 | no-change 123；damaged 4；reduced 10；added 2 |
| Tank | 2,212 | no-change 2,046；damaged 113；reduced 24；added 29 |
| Crater | 6,024 | added 6,024 |
| Aircraft（历史 T3） | 1,021 | no-change 274；damaged 49；reduced 351；added 320；replaced 27 |
| Vessel（历史 T3） | 1,207 | no-change 210；damaged 73；reduced 547；added 291；replaced 86 |

- 实例条目均未见 split 字段；样本键没有 synth 前缀。后者仅为命名核查，不能证明所有影像真实，或合成数据已接入训练。
- 将索引键与当前各 names.txt 精确匹配：train 3,146；val 197；test 197；每两份名单内共有的索引键均为 197；索引键不在三份名单的数量为 0。此为文件名重叠事实；尚未核查影像哈希、场景级归属或这些条目的实际数据路径，不能据此直接量化场景泄漏。
- 重叠例 `Kherson_000000_port_000000000132_01_00` 的 `train/image/*_pre_war.tif` 和 `*_post_war.tif` 均不存在（单例检查）。实际有效加载数、全量缺图数与类别变化仍未知，不能把 11,082 当作当前 DataLoader 的可用样本数。

### 已核实：训练入口存在 P0 数据划分缺陷

来源：`script/train_instance.py:45–111`、`:213–224`；`models/instance_classifier.py:45–70`。

1. 生成 `train_indices/val_indices` 后，两个 DataLoader 仍接完整 InstanceDataset，没有 Subset 或样本过滤；验证遍历全量有效实例，包含训练可见实例。
2. WeightedRandomSampler 的权重按随机 `train_indices` 顺序产生，却直接采样完整 dataset 的位置 `0..n_train-1`，没有将采样位置映射回 `train_indices`，权重与实际实例错位。现有 print 的 train/val 数不能证明真实划分成立。
3. 即使补上 Subset，按实例随机划分仍可能把同一图像/场景的其他实例分到两边；需先和 04 确定官方 split 与场景约束，不能只修局部索引。
4. 类别与状态 loss 权重使用 full_dataset 统计，后续应只从训练集估计。当前训练同时用类别逆频率采样与加权 CE，效果未验证，应拆开消融。
5. 数据集固定读 `train/image`，不按 split 路由；缺一时相时会把另一时相用作两次输入，双图皆缺则静默跳过。需显式审计缺图，不把替代后的输入当作真实双时相。
6. 索引生成 bbox 终点为包含端点（height/width 为 +1），裁剪实现使用 Python 半开区间、宽高不加 1（模型 :95–125）。属于坐标约定差异，影响范围尚未实测。

`evaluate`（train_instance.py:117–157）只输出 GT 裁剪输入上的 target/state accuracy 和每类 target accuracy；best_model 以 target accuracy 选择（:286）。没有验证候选召回、FP、检测 AP、完整状态检测或像素 BDD 指标，现有准确率不能替代完整系统结果。标准保存目录 `checkpoints/instance_classifier` 本次未见；未搜索所有其他目录，也未核查 WSL/screen/GPU，不能断言不存在任何历史训练或当前进程。

### 已核实：重建脚本与当前索引不能直接互换

- `script/build_instance_dataset.py:17–19,30–37,72–107` 只读 train 名单，筛除 synth_ 前缀，从旧 `0617final_256/train/label` 提取七类（无 Bridge），按每个原始状态值分别做连通域，最小 10 pixels，覆盖写入 v2 索引。直接重跑会失去当前 T4 Bridge 元数据/实例。
- `script/add_bridge_instances.py:82–84,98–120,210–219` 将 Bridge 写为 T2_Surface，扫描多个来源和 train/val/test，向已有实例列表 extend；没有幂等去重。直接重跑存在重复追加和污染当前 T4 分类元数据风险。
- Bridge 脚本先合并类别不同状态的 mask 做连通域，再取框内多数状态；主构建脚本按 raw 状态拆连通域，实例定义存在不一致。需由 04 确认物体连通、状态混合、边界截断、邻接规则与坐标约定。
- 当前 JSON 的 Bridge 任务已是 T4，且 statistics 结构与该脚本输出也有差异；说明该脚本不是当前文件的完整可重现来源，后续还须追踪调整来源。
- 内层仓库限定状态：`script/add_bridge_instances.py` 和本工作流文件为未跟踪；其余本轮所读路径未显示工作区修改。此观察仅覆盖上述路径，不代表整个仓库干净或代码已推送。

### 已核实：候选来源仍是实现缺口

- 旧计划 `docs/instance_level_plan.md:113–145` 从像素稀有类预测提取连通域，但其 T3/T4 ignore 方案与候选依赖冲突。当前 `0617final_v11.yaml:79–83` 仍将 Bridge raw 18–22 映射 target 4；这既不能证明已彻底忽略 T4，也不能作为其余 T4 五类的候选来源。
- 已有 `script/infer_instance.py:11–29,38–79` 固定 SECOND 六类配置、v7_second_fix checkpoint 和 SECOND val，最多前 20 个样本，从 Building 概率通道 4 取连通域，再调用主模型 instance_head。它没有调用独立 InstanceClassifier，不能作为 WD-CD T4 六类的现成推理入口。
- 该示例的匹配按每个 GT 独立寻找最佳预测，没有排除重复使用同一预测（:105–127），且仅报告召回与匹配后的状态准确率，未计 FP precision/AP；不可沿用为正式完整检测评估。
- 本轮只审查以上现有入口，未穷举仓库所有候选实现。当前应区分 GT 框分类诊断、预测候选质量、完整系统结果；纯 GT 框裁剪实验只能证明条件分类能力。

### 优先级与具体下一步（后续实作任务）

| 次序 | 工作与产出 | 依赖/完成条件 |
|---|---|---|
| P0-1 | 确认 T4 六类 target/state ID 表、split/场景规则、实例与 bbox 定义，建立可追溯数据清单 | 03 + 04；解释 197 个重叠键、缺图、索引生成来源；T3 保持像素意图 |
| P0-2 | 修正 split 路由、子集和 sampler 索引；加入 Bridge、筛除本入口的 T3；训练权重只取训练集 | 03；先以少量 CPU 数据检查证实索引映射、训练验证隔离、双时相完整性；本轮未修改 |
| P0-3 | 明确 T4 六类推理候选及接口，形成可评审方案 | 03 + 01 + 04；逐类说明候选监督来源、未变化目标覆盖、背景拒识与漏检/FP 评估，避免依赖被 ignore 的类别预测 |
| P1-1 | 在有效划分上建立真实数据 GT 框分类诊断，再做类别/状态采样与加权 CE 消融 | 03 + 01/02 协调资源；按类/状态报告支持数、混淆矩阵与宏平均指标；合成只用于训练并保留来源 |
| P1-2 | 实现候选到双时相裁剪/分类的完整流程，开展独立候选与完整系统评估 | 03 + 04；一对一匹配、precision/recall/AP 或协议确定指标、状态和计数口径，不能用 GT 框成绩冒充检测成绩 |
| P2 | 形成像素/实例组合和完整消融材料，移交论文证据与实验版本记录 | 05/06；明确独立训练/联合训练边界，“端到端”“频域”“稀有类提升”均依实际实现与实验决定 |

跨线：01 确认像素 ignore 与共享特征/候选接口；04 确认数据、标签、实例和指标；01/02 统一 GPU 占用；05 审核论文表述；06 记录版本和实验产出。需要协调的证据已保存在本文件，未向其他会话外发消息。

本轮仅更新本工作流文件，保留原来源与范围；未改业务代码、配置、数据、实验日志或公共记忆，未启动/停止训练、重启 WSL、推送、创建自动任务。接手已完成；实现缺陷尚未修复，实例实验尚未完成。

