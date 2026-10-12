# 04 数据与评估接管记忆

建立：2026-10-02。负责人：HICD 项目“数据与评估口径”会话。来源：HANDOVER 第 3、4、6、8 节；0617final*.yaml；evaluation/bdd_protocol.py；baselines_compare/PROTOCOL.md。

## 最新状态：2026-10-08 严格联合协议已实现与测试

### 2026-10-09 摘要停更故障修复

用户指出summary仅到epoch10而实际训练到40多。现场核实：训练主PID1716仍正常，train.log已进入epoch44，metrics完整epoch001–043存在；原报告器PID11309已退出（无进程、无独立screen）。此前依附持续exec session的启动方式未保证持久性，不能据前一日存活就承诺后续更新；未确证退出时点/退出码，不猜测具体终止机制。

已以现有完整metrics补写epoch11–43；重新读摘要验证43个完整block、与43个artifact一致，零遗漏/零重复。没有重算模型、改变指标、重写原JSON/训练log/CSV、停止训练或占用GPU。全部指标仍四位小数。

报告器改用独立detached screen `118431.hicd_strict_summary_t123`，实际Python PID118433，绑定训练PID1716；在启动shell返回后及独立后续检查中持续存活，自动记录PID至evaluation_reporter.pid。首次直接screen fork未保活，改为启动shell保留3秒初始化后成功；新增 `launch_strict_summary.sh` 把初始化存活/Detached校验做成最多10秒的有界流程，识别同run已运行的publisher并拒绝重复启动；实际调用返回SUMMARY_ALREADY_RUNNING pid=118433。bash -n及4项presentation数值/去重测试通过。

新launcher用于后续持久启动，报告器仍读取原子完成的epoch JSON、持有单实例flock；训练退出后最终补一次并退出。该run旧manifest/source快照继续保留，不冒充最新展示层代码快照。最终已完成artifact数以文件实际新增为准，本记录初始核查为43；epoch44训练/验证中的数据不伪造为已完成报告。

### 2026-10-08 后续用户要求：日志四位小数，下个epoch可见

用户直接要求修改代码输出而非只改聊天展示。核实当前实际运行的是 `v12_strict_joint_t123_trainfix_20261008`，主Python PID1716；查看时处于epoch2。01新增训练器的TrainingEvaluator覆盖了原四位小数formatter，用原始float插值，导致冗长指标。现新增 `evaluation/strict_joint_reporting.py`，统一compact格式；公共包装器/训练器formatter调用它，训练器epoch/best日志及新启动CSV浮点字段四位小数，CPU缓存入口也使用同简表。省略全空类别，保留仅FP类，核心统计/best比较/完整JSON精度不变。

当前进程已将旧formatter加载到内存，改源码不使其下轮自动重载。因此没有停止训练或注入进程，新增并实际启动 `script/report_strict_epochs.py --watch --training-pid 1716` CPU报告器，读取原子完成的逐轮JSON、以相同数据写当前run的 `evaluation_summary.log`；已有epoch1简表已生成核实，新epoch报告完成后自动追加，训练退出时报告器结束。它持有单实例flock，不读写权重，不消耗GPU、不改原log/CSV/manifest。最初WSL nohup后台尝试未保活，改用独立持续exec session（57238）实际核实报告器PID11309，记录于evaluation_reporter.pid；stdout由该exec session保留。evaluation_reporter.log为初次尝试留下的空文件，不当成运行证据。当前原train.log仍由原进程写旧格式，查看新简表须打开evaluation_summary.log。

新增4项presentation测试，连同原27项strict测试，31项本地CPU验收通过；包括四位小数、计数整数、仅FP/空类、JSON不变和多轮报告去重。源码在原训练冻结快照之后有展示层修改，原manifest/source快照不覆盖；实际当前训练数值路径仍为其已加载实现，后续启动须使用新快照，不能复用旧hash冒充一致。

**用户新决定覆盖下文旧 BDD 主口径和 10/02 只读接手限制。** 本轮授权来自总览会话转交的用户明确要求“严格的什么类别就是什么类别”“类别f1也按照目标、状态全对才算对”。本工作线新增严格指标/独立包装器/CPU入口/测试/协议文档，不改旧业务入口、模型、loss 或数据；主会话维护公共决定。

### 已核实实现

- `evaluation/strict_joint_protocol.py`：冻结 `hicd.strict_joint.v1`，全前景精确 target+state micro 为 `strict_joint_f1`。原七状态均独立；state 5 replaced 保留，预测前景 state 6 none 不筛掉，错状态 FP+FN。统一 `(gt_target!=255)&(gt_state!=255)` 用于全部统计，有效背景及其他类别上的错前景均计 FP。
- 固定 106×106 int64 联合矩阵和 15×7×7 int64 GT目标区域状态矩阵，矩阵共 95,768 字节；仅当前批次临时数组，不累计全 test 预测。按数据全局累计整数再算 F1，不平均 batch 分数。提供 update/compute/merge；无 GT 但有 FP 的类别保留，GT/预测均空的类别记 null 且排除 macro；单状态完美 F1=1。
- 每任务（最新 Bridge→T4）micro/macro；每目标主 joint P/R/F1 与 TP/FP/FN/support/predicted_support；并列 target_localization P/R/F1 诊断；每目标 GT 区域内 state_on_gt_target 全七状态 P/R/F1、micro/macro、support、7×7矩阵。状态诊断不按预测 target 筛选，含漏检/错类像素，不冒充系统联合成绩。
- 全部105个 target-state 类详细统计及联合矩阵全部非零项可追溯。背景 target15 合为一个 bin、不进前景主分；背景预测无论 state 为何都是背景。Tower 不在四任务但若输入有效 GT/预测仍纳入全前景。
- `evaluation/strict_joint_evaluator.py`：训练可显式 opt-in 的 `StrictJointModelEvaluator(dataset_config)`，稳定接口 `reset()` / `update(predictions, targets, loss_dict=None)` / `compute()` / `format_results(result)`。读取 target_logits、默认 raw state_logits（非 refined），GT target_gt 与 state_gt/change_state；GT nearest 对齐 logits 实际网格，通常64×64，与旧训练主BDD的空间/支路口径保持一致；输出记录输入GT/评估网格与样本数。包装器不依据 loss 的 state_ignore_targets 删除 Farmland。
- `script/eval_strict_joint.py`：NumPy CPU NPZ 分片入口；四个已解码同网格整数数组；记录来源 SHA256/producer provenance/声明 split；拒绝缺字段、重复文件、空输入和覆盖已有输出。只统计提供分片，不自动证明全 test 覆盖或样本唯一性。
- `docs/strict_joint_protocol.md`：v1 标签、micro/macro/背景/ignore/none/无支持类政策、字段表、接口、缓存示例、数值验收及可比性限制。

### 已执行验证（2026-10-08，非模型成绩）

在 `F:/mambacd/home/HICD_v7` 使用 Windows `F:/python/python.exe`（Python3.14.6、NumPy2.4.4），执行：

`python -m unittest discover -s tests -p test_strict_joint_protocol.py -v`

**27 tests 通过、无 skip**，含真实CPU模型包装器张量测试和CPU CLI临时NPZ分片调用，不启动模型/GPU。固定反例：

1. 4像素全Building，GT state=[0,3,1,2]，pred=[3,0,2,1]：TP0/FP4/FN4、joint micro/macro=0，loc F1=1；测试状态诊断F1=0。
2. GT target=[6,6,6,15,12,15,255]，state=[3,1,0,6,5,6,1]，pred target=[6,6,6,6,6,15,6]、state=[0,1,6,3,5,0,0]：jointTP1/FP4/FN3，globalmicro2/9、macro1/6；T1micro1/4、macro1/5；T3F1=0；Building loc3/4。
3. 错目标/预测背景但状态正确时joint/loc=0，GT目标状态诊断=1，证明诊断没有只挑正确定位像素；前景none错预测计FP+FN；单状态Crater完美micro/macro=1。
4. 随机 tuple Counter 参照（与矩阵编码独立）及跨批次累计/merge核验通过；GT target/state任一255均不影响全部统计；空类/仅FP、schema缺状态/错序、非法ID/浮点/非有限logits/尺寸错配均覆盖。

### 当前交付边界与依赖

本轮只编辑两个新增 evaluation 模块、一个新增 script 入口、一个新增 tests 文件、协议文档和本04记忆。旧BDD/evaluator/__init__/训练入口/log/CSV均未改。当前 v11 的原 CSV/BDD 字段继续标为 legacy，不能改名为 strict；边际混淆矩阵或旧汇总分数无法反推联合分数。

来源为主会话10/08后续消息：用户已要求终止v11后按新标准重新训练，01正在准备独立新实验 `v12_strict_joint_20261008`。本工作线没有核查或操作训练进程，不把转交的终止/准备状态当作已独立验证的实时事实；不改01的新脚本/launcher，不启动推理或训练。

**真实权重 strict 重评未做，未产生模型严格成绩。** 当前没有本工作线确认可用的共同 target/state NPZ；主会话报告只见旧边际 confusion_matrix.npy。交付稳定训练包装器与缓存CPU入口，依最新范围不另做完整checkpoint推理入口。后续权重重评须按 checkpoint args/config快照建模、strict恢复动态buffer/模型键，资源协调后独立执行。

已知有效性风险仍保留：v11大部分T4 GT映射255（Bridge例外），像素T4新分不代表六类实例系统；预测在其他有效区域的T4类仍计FP；train/test同名197、val/test341尚未解决。GTnearest采样会改变小目标/ignore覆盖；raw/refined/网格/标签解码不同必须在实验证据中分开，协议版本相同不保证实验条件相同。

下一步：01在独立新训练验证循环opt-in包装器，保留诊断与legacy字段明确分列；02以同全局ID/GT政策/网格/支路重评基线；03另定义实例精确目标状态匹配，不能套像素分母；06记录协议/配置/名单/代码/checkpoint版本；05仅使用实际strict结果，旧数字不可反推。数据独立性、标签归属和实例候选链路的旧P0仍未由本指标实现解决。

## 接管范围

- P0：固定 train/val/test 名单口径、真实/合成/采样备份，审计重复与 split 交集。
- P0：T1–T4 类别归属、原始编码/target ID、Bridge ignore 与 state mask 一致性。
- P0：BDD F_loc/F_clf/F_overall、SeK、背景及空类边界；HICD 与基线相同输入的输出一致性。
- 建立“实验-数据-split-任务-协议-checkpoint”证据记录供结果比较。
- 对比文献数据/协议，先明确 79.53% 是否可直接作为本项目比较目标。

## 初始核实

2026-10-02 names.txt 非空行 61029/13418/13418；无 synth_crater_ 行，尚未验证唯一性、文件齐全或泄漏。
当前两个 YAML 中 Bridge 原始 18–22 → target 4，非 ignore；BDD 的 T4 列表含 Bridge。T3 Aircraft/Vessel 在 YAML 中仍保留。
标准 IoU 包含 FP；旧“忽略假阳性”说法可能是 masked 评估实现问题。
PROTOCOL 自述跨 split 重叠，但这仅是待审查来源记录。

## 本轮接手输出

做有限文本和名单核查（可计算集合交集），记录精确文件/范围和冲突；只更新本文件并直接向用户回复接手清单。本轮不重划数据、改 YAML 或评估器，不将未跑的数值验证写为已通过。

## 关键路径与边界

主代码：`datasets/dataset_v7.py`、`datasets/configs/0617final.yaml`、`datasets/configs/0617final_v11.yaml`、`evaluation/bdd_protocol.py`、`evaluation/icd_eval_v6.py`；基线：`task_labels.py`、`bdd_unified.py`、`check_leakage.py`。共享协议修改先通知相关会话/记录依赖，由本线集中处理。

## 接手核查记录

### 2026-10-02（Asia/Shanghai）：接手完成，协议尚未对齐

本会话注册：`01a0faa7-4c8d-76c3-a27c-d70dae628c62`，标题 `HICD｜数据与评估口径`；依据 `../threads.json`。已阅读根 AGENTS、README、project_context、decisions、risks_and_questions、本工作流及 HANDOVER 相关段落。核查为 Windows 本地名单读取、文件哈希与代码静态审计；未运行数据集加载、数值回归、训练或模型评估，未改变划分、配置与业务代码。内层 Git 根已核实为 `F:/mambacd/home/HICD_v7`；本文件在核查时为未跟踪文件，未提交。

#### 已核实：当前名单重复与交集

来源：`D:/CD/0617final_256_v2/{train,val,test}/names.txt`。按逐行 Trim、去空行、大小写敏感（Ordinal）字符串集合统计，口径对应 dataset_v7.py:354–355 的 strip/去空行。本轮不规范化路径、不按图像内容去重。

| split | 总行/非空行 | 唯一名称 | 内部重复额外行 | synth_crater_ 行 |
|---|---:|---:|---:|---:|
| train | 61,029 | 61,029 | 0 | 0 |
| val | 13,418 | 13,418 | 0 | 0 |
| test | 13,418 | 13,418 | 0 | 0 |

| 名单集合 | 同名交集 |
|---|---:|
| train ∩ val | 197 |
| train ∩ test | 197 |
| val ∩ test | 341 |
| train ∩ val ∩ test | 197 |

三个集合共 87,865 个名单条目，按上述名称口径全局唯一 87,327 个；val/test 还有 144 个共享名称不在 train。例：`Kherson_000000_port_000000000128_00_00` 同时在三个集合；`Donetsk_000001_20140220_20240208_000000000020` 出现在 val/test 的交集。完整交集未另存文件，必要时按同一口径重算。

名单 SHA256（本次文件快照，后续以哈希识别版本）：

- train：`E8E157E78AF20B6160D01596B17EF66199D2B73B8E98F378E29B323542C38F77`
- val：`9075C7365E4A6D8FDCB195D1BD0A68AD39D64E11ABABB79DEA206740851D1786`
- test：`D8E844F5E5BCA50339CCA4C64669DFE6ABD33A67AD221B6653D543DB66B7C630`

**科学证据风险 R05，P0：** PROTOCOL.md 所述跨集合重叠已获得名称级证据。“有意重叠”不能证明独立泛化评估有效；同一名单/公式也不足以消除训练与测试重叠的影响。dataset_v7.py:352–406 按文件名索引搜索所有原 split 和兄弟数据目录，并保留首次遇到的路径，因此相同名称有解析到相同源文件的风险，具体路径/文件内容仍待核实。未全面检查前不能将同名交集直接等同已经验证的文件内容重复，也不能声称其余样本无场景泄漏或文件齐全。无 synth_crater_ 仅排除此命名模式，不证明全部真实。

#### 已核实：训练名单备份

来源：仅 `D:/CD/0617final_256_v2/train/` 的三个备份，使用相同字符串口径。

| 文件 | 非空行 | 唯一名称 | 重复额外行 | synth_crater_ 行 |
|---|---:|---:|---:|---:|
| names.txt.bak | 64,481 | 64,481 | 0 | 3,452 |
| names.txt.bak_synth | 83,900 | 71,919 | 11,981 | 10,746 |
| names.txt.bak_rare | 76,700 | 71,775 | 4,925 | 10,746 |

备份含合成命名和重复条目已核实；重复条目的生成原因、对应历史实验、采样倍率未核实，不由文件后缀推断。`bak_synth` 的行数不是简单的 61,029 + 10,746；后续恢复名单时必须按完整版本记录追溯，不能只按后缀选文件。

#### 已核实：标签与任务定义冲突（R03/R04，P0）

来源：`datasets/configs/0617final.yaml`、`0617final_v11.yaml`、`evaluation/bdd_protocol.py:193–214`、外层 `baselines_compare/task_labels.py`。

| 项目 | HICD 当前两个 YAML / BDD | 基线 task_labels.py |
|---|---|---|
| Bridge 原始 18–22 | target 4，未 ignore；BDD 归 T4 | target 4，但 TASKS 归 T2 |
| Shelter 33–37 | target 255 | target 7 |
| Pier 43–47 | target 255 | target 9 |
| Dock 48–52 | target 255 | target 10 |
| Tank 53–56 | target 255 | target 11 |
| Crater 67 | target 255 | target 14 |
| Tower 38–42 | target 255 | target 255 |
| 原始 68 | target 255，state 3 | target 255，state 6 |
| Aircraft / Vessel | target 12/13，保留像素类别 | target 12/13 |

当前 YAML 的 `state_ignore_targets` 仅含 Farmland(0)，Bridge 不在其中；其他 T4 虽 target ignore，其 raw state_decode_map 仍保留状态。state loss 是否同时按 target=255 屏蔽、实际训练读取哪个配置，须与训练线沿实际入口追溯。本轮没有把静态 YAML 当成 checkpoint 的有效训练配置。

`bdd_unified.py` 直接导入 HICD 的 evaluate_bdd_protocol，但任务列表来自基线 task_labels；`eval_fcef.py:69` 向其传 raw_label，GT 会通过基线自己的静态 LUT 解码。故“共用公式”已核实，“相同类别/标签/任务协议”目前不成立；基线文件声称 LUT 来自 HICD YAML，不等于运行时同步读取 YAML。

#### 已核实：BDD、IoU 和 SeK 的实际边界（R06/R07/R08）

- `bdd_protocol.py:108–157`：无该类 GT 则返回 None；F_loc 对传入全部位置计 TP/FP/FN，没有显式剔除 gt_target=255；GT ignore 区域若预测为该类，也会进入其 FP。背景参数 bg_class 未用于显式过滤。
- F_clf 使用 ND={0,3,4}、D={1,2}，排除 GT state 5/6；定位错误会进入 FN，但状态 FP 仅来自该目标 GT 内的 ND/D 互相错判。背景/其他目标上的状态预测不进入 F_clf 的 FP。不能将其描述为全图联合类别 F1。
- F_clf 是 F_ND/F_D 的调和平均，F_overall=0.3 F_loc+0.7 F_clf；若某折没有 GT，其 F1 按当前分母保护可为零。仅 added 的 Crater 等单折任务需要专项边界核验，不能直接把低 F_clf 解读为模型失败。本轮只静态读取，未执行这一数值用例。
- `icd_eval_v6.py:250–284`：任务内对存在 GT 的类别做算术平均；没有可评类别的任务记零；`bdd_f1` 再平均四任务。顶层 `F_loc/F_clf/F_overall` 专指 Building(6)，并非四任务均值。历史数字需回到对应输出字段与代码版本核验，不能直接拼表。
- `bdd_protocol.py:245–257` 的注释出现 best/weighted average，但实际为 np.mean。应按实现记录算术平均；缺失类别、NaN 与跨 batch 聚合影响待核验。
- `icd_eval_v6.py:294–305`：IoU 在 GT!=255 的位置用 intersection/union，**包含该有效范围内的 FP**。因此 HANDOVER 的泛称“IoU 忽略假阳性”不正确；被 ignore/mask 排除区域的 FP 不进入该 IoU。类别平均的 present 实际是 union>0，包含纯预测出现类，不是注释所说仅有 GT 的类。
- `bdd_protocol.py:162–168`：类别 SeK 仅保留 GT 目标区域且 GT/pred state<6，不检验预测 target 正确与否；预测 none(6) 的位置被排除，存在选择性 mask 风险。
- `icd_eval_v6.py:234–248`：global SeK 排除 GT background(15)、ignore(255)，并要求 GT/pred state<6；NaN 转零。它是目标区域状态指标，不能直接称为全图语义变化 SeK。
- `bdd_protocol.compute_sek` 在 eta≈1 时返回 1/0；`icd_eval_v6._compute_sek` 的相同分母边界返回 NaN（调用点 144/157）。后者按 num_states（可为 7）过滤 GT，未对预测同步过滤，也不直接排除 state=6；两条 SeK 路径的边界与 mask 不一致。“已修复”只能用于对应函数，不能泛指所有输出。尚未运行边界回归或核对论文数学定义。

#### 遗留问题、优先级与具体下一步

| 优先级 | 当前状态/阻塞 | 下一步交付 | 依赖 |
|---|---|---|---|
| P0 | 跨集合同名已经实测；同源文件/场景级泄漏未知 | 解析重叠 197/341 名称对应 pre/post/label 路径，核对哈希、场景/坐标/时间来源；输出泄漏清单与现有结果适用范围。先审计，不覆盖当前 split | 训练/基线提供实际数据加载入口与根目录；实例提供索引来源 |
| P0 | Bridge 归属、T4 标签 ignore、state mask、基线 LUT 不一致 | 建立原始值→target/state→任务→train/metric mask 的逐项协议表；追踪 v11 启动配置与 loss；交由关联线协调后修复 | 01 训练、02 基线、03 实例 |
| P0 | 同函数不等于同输入；BDD 两层均值与 Building 字段混用风险 | 小型同输入数值对照：无该类 GT、纯 FP、完美 ND/D、单折、ignore 区、预测 none、eta=1、全 no-change；记录 TP/FP/FN、NaN 与聚合策略。用合成数组验证实现，不跑长评估 | 02 基线与 01 训练确认规范；需可用 NumPy 环境 |
| P1 | 历史实验缺完整可比记录，当前名单未证明等于历史名单 | 建立实验证据表：实验名、数据/名单哈希、真实/合成/重复、split、任务/粒度、mask/聚合、checkpoint、有效代码版本、指标字段。追溯 v8/v10/v11 和基线来源 | 01/02 提供日志/配置/权重；06 管理版本与台账 |
| P1 | 像素 T4 与实例 T4 度量单位不同；单折类别解释未统一 | 与实例线定义对象匹配、候选来源、检测/分类分母；明确像素 BDD 与实例结果的表格边界 | 03 实例、05 论文 |
| P2 | CD-Lamba 79.53% 的数据、split、任务、mask 及引用未核验 | 与论文线核对原文与协议，制作“可直接比较/仅参考”映射，未齐前不作 SOTA 结论 | 05 论文 |

本轮没有环境阻塞：名单和代码静态核查均可完成；运行 NumPy、真实 loader、模型及完整数据文件核查尚未尝试。研究结果用于论文主表的阻塞来自上述独立性与协议冲突，接手完成不表示问题已经修复。

#### 协作边界与当前交付

本会话集中维护共享标签/划分/评估契约；训练线负责实际训练/loss，基线线负责基线入口/LUT，实例线负责实例索引与检测评估，论文线负责文献协议与结论，实验管理线负责版本和证据台账。本轮仅修改本工作流记忆；没有向其他会话发送消息、改公共注册表、执行 WSL 检查、启动或停止训练、重划数据、提交/推送或创建自动任务。后续业务实作按用户在本专题交付的任务推进，修改共享协议前先核对各线记忆与运行依赖。

