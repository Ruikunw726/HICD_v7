# 02 多任务基线接管记忆

建立：2026-10-02。负责人：HICD 项目“多任务基线对比”会话。来源：HANDOVER 第 10 节；baselines_compare/PROTOCOL.md、MULTITASK_PLAN.md。

## 接管范围

- P0：核实 FC-Siam-diff 当前队列/日志/权重与 FC-EF 已有 test 结果，历史“35/50”不能视为现状。
- P0：实现准备：全类别联合 target-state + background 的 FC-EF-multi / FC-Siam-multi，单模型、一套权重。
- 历史推荐预算 50 epoch、batch 8、约 4GB；具体条件需当前设备和公平比较要求支持。
- test 按 T1–T4 的 BDD 输出，并保存可复现主表；单任务历史结果独立标注。
- 扩展候选 SNUNet、STANet、BIT、ChangeFormer、ChangeMamba 等先审查代码/数据/预算，再决定实验。

## 关键路径

`F:/mambacd/baselines_compare` 下：`train_fcef.py`、`eval_fcef.py`、`task_labels.py`、`bdd_unified.py`、`models_fcef.py`、`runs/`、`results/`、`repos/FCEF_daudt`；计划中的 `train_multitask_fcef.py` 初始目录列表未见。

## 本轮接手输出

有限只读核查文件、现有实验状态及多任务实现差距；更新本文件并直接向用户发负责清单、优先级、下一步及依赖。本轮不启动队列、不改基线代码/共享协议。

## 依赖与风险

04 负责划分交集和协议对齐；01/03 共享 GPU；05 接收公平主表；06 接收台账。PROTOCOL 里的跨 split 重叠和外部数据不可比说明需要实际审查。Bridge/Tower/T3 采用公共冲突清单，不能复制旧类别口径。

## 接手核查记录

### 2026-10-02（Asia/Shanghai）首次接手完成

会话：`01a0faa7-40ef-7582-aade-33727bf88614`，题名“ HICD｜多任务基线对比 ”，归属已对照 `../threads.json`。已读 AGENTS.md、公共背景/决策/风险、本工作流、HANDOVER 第 10 节、PROTOCOL.md、MULTITASK_PLAN.md。以下为本轮有限只读核查；仅维护本文件，不代表已完成多任务改造或实验。

#### 已核实：训练文件快照

来源：`baselines_compare/runs/<run>/history.json`、`summary.json`、权重目录及 `runs/logs/fcef_fullstate.log`。数值均为历史训练文件中的 **val / legacy 统计**，不是本轮运行评估所得，不是 test 公平主表。best 为 summary 中 `best_val_score`，末轮为 history 中 `val_F_overall`。

| run | 已保存完整轮次 | best val score | 末轮 val F_overall | train / val（日志） | 文件状态 |
|---|---:|---:|---:|---|---|
| ef__T1 | 50 | 0.893126 | 0.890050 | 79,303 / 30,127 | summary、best.pth、last.pth；history/summary 修改日期 9/24 |
| ef__T2 | 50 | 0.652171 | 0.640419 | 61,029 / 13,418 | summary、best.pth、last.pth；history/summary 修改日期 9/25 |
| ef__T3 | 50 | 0.452546 | 0.407254 | 61,029 / 13,418 | summary、best.pth、last.pth；history/summary 修改日期 9/26 |
| ef__T4 | 50 | 0.156996 | 0.146874 | 61,029 / 13,418 | summary、best.pth、last.pth；history/summary 修改日期 9/27 |
| siam_diff__T1 | 50 | 0.508203 | 0.490621 | 61,029 / 13,418 | summary、best.pth、last.pth；history/summary 修改日期 9/28 |
| siam_diff__T2 | 23 | 未有完成摘要 | 0.228216 | 61,029 / 13,418 | history、best.pth、last.pth；history 修改日期 9/29 08:42；日志末尾为 ep24 step6000/7628，修改日期 9/29 09:16 |

- 当前 `runs` 目录未见 `siam_diff__T3` / `siam_diff__T4`；不能将它们记作已完成或正在排队。
- 交接的 Siam T1 “35/50、0.42”已被更晚文件覆盖；T2 只确认保存到完整第 23 轮，未读取 checkpoint 内部 epoch/优化器元数据，不能直接认定可无损续训。
- ef__T1 的日志还记录 train missing=4,597，其他以上启动记录 missing=0。**79,303 在该 run 中是实际解析出的 train_n，不只是可以忽略的旧扫描数字**；PROTOCOL 第 3 节对此数字的解释和“所有基线同划分”的陈述不能覆盖该 run 事实。名单历史版本及样本组成待追溯。
- 只读 WSL 默认发行版进程快照：`ps -eo pid,etimes,args` 过滤 train_fcef.py、watch_and_summarize.sh、eval_fcef.py 未返回匹配；同次 nvidia-smi 返回显存 1,819 / 24,576 MiB、GPU 利用率 17%。仅对该查询时点和发行版有效，不证明所有其他任务空闲，不据此启动训练。探针设 12 秒等待上限，未重启 WSL、未停止训练。

#### 已核实：test 证据与现有实现差距

1. 当前 `results/` 只有 `summary.md`、`eval_batch.json`、`protocol_compare_T1.json`。未见 `ef__T*.json`、`siam_diff__T*.json` 或 `eval_test.json` 成功产物。`eval_batch.json` 的 EF T1/T2 均为 error=true；`summary.md` 仍是旧 loc-only 表，test 表为空，不能作为当前汇总。此结论仅覆盖该目录，未搜索全盘其他归档。
2. `protocol_compare_T1.json` 是 **val、checkpoint epoch19** 的旧诊断：delta_overall=0.00013684；joint GT 与 raw GT 的 SeK 分别为 0.0200505 / 0.0568631。它不能证明当前所有任务/指标/权重已对齐，也不能作为 test 证据。`runs/fcef_siam_summary.json` 仍是 9/20 的全 error=true 旧汇总，不覆盖更晚单 run 摘要。watcher 日志只见 9/21、9/22 的 watch start，未见成功 test 完成记录。
3. `train_fcef.py` 按 variant、task 循环调用 `train_one()`，分别保存 `<variant>__<task>`；dataset、wrapper、joint decode 均依赖 task_id。`--tasks T1 T2 T3 T4` 是四个独立模型，不能算单模型联合训练。本轮检索 baselines_compare 的 *multitask* 文件未见实现，顶层未见 `train_multitask_fcef.py`；MULTITASK_PLAN 仍属于计划。
4. 当前输出是 **稀疏有效 target-state 联合类 + bg**，包含 T3 的 replaced；task_labels / models_fcef 部分说明文字还写 ND/D 双类，已与函数实现不一致。已有组件可作为联合改造起点，但尚无全类别共享 class registry 和单模型四任务评估入口。
5. `task_labels.py` 当前 T2=[0,1,2,3,4,5]（含 Bridge），T4=[7,9,10,11,14]（不含 Bridge），Tower 原始38–42 ignore；与 9/28 最新 T2 不含 Bridge、T4 含 Bridge 的决策不一致。旧基线 T4 是像素模型，新 HICD T4 是实例分支，不能仅迁移类别就宣称公平。
6. `train_fcef.evaluate()` 在 joint!=255 范围计算，把任务目标合为 object 并合并 ND/D 计数；`bdd_unified.bdd_hicd_style()` 用 raw GT 解码后逐目标调用 HICD evaluate_bdd_protocol，再对多类作均值。两者的 mask、类别混淆和汇总方式不同，T1 旧样本诊断的接近值不能推广至 T2–T4。summary 的 best_val_miou 实为 best_val_score 别名，不能当 joint mIoU；checkpoint 选择/恢复中的 mIoU 与 F_overall 混用也需专项核查。
7. `eval_fcef.py` 当前会把全 split 的 pred/gt/raw 留在内存并拼接；后续全量评估应先审查内存预算和流式累积方式。`bdd_unified.py`、train/eval 含固定 WSL 路径，不能未经环境检查直接作为 Windows 入口执行。
8. `data/dataset_task_0617.py` 从同一 names.txt 获取名字，但跨多个根目录/三个 split 索引，按同名文件首次命中选图像与标签；共享名单本身不证明共享文件内容、无泄漏或路径一致。本轮未扫数据和名单交集，交给 04 联合核查。
9. 外层 Git status 对 baselines_compare 显示 `?? baselines_compare/`；现存实现未被该仓库当前跟踪，不能用 HEAD 代替实验代码版本。本轮保留所有既有文件与改动。

#### 遗留问题、优先级与具体下一步

**P0：先形成可以执行的公平协议和实验清单。**

- 与 04 对齐目标/状态映射、Bridge 归属、ignore 与 FP 范围、类均值/全局累积、SeK 边界、split 交集和文件路径/内容；与 01/03 明确“像素 T1–T3 对比”及“T4 实例/系统对比”分别使用何种证据。冻结版本后，逐 run 记录名单及配置 hash、样本数、checkpoint hash/元数据、代码快照与实际运行参数。
- 优先追溯 EF T1 的 79,303 / 30,127 / missing4,597 来源；不能把它补成同划分数字。旧单任务结果作为诊断，主表等待重训/对齐。核查已完成 run 的 best/last 内部元数据，并制定旧 run test 补评方案；本轮没有启动补评。
- 核对 WSL 发行版、相关进程和共享 GPU 使用者后，决定 Siam T2 旧任务是否需要续训、T3/T4 是否仍有诊断价值；“等旧队列全部完成”不是已确认前提。旧 `retrain_fullstate.sh` 含 pkill 和归档移动，不能直接用作下一次启动入口。
- 准备 FC-EF-multi 最小改造方案：一个稳定的全类联合映射、一个数据集/模型/权重、统一验证和 test 的四任务输出，明确 pixel 与 instance 边界。验收关键项为跨任务标签 round-trip、跨类别误预测、其他目标上的 FP、replaced/ignore/空类、独立权重不存在；再做小样本前向与损失 smoke，不直接上 50 轮。

**P1：完成最小联合实现及基线实验。**

- 在当前用户下达实作任务并完成依赖核对后，先实现/验证 FC-EF-multi，再复用映射和评估到 FC-Siam-multi。历史 50 epoch / batch8 / ~4GB 为预算起点，需与 HICD 预算、预处理、增强、优化器和 best 选择条件明确对照。
- 修正验证口径与 best 选择记录、test 流式统计和元数据；全量训练前留独立 run 目录，避免覆盖旧结果。T4 F_clf=0 需要逐类支持/预测和指标边界诊断，不能单凭 aggregate 认定模型全部失效。

**P2：对照扩展与论文交付。**

- 向 05 交付有来源的单模型公平主表与限制说明，向 06 交付 run 台账、成功/缺失 test 产物和版本清单。SNUNet/STANet/BIT/ChangeFormer/ChangeMamba 扩展先审实现、许可证、任务适配和预算，本轮未验证或启动这些模型。

依赖分工：04 提供数据评估协议；01 提供 HICD 可比 checkpoint/训练范围及 GPU 协调；03 提供实例候选和 T4 评估边界；05 负责研究表述；06 负责台账/发布准备；公共记忆和 threads.json 由主会话维护。本轮未修改业务代码、协议、数据、实验日志或公共记忆，未训练/长评估/推送/创建自动任务。

### 2026-10-08（Asia/Shanghai）旧对照退出，新SCD模型T1–T3联合接入

**最新用户授权与范围：** 用户要求清除此前有问题的对照，引入语义变化检测模型进行单模型多任务联合训练；随后明确“现在T4不训练，只训练1-3”。此要求替代旧四任务像素计划和“等Siam队列全部完成”的前提。T1=Building，T2=Farmland/Runway/Taxiway/Apron/Highway，T3=Aircraft/Vessel；本轮不训练T4、不做实例基线。

#### 已执行：活动旧记录清理与保留

- 已将 baselines_compare 的 runs、results、runs_loc_only_legacy、runs_nd_only_legacy 同盘移入 `legacy/20261008_retired_per_task/`，并保存旧源码/脚本/说明及data代码快照。移动前manifest记录54个实验文件的路径、大小、mtime、SHA256，共282,970,217 bytes；没有删除权重或改写原实验文件。归档状态为retired_excluded_from_current_comparison，不宣称每个旧计算均已证明错误。
- 顶层旧shell入口以及train_fcef.py/eval_fcef.py/summarize_results.py明确拒绝执行，避免恢复旧队列或产生旧主表；原脚本保存在归档source。README/PROTOCOL/MULTITASK_PLAN已改为新T1–T3协议。新实验写runs_joint/results_joint，不能加载旧任务权重。

#### 已接入模型与统一入口

- Bi-SRNet：官方 https://github.com/DingLei14/Bi-SRNet ，commit `012f35aa9742e468b568a71107028fc3aa1e08b8`；SCanNet：官方 https://github.com/DingLei14/SCanNet ，commit `9c80d463bd0ca8cacca68b71c8b9adb7efd9f7a3`。已本地克隆并固定commit，源码保持checkout内容，私有命名空间装载；Windows/WSL的CRLF识别已通过本地core.autocrlf配置统一，不用文本差异误判架构修改。
- `scd_joint/models.py` 保留各自官方双时相/变化特征及SR/CotSR或CSWin时空交互，移除原两期语义/二值输出分类器，使用相同384→128融合和target16/state7双头。命名为HICD-T1T3-adapted，不能称为SECOND原论文复现；本数据只有target/state，不伪造pre/post语义图，不用无法监督的原语义一致性loss。
- ImageNet ResNet34骨干从PyTorch官方下载到本线weights，SHA256=`b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f`。任务头从头初始化，无旧FC-EF/Siam权重。与HICD DINOv3预训练不同，单独记录骨干/参数量和优化差异。
- `train_scd_joint.py` 支持CPU preflight和从头单模型联合训练；`eval_scd_joint.py` 独立严格test；`launch_scd_joint.sh` + `scd_joint/train_worker.sh` 提供screen/flock/资源拒绝与stderr/exit保留，无自动队列、监控或resume实现。每个run保存配置/源码、manifest、实际解析路径索引、每轮strict JSON、best/latest及optimizer/scheduler/AMP/RNG。

#### 与最新HICD参考对齐（本轮期间有跨线更新）

最初只读进程为旧v12/PID459，参考配置还包含Bridge。本会话据用户范围先局部排除Bridge，并完成早期CPU预检。随后只读启动保护发现01已经切换到 `v12_strict_joint_t123_trainfix_20261008` / PID1716；重读01最新记忆与manifest后，**最终基线已经改为该修复版本参考**。旧早期checks保留为superseded，不用于最终实验。

- 新参考run_identity=`fd1ae754c259bb7bd3b550c4cdc1b3ac04b9932fbe2a723c7a52459d206a59b0`；本线 `configs/t1_t2_t3_joint.yaml` 是其snapshot/0617final_t123_trainfix.yaml字节一致副本，SHA256=`c4beb1e57eaf3db996ee2ba0e37eb1341b6ffcf346a3d68d973bc3b8338e0ad9`。所有T4/Tower target/state GT均255，初始target权重0；只保留八个T1–T3前景目标与背景。16/7输出保留全局ID，排除类别误预测仍计FP，不用输出mask删错预测。
- 冻结复用该参考的dataset_v7及strict核心/包装器到scd_joint/frozen_hicd，source_lock及manifest验证hash。完整配置必须与参考解析一致，名单和scene_state_info/instances存在性也核对；不依赖旧task_labels静态LUT。正式入口核对train=61029、val/test=13418和实际解析数量，保留同样RGB[0,1]、256输入、增强与state-aware采样。
- 主协议hicd.strict_joint.v1，target与具体state全对才TP，replaced保留、none/背景FP计错，GT ignore统一；raw头64×64，GT nearest，流式计数，任务主表仅T1/T2/T3。best按val全前景精确联合micro最大；完整错误诊断不删无监督类别FP。旧ND/D/BDD不作为新主成绩。
- 基线loss明确为配置加权target CE + 2×state CE，保留静态Farmland/none/ignore训练选择，但strict不据此删评估像素；不复制HICD方法特有图、动态有效性、Dice/背景抑制/自适应loss。预算起点50ep/batch32/AdamW基础LR1e-4/weight_decay1e-4/cosine/AMP/workers4/state-aware；显式grad_accum变化会记录参考偏差。协议/范围对齐不等于模型、预训练或全部优化细节相同。

#### 核验与剩余工作

- 两个官方真实架构的CPU预检验证同一batch中的T1/T2/T3联合更新、骨干/目标头/状态头有限非零梯度、strict输出和checkpoint严格键往返，以及两个真实val样本；Bi-SRNet约23.42M参数、SCanNet约27.94M参数。最终参考版本产物为 `checks/bisrnet_t123_trainfix_20261008/` 和 `checks/scannet_t123_trainfix_20261008/`，以preflight.json为最终结果，诊断值不当模型成绩。早期`*_t1t3_20261008`是旧参考预检。
- 6项本线CPU验收通过：T4/Tower scope拒绝、训练ignore与strict范围独立、全ignore loss有限零且可反传、nearest/replaced、HICD活跃时拒绝GPU、整轮训练/尾累积/checkpoint/流式验证。脚本py_compile、两个shell语法检查通过；旧train_fcef实测拒绝执行。launcher实测在HICD活跃时拒绝，未创建新训练screen/模型进程。
- 真实train loader只读抽样记录61029/61029、state-aware6741加权、WeightedRandomSampler replacement=True，增强batch无T4 GT；来源 `checks/train_loader_20261008.json`。该早期loader检查使用初始本线scope配置，最终配置/源码以新版preflight及manifest为准。
- 当前GPU由训练线新T1–T3实验占用，未抢GPU、未停止它；**本轮没有启动完整50轮基线训练或全量test**，预检成功不等于模型已训练。待空闲资源后先检查显存/有效batch，明确启动一个方法、核验连续batch健康，再训练另一方法；不能承诺后台自动排队。
- P0剩余：正式GPU预算/早期健康、数据独立性审计的跨线结论；P1：两方法从头联合训练及同协议test；P2：适配MambaSCD并扩展对照。MambaSCD只为官方候选，尚未完成本轮接入，不能用旧smoke冒充新验收。
- 04既有split同名交集仍未消除，名单一致只能保证内部条件相同，论文独立泛化仍需路径/内容/场景核查。01负责参考HICD与资源；04负责冻结指标及数据；05/06接收主表与台账。本轮仅修改baselines_compare和自己的02记忆，不改公共记忆/注册表/主模型配置，未外发消息、推送或创建自动任务。

### 2026-10-09（Asia/Shanghai）首项Bi-SRNet对比已实际启动

**当前授权：** 用户要求“可以开始启动对比实验了，主实验已跑完，跑个对比实验看看效果”。本轮先执行一个Bi-SRNet-HICD-T1T3-adapted联合实验；T4继续不训练，SCanNet不并发启动。该一次任务包括50轮成功结束后的val-best独立test；不创建周期自动任务或等待其他模型队列。

#### 启动前核实与必要修复

- HICD参考 `v12_strict_joint_t123_trainfix_20261008/status.txt` 记录2026-10-09 11:32:30 +08:00 FINISHED、python_exit=0/tee_exit=0；日志明确50/50 TRAINING_COMPLETED，最佳val strict joint micro-F1=0.7190867247396852，best_epoch=7。epoch_007.json的val macro-F1=0.40242061194319434。这些为验证成绩，不是test。启动前无HICD/基线训练Python或screen，GPU compute列表空；F盘可用约70GiB，D盘约1.1TiB。
- 现有T1–T3配置、三个names、scene_state_info及冻结loader/strict代码与HICD参考manifest仍一致。正式run的reference_mismatches=[]；没有改主实验、原始数据、split或共享评估。
- 隔离CUDA实测首先暴露默认AMP scale65536导致首batch梯度溢出，而原clip_grad_norm(error_if_nonfinite=True)直接终止；失败保留在 `checks/bisrnet_gpu_20261009/failure.json`，没有正式权重或训练进程遗留。
- 本线修复：AMP初始scale1024，`scd_joint/common.py::step_optimizer`让GradScaler在unscale发现非有限梯度时跳过step并降scale，不在非有限梯度上裁剪；非AMP异常仍失败。每epoch记录实际optimizer_updates/amp_skipped_updates，连续10次无有效更新报错。协议/模型/loss/有效batch/预算不变，数值AMP策略写manifest。
- 新GPU预检 `checks/bisrnet_gpu_20261009_ampfix/gpu_preflight.json` 通过：2个真实增强batch32覆盖三任务和全部8类，每步target头实际更新、梯度有限，loss6.98949/6.78694；AdamW覆盖178个参数张量一次，无漏项/重复；2个真实val样本raw64 strict通路通过。峰值allocated4672MiB/reserved5016MiB，无OOM。smoke在独立进程，正式训练重新初始化、不复用smoke权重。
- CPU8项验收全部通过（旧6项+AMP溢出跳过/非AMP错误两项），py_compile及两shell bash -n通过。首次screen只留死socket未初始化，精确确认本次PID144145不存在后只清理该socket；launcher加入最长20秒等待python.pid/存活及Detached检查，避免仅凭派发宣称成功。再次启动实际成功；未全局wipe/重启WSL或停止其他任务。

#### 正式实验与已核实健康

- run：`baselines_compare/runs_joint/bisrnet_t1t3_20261009_144655/`；独立console为同级 `bisrnet_t1t3_20261009_144655.console.log`，阶段状态同级 `.status.txt`。2026-10-09 14:46:55 +08:00 worker STARTED。
- screen=`144474.scd_joint_bisrnet_t1t3`（Detached），Python PID=144481；run_identity=`2478e2272c1543b541da0933804f2d1436ed30c655e92a520fbe7bc54f45b546`。完整参数/source/上游模型源码/config/hash/实际路径索引/阶段状态已保存。
- 参数50epochs、batch32、img256→raw64、AdamW lr1e-4/weight_decay1e-4、cosine、AMP、workers4、state-aware。加载ImageNet ResNet34；任务权重/optimizer从零，日志FRESH_START epoch1/50 checkpoint=None。train/val解析61029/13418，6741样本按原state-aware规则加权。
- 2026-10-09 14:49:57 +08:00健康快照：epoch1至少201/1907batch；1/51/101/151/201批运行平均loss依次6.954260/5.213687/4.435332/3.990535/3.712935，连续推进、有限；fatal_patterns=[]，日志AMP overflow skip=0。进程/screen存活，GPU当次7443/24576MiB、利用率5%（仅瞬时快照，不能据低利用率认定停训）。证据run/early_health.json。
- launcher实测重复启动拒绝当前GPU PID144481，没有另起screen/覆盖run。初始健康证明训练通路正常，不证明方法效果；尚无epoch1完整验证或test成绩。

#### 效果产物与后续

- run/reference_validation_best.json冻结HICD第7轮最佳val、各任务/总体strict和source hash；每次完整验证会生成validation_comparison_latest.json，对照**val与val**，不混比HICD val和基线test。
- worker在训练Python和tee均exit0后自动调用eval_scd_joint.py，用本run best.pth独立评估test；预计产物 `results_joint/bisrnet_t1t3_20261009_144655_test/test_strict.json`。当前只是配置好的后续步骤，没有提前创建test成绩；若训练失败则不进入test，若test失败则保存阶段退出状态。
- experiments_joint.json已更新到running_early_health_verified，保存run/健康/PID/参考/自动test状态；README同步。下一次查询先核实实际PID、console/status、metrics/best/latest和test文件，再汇报结果。不得因启动通过就称完整对比已完成，也不默认启动SCanNet。
- 本轮仅修改本线入口/AMP实现/测试/文档及02记忆，没有改公共记忆/注册表/主实验，未推送、外发或安装周期监控。既有跨split重叠和不同骨干/loss的研究限制仍需随最终结果披露。


回复前复核：2026-10-09T14:52:58.109912+08:00，epoch1至少551/1907batch，运行平均loss=2.715406，fatal_patterns空、AMP overflow skip=0，PID仍存活；early_health_followup.json与前次201batch证据共同确认持续训练。

### 2026-10-09：完整精度日志/CSV补齐，不重启训练

用户要求“像我们主实验一样统计各种精度参数”，指出日志只有联合F1。已核实：原训练器的每轮metrics/epoch_NNN.json其实完整保存strict微平均P/R/F1、宏F1、TP/FP/FN、support、任务/目标/105联合类别、定位和GT前景区域状态诊断/混淆；console只打印一个总体F1，缺的是呈现。当前训练PID144481已进入epoch3，仍运行；epoch1/2验证文件完整。

- 已添加独立CPU报告器 `baselines_compare/report_scd_joint.py` 与 `launch_scd_reporting.sh`。回填epoch1/2并持续跟随本run的新验证，随着现有训练/test worker结束而退出；这是当前任务的日志呈现，不是周期自动任务或新增训练。GPU训练未被信号/重启，模型、优化器、原console、原JSON/权重均未改。为更新展示逻辑，仅精确重启过自己的CPU报告helper（旧PID148947→新PID149294）。
- 新优先日志：`runs_joint/bisrnet_t1t3_20261009_144655/train_detailed.log`，镜像原batch进度，在每轮验证后插入总体与T1/T2/T3 P/R/F1/macro、计数、8类联合/定位P/R/F1与定位IoU、GT前景状态P/R/F1、可评目标—具体状态行。console的tee打开方式不允许安全外部追加，所以单独镜像，不写原console以免被tee覆盖。
- `reporting/train_log.csv`每轮宽表含总体/任务/目标精度；`per_target.csv`15个目标（排除目标仅FP诊断）、`per_target_state.csv`完整105联合类、`state_on_gt_target.csv`15×7状态行、`joint_confusion.csv`非零混淆长表。`latest_validation_summary.json`与`provenance.json`保留全精度/每个源JSON SHA256。日志/CSV浮点显示4位，整数计数不四舍五入；N/A保持缺失，纯FP类为0并保留FP。
- 衍生diagnostics：同一strict GT-valid范围的联合/目标像素准确率（列名明确含背景）、前景/含背景目标mIoU、前景联合mIoU；GT前景全区域状态accuracy/mIoU，不按预测目标正确筛样本。通过联合/状态混淆计数重建，不重新推理。与旧loss反馈mask的state IoU不同，名称/有效范围明确，不引入旧ND/D/BDD/SeK冒充主成绩。
- **数据缺口明确保留：** 当前旧训练循环未累计train精度或validation loss，只有train loss和val精度；无法从loss或val推算train P/R/F1。本轮没有虚构训练分数、重跑GPU评估或为了日志重启epoch3；如果后续需要train精度，须在新的训练入口/明确续训方案中按batch累计，不能假装此run已经有该项。
- 现有test成功产物出现时，报告器会另导出reporting/test并在详细日志追加TEST块，不与val混合，也不把HICD val当test对照。
- 报告6项CPU测试通过：固定P/R/F1计数、105类与缺失值、同目标错状态与定位的差异、损坏计数拒绝、各任务/目标/状态日志、重复回填不重复且原JSON/console哈希不变、test/val标签分离。已复核原训练manifest所列所有.py的当前SHA256全部一致，自动test源码检查不受新报告器影响。

已结束第2轮 **val**：总体P/R/F1=0.4062066418/0.4375994525/0.4213190809，macroF1=0.1918776462；T1 P/R/F1=0.5130876901/0.5377870330/0.5251470997；T2=0.3528782459/0.3727024626/0.3625195370；T3=0.1534686397/0.5740143571/0.2421862859。来源epoch_002.json，早期验证观察，不代表最终效果或test。

README/experiments_joint.json已记录详细日志及CPUreporter状态；2026-10-09 15:41:52 +08:00快照报告PID149294运行、已导出2轮，训练PID144481仍存活。本轮未修改主实验、共享指标/原始数据/划分，未启动第二个模型、外发、推送或创建周期任务。

### 2026-10-09 15:52（Asia/Shanghai）：按用户要求终止，等待解决样本泄露

用户最新要求：“终止训练吧，我先把样本泄露问题解决掉然后重新进行主实验”。已实际终止本会话的Bi-SRNet对比及后续自动test，停止详细报告helper；不是继续后台等待或自动恢复。

- 先核实命令行和run路径，确认训练PID144481及worker144476属于本run的独立PGID144476；只对该组发送SIGTERM（包含tee与4个明确DataLoader子进程），取消训练成功后的自动test。独立报告PID149294也精确核实后停止；关闭本次screen。未重启WSL、未处理其他项目进程。
- 2026-10-09T15:52:54.354231+08:00停训核查：目标进程剩余=[]，之后再次ps未见本线train/eval/report/worker，screen无socket，GPU compute进程列表空。GPU剩余约2149MiB为查询快照，不能宣称整卡显存全部清空。
- 实际已保存epoch1/2/3完整验证，终止前日志进入epoch4的batch1；best.pth/latest.pth、三个epoch JSON、配置/源码快照、路径索引和原/详细日志全部保留。没有成功test_strict.json。
- run/termination_requested.json与termination.json保存精准进程组/命令/成员及停训证据；run/status.json和同级.status.txt明确user terminated、sample leakage、automatic test cancelled。experiments_joint.json同步terminated_by_user_sample_leakage，SCanNet保持未启动且用户要求暂缓。
- 现有旧划分结果仅作排查记录，不进入修复泄露后的正式主表。下一步由用户先处理泄露、重跑主实验；未来基线必须核对新划分/名单与哈希、新HICD参考及配置，从头训练，不用本run旧权重续训充当新数据结果。本会话不自动启动下一项，不改用户数据/划分或公共记忆。

### 2026-10-11：恢复划分后Bi-SRNet从头100轮对照已启动

**最新用户授权与选择：** 用户告知样本泄露已解决、主实验已完成，要求准备重新训练对比方法。随后对可选范围明确回复“先跑Bi-SRNet，看结果后再跑第二项”。本轮只启动Bi-SRNet；SCanNet已经完成新版CPU接入检查，但不启动、不安装自动第二项队列。

#### 新参考、数据和语义核验

- 已读01关于原名单恢复、新catalog loader、100轮完成的最新记录；参考为 `v12_strict_joint_t123_disjoint_e100_20261009`，run_identity=d1584ee10d43b2df6e2b2e6922535cb2451a9f44941bedd8bcb23dba4ea93d75。其status确认2026-10-11 06:13:10 +08:00正常完成100轮；val主micro-best90=.895640499834862、macro=.5782007805694767，独立macro-best56=.6226105183775097。
- 新root `/mnt/d/CD/0617final_256_v2_disjoint_20261009`，train61029/val13077/test13077；split identity=9f66845372df4c8730a543ac2899575b8ba40eff31b676355b183991e09db05b。本线独立重算三对split的名单、末尾_NN_NN母样本键、pinned pre源路径交集均0；三个names/catalog、split_manifest、scene metadata的SHA与新主实验manifest一致。缺失/变动catalog启动即拒绝，不fallback到原目录。
- 该修复覆盖用户所指的共享名单/母样本切片问题，是恢复原划分；按发布协议没有内容hash去重、城市留出或地理footprint审计，不把0交集扩大称所有因素完全独立。没有改数据或旧名单备份。
- 基线配置使用新主snapshot字节一致副本，SHA256=80798b84930ae191ff3dfb0c89aa1e316ace193652dfb331c6bbc58c683b17f0。冻结复用其dataset_v7、strict核心/包装器/报告格式、farmland_semantics五项源码并核hash；只T1–T3，T4/Tower ignore。
- 预测语义对齐hicd.predicted_farmland_nochange.v1：按预测target=Farm固定pred state0，不用GT决定；其他目标的damaged/reduced/added/extended/replaced/none保持具体状态strict计数。使用主训练相同的recompute_report做精确计数变换，保留raw支路/nearest64网格与预测policy metadata，不声称Farm学会多状态损毁。

#### 本线入口更新与验收

- 默认reference/data root改为新版、budget100/batch32/AdamW基础LR1e-4/weight_decay1e-4/cosine T_max100/AMP/workers4/state-aware。加载正常ImageNet ResNet34，任务头/optimizer从零；旧10/09中止run不续训，不加载其任务权重。模型/标准加权CE与HICD骨干/方法loss的差异仍明确记录。
- 这次同时按batch累计**train完整strict**与val完整strict/val loss，train反映增强+state-aware+drop_last的61024样本/轮，val完整13077。每轮JSON含train/validation/计数/精度、targetCE/stateCE、更新/AMP跳过次数；console直接输出完整类别、任务、具体状态明细，CSV分train/val，详细报告helper自动接入。修复了上轮“没有train精度只能呈现val”的缺口，不回填虚假旧train精度。
- 主best按val micro，另保存best_macro，latest每轮、编号每10轮。训练成功后独立test主micro-best，不按test择优。已冻结新主实验val-best与同轮test作为同split比较来源（source hash/身份/样本数/语义验证），后续生成test_comparison.json；不混比val/test。
- 19项CPU验收通过：原训练/AMP/报告测试，以及catalog身份/完整性、缺失catalog不fallback、不同切片同母样本拒绝、catalog变动拒绝、预测侧Farm规则/其他状态严格计错、train完整precision和val loss/macro checkpoint/CSV实存等。证据checks/acceptance_disjoint_20261011.log；py_compile及两shell语法通过。
- 两个实际架构在新版root的CPU synthetic更新+2真实val前向/strict/strict键往返均通过，来源checks/bisrnet_disjoint_e100_cpu_20261011与scannet_disjoint_e100_cpu_20261011；这些是管线诊断，不是模型成绩。

#### GPU等待与正式启动

- 初始发现01的主best90完整test正在运行（PID365143及workers），没有抢占/停止；本轮先CPU准备。它已正常完成13077 test并退出，FULL_TEST_COMPLETE micro=.8999531626159337、macro=.622191871766802，来源主run/test_epoch90_20261011/test_metrics.json。本线仅读取并冻结参考，未操作主test。旧公共记忆中“未test”等历史状态不覆盖该新文件事实。
- 新版Bi-SRNet独立GPU预检通过2真实batch32与2真实val，178个optimizer参数张量覆盖一次、有限梯度、每步目标头实际更新；两步loss6.94748/6.75549，peak allocated4672MiB/reserved5016MiB，Farm预测metadata正确。证据checks/bisrnet_disjoint_e100_gpu_20261011/gpu_preflight.json；smoke任务权重不复用到正式run。
- 正式run：`runs_joint/bisrnet_t1t3_20261011_164416/`，2026-10-11 16:44:16 +08:00 worker STARTED。run_identity=049d2290fd21b7dd114a8d41f80a4ac3d1647fbcddafcac11c3d1c11ebbeec82。screen366716.scd_joint_bisrnet_t1t3（Detached）、Python366723；详细报告screen366806.scd_metrics_bisrnet_t1t3_20261011_164416。日志明确FRESH_START epoch1/100 checkpointNone。
- 2026-10-11 16:46:53 +08:00健康：epoch1至少151/1907batch，1/51/101/151批运行平均loss6.954261/5.219154/4.434846/3.988344，连续推进且有限；fatal_patterns=[]、AMP skip=0，GPU7605/24576MiB瞬时快照。主进程和两screen存活，reference_mismatches=[]，train/val解析61029/13077，旧任务权重None。证据run/early_health.json。尚无完整epoch验证/test，不能宣称新基线效果。

#### 产物和后续

优先日志run/train_detailed.log（同级console保留）；metrics/epoch_NNN.json、reporting/train_log.csv及reporting/train分离精度；reference_validation_best.json、reference_test_micro_best.json；manifest与data_snapshot包含三个names/catalog/identity，源码和上游实现也留快照；best/latest/best_macro/编号权重按轮结束生成。正式test预计输出results_joint/bisrnet_t1t3_20261011_164416_test/test_strict.json和test_comparison.json。

experiments_joint.json已保留旧run到past_runs并解除当前用户hold，更新100轮/数据/语义/状态；README优先段记录新协议。下一步先验收完整epoch1的train/val计数、精度、best/latest身份，待Bi-SRNet结果后再按用户决定第二项。不自动开启SCanNet、创建周期监控/推送/外发；本轮没有改主实验/公共记忆/注册表/数据。

回复前复核：2026-10-11T16:48:47.362563+08:00，epoch1至少351/1907batch，运行平均loss=3.133546，fatal空、AMP skip=0，训练PID仍存活；early_health_followup.json与前次151batch观察共同证明持续推进。用户明确先Bi-SRNet后看结果再决定第二项，状态已保存。

### 2026-10-12 09:00（Asia/Shanghai）：用户查询当前指标，45轮完整验证

- 只读检查确认同一run/PID366723、训练与报告screen均存活；09:00:53快照正在epoch46/100、至少1001/1907batch，运行平均loss=.113876，console无指定fatal标记。没有停止/重启训练、启动第二项或test。
- 已完成epoch1–45，各有完整train/val统计。第45轮同时为目前micro-best和macro-best：val P=.8786215341、R=.8966091891、micro-F1=.8875242310、macro-F1=.4532169356；train（增强+state-aware采样）P=.8865397854、R=.9229868714、micro=.9043962738、macro=.6006777427。train loss=.1161736328，val loss=.8286829389；本轮1907次optimizer更新、AMP skip0。
- 第45轮val T1 P/R/F1=.8161402761/.8170456312/.8165927027；T2=.9224306889/.9484364982/.9352528479；T3=.4742682167/.7026189641/.5662902078。train任务micro分别.8586302408/.9530607815/.8039445100；T3 train/val差距较大，但训练来自增强/加权分布，不直接等同独立训练集推理泛化估计。
- 目标strict F1：Farmland.9483231974、Runway.9725301281、Taxiway.9061311720、Apron.8608182762、Highway.7091824857、Building.8165927027、Aircraft.6401844026、Vessel.5336151013。Aircraft replaced P/R/F1=.2608695652/.0259365994/.0471821756（GT694px）；Vessel replaced=.3647762346/.3174416653/.3394668342（GT5957px），细状态仍弱。
- 对照主HICD的val micro-best90=.8956404998、macro=.5782007806，基线当前micro低.8116269个百分点；T1/T2/T3分别低约.8743/.4317/8.1535个百分点。此为进行中的val对照，不与HICD test=.899953混比，也不是100轮最终结论。
- Macro基数核实：双方val有GT组合均26，基线可评组合35（9个仅预测出现类）、主HICD90可评29（3个仅预测出现类）。既定macro包含仅FP组合的0分；不删这些类或改宏定义来抬分。参考epoch90源SHA仍与冻结文件一致。
- 45轮共90份train/val报告的micro均由TP/FP/FN重算精确一致，预测语义版本匹配；train每轮61024、val13077完整覆盖。当前基线test未产生。统计快照保存在run/reviews/status_20261012.json（含全指标、参考、源SHA、覆盖和进程状态）；experiments_joint.json更新进度/45轮最佳指针，原指标/权重未改。

### 2026-10-12 09:11：收敛状态检查

用户问Bi-SRNet是否收敛。只读分析完整epoch1–45，训练PID仍在第46轮（最后训练进度1901/1907，可能处于随后验证）；未修改/停止训练、早停或调度器。

- 当前判断：**仍在缓慢改善，未形成稳定验证平台**。最近36–45轮micro从.8786447515升至.8875242310（+.8879479个百分点），线性趋势约+.1312个百分点/轮，最佳在36/38/39/42/43/45共6次更新。最新45也为micro和macro最佳，不是长期不再改善。
- 同窗train loss从.1385474降至.1161736；val loss在.7959784–.8882899波动，没有持续下降。macro在.4142707–.4532169波动；不能只看训练loss下降判定泛化已收敛，也不保证后半程收益。
- T1/T2同窗F1范围.79495–.81659/.92231–.93525，仍慢增；T3 .55074–.60073，较明显波动，45轮.56629低于43轮.60073，小类仍不稳定。按既定cosine100配置当前LR约5.82435e-5，后续低LR段尚未完成。
- 10/20/30/40/45验证micro=.8107008/.8529598/.8665421/.8804336/.8875242，macro=.3158860/.3966094/.4061055/.4441899/.4532169。建议继续已授权100轮预算，以保持与主实验预算一致；本轮只是分析，没有新增早停规则或第二项队列。
- 审计保存run/reviews/convergence_20261012.json（全部45份源SHA、窗口指标、最佳更新、时间/进度）；experiments_joint.json更新收敛评估，原指标/权重/进程不改。

### 2026-10-12：Bi-SRNet规模与方法地位说明

用户询问模型大小以及是否当前SCD强方法。本轮CPU重新实例化实际适配版（不加载任务checkpoint、不使用GPU），核实总/可训练参数均23,417,786=23.42M；features23,365,411、fusion49,408、target head2,064、state head903。FP32纯参数93.671144MB（89.331764MiB），不含buffer、optimizer和激活。当前HICD新主train.log第11行记录Total params95.59M，因此Bi-SRNet总参数约其24.5%；不以总参数代替计算量/部署内存或比较可训练部分。

- 原论文Bi-Temporal Semantic Reasoning for the Semantic Change Detection in HR Remote Sensing Images，arXiv2108.06103、TGRS2022，作者repo DingLei14/Bi-SRNet；共享ResNet34与单/跨时相语义推理构成经典CNN SCD基线。本地保留其双时相交互/变化特征、适配为目标16/状态7；不是SECOND原论文复现，原语义一致性loss没有伪造监督。
- 主源证据：https://arxiv.org/html/2404.03425v3 （ChangeMamba，2024），TableVII SECOND报告Bi-SRNet F_scd62.61/SeK23.22，SCanNet63.66/23.94，MambaSCD-Small64.10/24.04，Base64.03/24.11；TableVIII Bi-SRNet23.39M参数，与本线实测适配规模接近。它说明有竞争力且后来方法在该表超过它，不是截至2026统一全领域排名；不能把SECOND F_scd与本项目strict joint F1横比。
- 原架构说明：https://arxiv.org/abs/2108.06103 。SCanNet作者研究：https://arxiv.org/html/2212.05245v4 / https://github.com/DingLei14/SCanNet （TGRS2024）。2026仍有新SCD研究，例如https://arxiv.org/abs/2607.19000 （SCDistill预印本）；本轮未全面审计2026方法统一榜单，不称Bi-SRNet为当今最强或给top-N结论。
- 当前最新完整50轮验证micro=.8929072380、macro=.4657107017，T1=.8248324239/T2=.9389678485/T3=.6174862878；对照HICD micro-best90=.8956404998、macro=.5782007806、T3=.6478249665。说明此适配基线在本任务已很有竞争力，但只是进行中val、尚未最终test；整体接近不能替代细类/效率/同条件消融证据，也不证明原论文在本任务直接获得此分数。

本轮只查本地模型和公开主源资料，未改训练/评估/权重、启动额外GPU模型或第二基线。下一步仍按用户选择完成Bi-SRNet并看结果后再决定第二项。
