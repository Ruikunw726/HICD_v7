# 01 训练与方法接管记忆

建立：2026-10-02。负责人：HICD 项目“训练恢复与方法改进”会话（ID 见 threads.json）。
来源：HANDOVER.md 第 2–7、9、13 节；experiment_log.md；launch_v11.sh；project_context.md。

## 接管范围

- P0：WSL/Python/CUDA/GPU/screen 状态，v11_no_t4 第 32/50 轮之后现状，检查点完整性与真实 resume 路径。
- P0：与数据评估线对齐 T4 ignore、Bridge、T3 保留及实际被加载的 YAML。
- P1：T1–T3 结果、F_clf 瓶颈、GNN 标签结构与 BackgroundSuppressionLoss；提出可辨识消融及方法改进。
- 实验验证后再更新训练台账，保留配置、数据名单、seed、日志与checkpoint关系。

## 关键路径

`models/HICD_v7.py`、`models/backbone/dinov3.py`、`models/losses/v7_loss.py`、`models/modules/label_structure.py`、`script/train_full_v7.py`、`launch_v11.sh`、`checkpoints/v11_no_t4/`，均相对 HICD_v7。

## 初始状态

已读日志最后 Epoch 32/50，F_overall 0.7668；不能当作 test。启动脚本会杀同名 screen，且要的 latest checkpoint 不存在于已查看目录。当前训练进程尚未核查。历史 v8/v10 数字见公共背景。

## 本轮接手输出

有限只读检查运行条件和配置；把新增核实写在本文件下方并向用户回复接手清单，标出可以接续的步骤与真实阻塞。本轮不启动/终止训练、不修改模型和共享配置。

## 依赖与边界

共享数据、标签与协议由 04 主责；GPU 排队需与 02/03 核对；结论交 05 论证；实验变更交 06 记录。旧材料不足以证明 SOTA 可比。

## 接手核查记录

### 2026-10-02（Asia/Shanghai）首次接手完成

会话：`01a0faa7-3a21-7672-b42c-b85f7c37947b`，标题“HICD｜训练恢复与方法改进”；已读取 AGENTS.md、README、project_context、decisions、risks_and_questions、threads.json 与本文件。来源交接截至 2026-09-28，以下为本轮只读核实，不能把历史中断原因当成已诊断故障。

#### 已核实：运行环境与检查点

- `wsl.exe --list --verbose` 显示默认 `Ubuntu-26.04` 为 Stopped（WSL2），`docker-desktop` 也为 Stopped。本次 Windows `Get-Process` 未列出 wsl/wslhost/python/pythonw；默认发行版内当前不可能维持活跃训练。未启动 WSL，因此 `/root/miniconda/envs/mamba/bin/python`、CUDA 可用性、依赖版本与 screen socket 留存情况仍待核实；不能由 Stopped 推断故障原因。
- Windows `nvidia-smi` 能访问 NVIDIA TITAN RTX，显存总量 24576 MiB、当时已用 1795 MiB、GPU 利用率 17%；这些只是单次快照，不代表 GPU 空闲或 WSL CUDA 已可用。列表中无法识别名称的 PID 44948 经 Get-Process 确认为 dwm；所见列表没有训练 Python，但启动前仍要再次核对跨线占用。
- `checkpoints/v11_no_t4/train.log` 与 CSV 都结束在 epoch 32；日志标注 F_overall=0.7668、F_loc=0.8188、F_clf=0.7446、SeK=0.1191。CSV 该行为 val_loss=1.9522，属于训练期间验证记录，不能当作 test。CSV T1_f1=0.7668、T2_f1=0.4384、T3_f1=0.5256、T4_f1=0；任务定义待 04 确认。
- 目录最高已见编号是 `checkpoint_epoch_32.pth`，745160462 bytes，文件显示最后修改为 2026-09-29 08:52:26；`best_model.pth` 为 745144846 bytes，最后修改 08:52:32；latest 文件不存在。Windows 文件时间为本次目录显示值。
- 使用 Windows `F:/python/python.exe` 的标准库 zipfile/pickletools，只读取 epoch_32 的 ZIP 目录和 data.pkl 指令，不执行 pickle、不导入 torch、不加载张量：1728 个归档项、1724 个 tensor storage 项；元数据含 epoch/model/optimizer/scheduler/scaler/loss/args；epoch=31，loss=1.9521795181734931，args 含 dataset=0617final、config_dir=datasets/configs、backbone_type=dinov3、max_epochs=50、use_amp=True、exp_name=v11_no_t4、resume=None。仅证明归档目录和元数据可读，未验证全部 CRC、张量完整性或当前模型兼容性；best 尚未做元数据读取。

#### 已核实：启动与恢复语义

- `launch_v11.sh` 第 9 行先 `screen -S ... -X quit`，会终止同名会话；只检测 `checkpoint_epoch_latest.pth`，当前无该文件会不带 --resume 启动；stderr 被 `2>/dev/null` 丢弃。因此不应直接执行旧脚本恢复。
- 脚本使用 `--dataset 0617final --config_dir datasets/configs`；`script/train_full_v7.py` 第 87 行按 dataset 名加载 YAML，实际入口为 `0617final.yaml`，并非 `0617final_v11.yaml`。train.log 开头记录的配置也保留 Bridge 18–22→4、Aircraft 57–61→12、Vessel 62–66→13；其他 T4 对应 target 多为 255。当前两个 YAML 均保留 Bridge，当前配置仍有 16 个目标通道和相应类名/权重，不能称已完整移除全部 T4。
- 当前 `_load_checkpoint` 恢复 model（strict=False）、optimizer、scheduler、AMP scaler，并令 current_epoch=checkpoint epoch+1；epoch_32 对应 current_epoch=32，循环应继续显示第 33/50 轮。缺失 resume 文件时构造器静默跳过加载；启动前应强制验证精确路径。
- best 按最低 val_loss 保存，不按最高 F_overall 选择；选 test checkpoint 时需明确选择准则。未验证 best 与 epoch_32 张量相同。
- `AdaptiveClassWeighter` 是普通 Python 类，其 running_iou/weights 未注册为模型 buffer；checkpoint 未显式保存它们、current_target_miou 或 RNG/sampler 状态。resume 后 `train()` 会执行 set_phase(2)，该函数重置自适应 running_iou 和权重。可恢复主要优化状态不等于严格连续复现；strict=False 也不能证明不存在 missing/unexpected keys。代码/数据名单/YAML 的完整快照未见保存于 checkpoint 字段，args 只保留路径与参数。

#### 方法线观察与待核实冲突

- 当前 `models/HICD_v7.py` 明确导入 `GNNLabelStructure as LabelStructure`；频率分支在日志及 YAML 中启用；`v7_loss.py` 有 BackgroundSuppressionLoss、自适应类别权重和 state loss。代码存在不等于已证明增益。
- R01 更新：旧交接“9/28 中断”已被 9/29 第 32 轮文件取代；当前 WSL 已停止，真正中断时间/原因未诊断。
- R02 更新：危险 screen 退出、latest 错配、缺失 resume 静默回退、错误输出丢失均由当前脚本/训练器核实；修复尚未实施。
- R03 更新：历史运行日志与当前 YAML 都保留 Bridge；需 04 对齐 target/state ignore、任务分组和评估 mask。若修改 Bridge ignore，应另建新实验，不能在同名 v11 后 18 轮默默改协议。
- 新评估冲突：CSV epoch 32 的 `F_overall=0.7668` 与 `T1_f1=0.7668`，但 `bdd_f1=0.4327`；日志把 `bdd_sek=0.1191` 写成 SeK。暂不把日志 F_overall 称为所有任务全局汇总，交 04 审查汇总范围/字段来源。
- 新日志冲突：epoch 32 的 train.log 验证损失详情中 loss_contrastive=0，而 CSV 同字段为 0.069116；需要确认 train/val 字段口径，不能据一个字段断言 GNN 没训练或增益有效。
- F_clf=0.7446 低于 F_loc=0.8188 是该验证行现象；原因需拆解类别混淆、状态混淆及 mask，不能直接归因于 GNN 或类别不平衡。

#### 优先级与具体下一步（未执行）

1. **P0：安全恢复前置。** 后续收到训练实作任务后，启动/进入 WSL 前与 02/03 核对 GPU；检查发行版、screen、Python/CUDA、挂载、预训练权重和剩余磁盘。对 epoch_32 做 CPU 完整加载与关键状态检查，核对 checkpoint 与当前工作区实现差异，显式报告 missing/unexpected keys，确定恢复路径。
2. **P0：保护既有实验。** 准备不会退出已有 screen、resume 不存在即失败、保留 stderr 的恢复入口；记录自适应权重/RNG 无法完全恢复的限制。冻结原 v11 实际配置、名单、代码摘要和日志，延续原协议才称续训；新 T4/Bridge 策略另起实验名并保留原结果。
3. **P0：协议对齐。** 与 04 明确 Bridge/T4 target/state mask、T3 保留、BDD/T1–T3 汇总、验证与 test、best 选择规则；当前日志 F_overall 与 bdd_f1 差异必须解释后才能对外比较。
4. **P1：F_clf 诊断与消融。** 固定数据/评估/训练预算，先分解 target 与 state 混淆；分别评估 GNN、背景抑制、自适应权重、频率分支的独立效果，避免同时改多个因素；预先记录开关、seed、完整配置、指标和 checkpoint。消融方案待协议确定，不本轮占 GPU。
5. **P2：可复现交付。** 实验执行后输出配置与数据名单摘要、代码版本及工作区差异、恢复来源、训练/验证日志、test 结果与选择准则；交 06 入台账，交 05 判断论文表述，不沿用未对齐的 SOTA 数字。

#### 协作与本轮边界

- 04 数据评估主责共享 decode/state mask/split/指标；01 提供实际加载配置、日志字段和训练器恢复证据。
- 02 多任务基线与 03 实例分支共同核对 GPU 排队；03 需解决像素忽略 T4 后实例候选来源（R09），01 不以 GT 框分类替代端到端证据。
- 05 论文只接收证据等级清楚的结果；06 管实验台账与版本发布准备；主会话维护注册表/公共记忆。
- 本轮只修改此工作流文件，没有更改模型/配置/数据/实验日志，没有启动或停止训练、启动/重启 WSL、执行模型评估、推送或创建排程。接手完成，训练恢复和消融尚待后续实作。

### 2026-10-08（Asia/Shanghai）已授权续训，启动与早期健康验证完成

当前授权：用户在总览会话明确要求 v11 继续训练；本轮据此实际执行，取代 10/02 接手轮的只读边界。仅延续原 v11，不改标签、模型、损失、评估或采样策略，不停止其他任务。

- 启动前：Ubuntu-26.04 已 Running，无现存训练 Python 或 screen；只有 VSCode Python 环境/语言服务进程。CUDA smoke 通过，torch 2.6.0+cu124，TITAN RTX；F 盘可用约 76 GiB，D 盘约 1.1 TiB，WSL 根目录约 920 GiB。最新检查点仍为 epoch32，日志/CSV 最后也为 32。
- 原协议保持：`0617final.yaml` 的解析结构与历史 train.log 配置完全相等；Bridge 18–22→4 保留，T3/其余 ignore 不改；使用 checkpoint 内完整 argparse 参数（batch32、img256、50轮、AMP、workers4、state_aware、freeze_backbone_epochs=0 等），只替换明确 resume 路径。实载 train=61029、val=13418，state-aware 加权=6741，与历史日志一致；test 名单=13418。
- 新入口：`launch_v11_resume.sh` + `script/resume_v11.py`；旧 `launch_v11.sh` 通过 exec 转接到新入口，旧正文仅作为不可达历史保留。入口拒绝已有训练/session、使用 launch/runtime flock，不退出任何已有 session；不存在 checkpoint 即失败；stdout/stderr 同时保留到主日志与独立续训日志。已通过 bash -n、py_compile、缺失检查点拒绝与实际运行期间重复启动拒绝验证。
- 完整预检：epoch32 在 CPU 全量 torch.load，模型与 optimizer 张量有限；按已有 checkpoint 初始化唯一动态 buffer `loss_fn.state_ce.weight` 后 strict=True 模型加载成功，其余 missing/unexpected 不放行。AdamW 有 349 项 state、5 个参数组；scheduler last_epoch=32；GradScaler scale=32768、growth_tracker=782；加载后实际训练器再次核验调度器、LR、AMP 和 optimizer 项数，日志出现 `RESUME_OK epoch=33/50 strict_keys=True`。历史最低 val_loss/恢复 best_loss=1.9521795181734931。
- 首次 screen fork 后仅留下死 socket、worker 未初始化，未进入训练；只清理自己的 `571220.hicd_v11_resume` 死 socket后重试，无全局 WSL 重启/进程清理。稳定 session 为 `571355.hicd_v11_resume`，worker shell PID=571358，训练主 PID=571365。11:05:07 +08:00 启动；11:07:49 +08:00 检查到第33/50轮至少41/1907 batch，运行平均 loss=3.0771，多个连续batch推进，未见 Traceback/OOM/nan/inf，screen Detached、主进程存活、CUDA 显存约12.4 GiB。
- 之后给 launcher 增加等待 worker status 文件及 Detached session 的初始化保护，防止只凭 socket 宣布启动。修改发生于训练进程完成输入 hash 验证后，仅改 launcher 外层启动检查；初始源码由审计目录 source 保存，最终 launcher 另存 `launcher_with_initialization_guard.sh`。当前训练器/模型/配置文件未改。再次恢复必须新建预检快照，不能绕过输入哈希复用旧 manifest。

审计目录：`checkpoints/v11_no_t4/resume_20261008_110218/`。保存原 train.log、train_log.csv、best_model、epoch32（权重各约745MB）、原配置及三份 names、35项有效代码/配置/名单/预训练权重 SHA256、源码拷贝、Git HEAD/status/diff、实际参数、精确命令、PID、状态及 `early_health.json`。HEAD=`2834e232512b4bb6e2df2a7716cb0ae9d41e5304`，既有未提交/删除项保留；HEAD不代表全部有效实现。另存 state-aware 的 scene_state_info.json 及其 SHA256；当前数据根没有本训练器读取的 instances.json，未据此改变实例行为。

精确 Python 命令：
`/root/miniconda/envs/mamba/bin/python -u script/resume_v11.py --mode run --checkpoint /mnt/f/mambacd/home/HICD_v7/checkpoints/v11_no_t4/checkpoint_epoch_32.pth --audit-dir /mnt/f/mambacd/home/HICD_v7/checkpoints/v11_no_t4/resume_20261008_110218`

查看：WSL 内 `screen -r hicd_v11_resume`（Ctrl-A D 脱离）或 `tail -f checkpoints/v11_no_t4/resume_20261008_110218/train_resume.log`。status.txt 记录启动和最终 Python/tee exit；主 train.log、train_log.csv 继续追加；CSV 在完整轮次验证结束后才新增。当前仅证实早期健康运行，未完成33轮验证/50轮训练，未新增 test 结果。

限制仍有效：旧 checkpoint 不含 RNG/sampler/自适应 running_iou/完整权重历史，原 train() 的 phase2 reset 保留；不能声称严格连续复现。当前配置/名单/代码哈希已记录，但9月训练没有同等历史哈希，未逐个验证全部图像内容未变。R03、BDD/T1汇总、loss_contrastive字段及论文可比性问题不在本次续训中修改。

下一步：用户后续查询时先读独立日志、status、CSV及新checkpoint，确认是否完成50轮；若失败，依据实际异常和最新完整检查点处理，勿直接重复启动。协议修正与消融仍按P0/P1另建实验；没有创建自动任务或承诺后台通知。

回复前复核：2026-10-08 11:09:30 +08:00，已到第33轮至少96/1907 batch，运行平均loss=2.9541，有限loss且无上述致命异常；补充证据存 `early_health_followup.json`。与11:07:49的41 batch记录一起证明持续推进。

### 2026-10-08（Asia/Shanghai）最新授权：停止沿用v11，严格联合协议新实验已启动

用户最新要求：“目前的v11训练已经终止，你直接按照最新的标准重新训练”。此要求替代上方v11续训计划；当前任务为新实验 `v12_strict_joint_20261008`，不再恢复v11。启动前核查没有旧v11训练/其他训练Python或screen；没有执行结束无关进程、WSL全局重启或清理旧实验。v11日志、CSV、权重和既有审计目录保持原样。

#### 新标准与不变范围

- 公共实现由04交付：`evaluation/strict_joint_protocol.py`、`evaluation/strict_joint_evaluator.py`，版本 `hicd.strict_joint.v1`，协议见 `docs/strict_joint_protocol.md`。主类别P/R/F1要求target和原始具体state同时正确；不做D/ND合并，不用旧F_overall加权。pred_none、错目标、错状态和有效背景上的前景FP都计错；统一GT ignore。真正无GT且无预测项为N/A（JSON null/CSV空），只有FP的项仍为0并保留FP计数。
- 训练和验证都明确调用04的流式包装器。使用raw `state_logits`，沿用输入256×256、GT nearest scale0.25→64×64输出网格，不改refined支路或评估分辨率。
- 报告包括全前景/任务 strict joint micro和macro、每个目标strict P/R/F1/TP/FP/FN/support、定位诊断，以及有效GT目标区域内不筛pred_target正确的具体状态诊断、105个target-state行、联合混淆计数。完整每轮数据写 `metrics/epoch_NNN.json`；CSV主列全部显式strict字段，无旧T1_f1/F_overall混用。
- best准则：验证集**全前景 strict_joint_f1 micro**最大；macro一并报告，平分不替换先前best。每轮通过临时文件+fsync+rename更新latest.pth，保存best及每10轮编号；checkpoint含protocol/args/config/run_identity、optimizer/scheduler/AMP、RNG、自适应权重/running_iou/target_miou等。没有宣称保存dataloader worker RNG或批次中途状态；本次入口明确拒绝resume，后续恢复须匹配新协议和实验身份。
- 训练预算、模型/损失/采样/标签均沿用原v11：50轮、batch32、img256、LR1e-4、AMP、workers4、state_aware、freeze0。新参数与v11审计manifest逐项比对，仅exp_name/resume不同；正常加载DINOv3 pretrained backbone，其余任务网络、AdamW、scheduler、GradScaler均新建，epoch=0，没有读取旧任务权重。
- 当前配置允许GT的前景类别：Farmland/Runway/Taxiway/Apron/Bridge/Highway/Building/Aircraft/Vessel。T4有效GT范围只有Bridge；Shelter/Pier/Dock/Tank/Crater以及Tower的GT映射仍ignore。它们在有效其他区域的预测FP仍计错；不能把该训练报告称为完整T4六类或实例检测成绩。当前split及既有独立性问题未改变。
- 原loss的自适应IoU反馈独立保留原mask和union规则：target GT!=255，state排除GT ignore及配置state_ignore_targets；只用于loss反馈/既有refinement warmup，不作为strict分数或best依据。小数组与旧ICDEvaluatorV6._compute_iou对照一致。

#### 已完成验证与实际进程

- bash -n和py_compile通过；04核心+包装器27项测试在WSL mamba Python全部通过无skip，涵盖背景FP、pred_none、replaced、ignore、错目标/错状态、无支持项、raw/refined选择、nearest、stream/reset/merge等；独立只读审查由总览协调完成。
- 独立GPU smoke进程执行2个合成训练batch，以及**一个真实val batch（2张样本）**：旧_prepare_targets、实际HICD前向、strict update/compute、format、原IoU权重反馈、JSON序列化均通过，记录 `preflight_smoke.json`；train_updates=2、val_updates=1、105条target-state、64×64/raw元数据。仅管线验收，不作模型成绩；正式进程重新创建全部任务/优化器状态，未复用smoke模型。
- 正式启动：2026-10-08 16:04:47 +08:00，screen `452.hicd_v12_strict_joint_20261008`（Detached），worker shell PID454，训练主PID459；日志明确 `FRESH_START epoch=1/50 task_checkpoint=None optimizer_states=0 scheduler_last_epoch=0`。
- 2026-10-08 16:06:57 +08:00，正式epoch1已连续推进至少44/1907 batch，运行平均loss10.6561，有限loss、无Traceback/OOM/nan/inf，CUDA显存约12123MiB；主进程存活。比首次12batch观察继续推进，证明不是只看到进程。证据 `early_health.json`。尚未完成epoch1完整验证/50轮，也没有新test成绩。
- 重复调用launcher实际拒绝已有session，不退出或覆盖运行任务。新入口 `script/train_strict_v7.py`、`launch_v12_strict.sh`；保持旧trainer/模型/数据配置/04模块不改。本线仅修改新入口及01记忆/experiment_log，公共记忆由主会话维护。

#### 快照、查看与后续

实验目录：`checkpoints/v12_strict_joint_20261008/`，manifest保存精确参数、名单/配置/scene metadata/预训练权重/有效代码hash及scope，snapshot保存原配置、三个names、scene_state_info、全部有效模型/数据/评估源码、协议文档和测试文件；Git HEAD/status/diff和资源快照也保留。run_identity=`994b373cb5cbac1fd3a88febbcd613b1b7b722cc3d0a6a3a821c903088ee3bce`。数据不复制重划；不存在的instances.json也记录为不存在。

精确启动参数见目录 `command.txt`。WSL查看：`screen -r hicd_v12_strict_joint_20261008`，Ctrl-A D脱离；或 `tail -f /mnt/f/mambacd/home/HICD_v7/checkpoints/v12_strict_joint_20261008/train.log`。status.txt记录启动及结束exit，python.pid标识训练主进程；CSV每个完整验证轮追加，metrics JSON和latest/best在轮结束生成。

下一步：以新strict实验的日志/每轮artifact跟踪结果，不拼接旧BDD/v11指标；首轮结束后核对strict支持数、scope和best/latest身份。仍不自动改T4数据、Bridge、split或方法；若运行异常，只处理精确识别的新任务并保留证据。本轮未创建自动监控、推送或外发消息。

回复前复核：2026-10-08 16:09:31 +08:00，新实验epoch1已至至少124/1907 batch，运行平均loss9.8819，有限且无上述致命异常；与先前44batch观察一致持续推进。`early_health_followup.json`保存此第二次检查。

### 2026-10-08 代码清查影响复核、修复及仅T1–T3替代训练

授权来源：用户在本会话要求“看下代码清查记录，评估一下实际影响并按需修复”，随后明确“现在不训练T4，只训练1-3”。本段为最新状态，替代上方Bridge仍参与监督的v12计划。审计来源为05记忆的“2026-10-08代码清查记录”A01–A16；10/02 cleanup_report为文件清理历史，不能混作本次代码审计。

#### 影响复核及处置

| ID | 当前实际影响 | 本轮处置与证据 |
|---|---|---|
| A01/P0 | 命中旧v12：frequency enabled，前向使用freq_branch/freq_fusion但AdamW不注册，分支未优化、梯度未被该optimizer清零；不能由loss有限证明学习路径有效 | 两模块加入frequency组，lr1e-4；所有模型参数（包括暂冻结者）各注册一次，构建及开训覆盖断言。真实GPU smoke中30个频域/融合参数有有限非零梯度且更新；实际启动覆盖488/488参数张量 |
| A02/P0 | 命中train增强入口：目标/图像与state独立几何抽样，导致可能错位，影响联合监督；原100种子演示的85次不一致不能当真实数据错误率 | target/state从同一raw label同步crop+nearest后解码，一次random_augment同时处理两图/target/state；100种子4×4坐标回归全部一致，真实增强train四样本的每像素配对均可由源decode产生 |
| A03/P1 | 大于训练尺寸图像实际crop时target未同步crop；常规256→256不触发，未统计全部图像大图比例 | 单次raw读取并与两图共用crop_coords，目标/state统一resize；右下角裁剪回归验证图像、target、state一致；crop sampler读取decoded目标ID。读标签失败、图像/标签shape不符明确报错，不再吞异常；未映射raw值默认ignore，不假装Farmland |
| A04/P1 | 活跃train copy-paste p=0.8：误把Farmland0当空背景，并随机编造状态，可能改坏target/state监督；未估计历史实际发生比例 | 从配置识别Background15，仅向双时相GT背景粘贴源对象的成对像素，保留其源state和合法decode组合；不制造随机damaged/replaced/added状态。无源state不增强。回归验证Farmland不改、来源replaced保留、双时相内容配对一致。属于明确修改的增强规则，另开实验，不改写旧成绩 |
| A05/P1 | 活跃条件解码将logit乘小权重，负logit反而提高；不是“严格抑制”的实现 | 两种LabelStructure及实际HierarchicalConditionDecoder统一加入log(weight)概率先验；负logit反例/实际decoder/有限梯度验证通过。仍为软先验，不能声称完全禁止非法组合 |
| A06/P1 | 当前grad_accum1/unfreeze0.5未触发所列可选边界；未来长窗口/解冻可出错 | 每epoch清梯度，尾窗按实际长度归一并step/清零；backbone冻结参数预先注册，后解冻不会漏入optimizer；比例0的block切片与DINO.unfreeze0不再[-0:]全解冻。3batch/accum2数值回归及coverage/零比例回归通过 |
| A07–A11 | 实例路径索引/输入/推理/bbox/名称风险存在，但用户暂缓实例且本轮仅T1–T3像素 | 未改、未启动实例。像素数据boxes加载/增强接口也不作为已修复实例承诺；随实例任务恢复专项处理 |
| A12–A16 | 基线公平单模型入口、旧评估FP/状态折叠、best/数据回退、标签/分辨率/外部0.7851可比性未落实 | 本轮不修改基线或强行重训；仍为正式对照前置。新strict公共模块未改且27项测试保持通过；不能拿新strict与旧BDD数直接横比 |

#### 最新T1–T3配置

新文件 `datasets/configs/0617final_t123_trainfix.yaml`，不覆盖原0617final.yaml。只允许GT target 0/1/2/3/5/6/12/13（Farmland/Runway/Taxiway/Apron/Highway/Building/Aircraft/Vessel）和Background15；Bridge/Shelter/Pier/Dock/Tank/Crater及Tower的raw target/state均255、对应target初始权重0。GT标签/正样本不为T4提供监督，不启动T4实例训练；保留16/7输出通道和严格协议兼容，误预测到排除通道仍计FP，不掩盖错误。名单、split、图像和原配置未改，Farmland state loss的原ignore选择仍保留。

#### 实验切换与验收

- 2026-10-08 16:23:52 +08:00，核对cmdline后仅结束旧v12主PID459及其4个明确DataLoader子PID585–588；status记录16:23:53 python_exit143/tee_exit0，旧screen随后自然结束。所有文件保留，`stopped_for_training_integrity.json`记载epoch1停在704batch、尚无完整checkpoint。此run是缺陷版本的排查记录，不用作修复后效果或正式成绩。
- 新实验 `v12_strict_joint_t123_trainfix_20261008`，2026-10-08 16:25:28 +08:00从零开训；session `1709.hicd_v12_strict_joint_t123_trainfix_20261008`，主PID1716、worker1710。预算50/batch32/img256/LR1e-4/AMP/workers4/state-aware不变。正常DINO pretrained，其余模型/optimizer/scheduler/AMP新建，不读取旧run权重。
- 40项CPU验收全部通过：原strict27+训练完整性13，另bash -n/py_compile和git diff --check通过。新GPU smoke为4个**真实增强train**+2个真实val样本，train_updates2/val_updates1；训练配对source-decodable、T4 GT support=0/0；freq_branch/freq_fusion的30个参数非零有限梯度与实际更新。smoke只作通路验证，正式worker重新初始化。
- 2026-10-08 16:27:54 +08:00，epoch1至少103/1907batch、运行平均loss9.4367，连续推进且有限，无Traceback/OOM/nan/inf；GPU约12371MiB、利用率63%单次快照，主进程与screen存活。证据early_health.json。尚无完整epoch验证或新test，修复提升多少未知。
- 数据/协议范围改变和实现修复均独立命名，不在旧v12中途热换定义。当前代码get_param_groups形状已变化，旧v11/v12优化器状态不能直接在新实现下续训；旧源码快照仍保留，历史恢复需匹配旧实现。

新run_identity=`fd1ae754c259bb7bd3b550c4cdc1b3ac04b9932fbe2a723c7a52459d206a59b0`，配置SHA256=`c4beb1e57eaf3db996ee2ba0e37eb1341b6ffcf346a3d68d973bc3b8338e0ad9`。实验目录含manifest、source/config/名单/metadata快照、40项验收说明、审计原记录副本、真实smoke JSON与日志、command/status/PID/health。主CSV/逐轮JSON/best/latest规则保持strict v1：验证全前景joint micro最大，macro并报，raw state/64网格不变；T4行仅误报诊断，不是T4训练成绩。

查看：`screen -r hicd_v12_strict_joint_t123_trainfix_20261008`（Ctrl-A D脱离），或tail该run的train.log。后续先验收首轮完整strict报告、支持数和checkpoint身份，再判断性能；实例与基线缺陷仍待各自任务。未修改05/公共记忆/04协议、原始数据/划分或其他实验，没有自动监控/推送/外发。

回复前复核：2026-10-08 16:33:29 +08:00，新run epoch1至少453/1907batch、运行平均loss7.9583，有限且无上述异常，主PID1716和screen存活；early_health_followup.json与103batch的前次记录证明持续推进。该loss变化不作为修复增益或完整验证成绩。

### 2026-10-09 完成验收与结果诊断（仅验证集）

用户要求查看已完成训练的结果。本轮只读核验与结果整理，未重训、改loss/配置、推理test或更换best规则。

- 已完整50/50轮：metrics/epoch_001至050齐全，CSV50行逐行与JSON的global strict micro完全一致；50轮loss有限、每轮验证13418样本、T4 GT support均0。status：2026-10-09 11:32:30 +08:00 python_exit0/tee_exit0，train.log有TRAINING_COMPLETED；无训练screen/Python。总体运行约19小时7分，原进程PID1716已结束。
- checkpoint元数据通过ZIP/data.pkl pickle指令检查（不执行pickle、不加载tensor）：best_model epoch零基6/显示7、score0.7190867247396852；latest及checkpoint_epoch50 epoch零基49/显示50、score0.6261505155892628；三者best_epoch7、协议v1、run_identity与manifest/JSON一致。仅元数据核验，不声称完整tensor加载/test推理验证。

| 验证指标 | 官方best，第7轮 | 第50轮 |
|---|---:|---:|
| 全前景strict joint micro-F1 | 0.7190867 | 0.6261505 |
| target-state联合类别macro-F1 | 0.4024206 | 0.5204461 |
| T1 Building strict micro | 0.7067725 | 0.8001874 |
| T2 strict micro | 0.7277867 | 0.5225332 |
| T3 strict micro | 0.5933199 | 0.7069399 |

任务记录峰值：T1 epoch36=0.8045918，T3 epoch32=0.7213168；全局macro峰值epoch32=0.5233032。它们不是官方micro-best的同一checkpoint，不能拼成单模型主表；32/36未单独保留编号权重。本次官方best仍为7，不事后改名为50或宏最优。

第50轮各目标strict P/R/F1、定位F1：Farmland 0.4712/0.4292/0.4492、loc0.9396；Runway 0.9875/0.9511/0.9690、loc0.9690；Taxiway 0.9138/0.8909/0.9022、loc0.9050；Apron 0.9062/0.8173/0.8595、loc0.8600；Highway 0.8717/0.5738/0.6921、loc0.6928；Building 0.9078/0.7154/0.8002、loc0.8282；Aircraft 0.6995/0.5849/0.6371、loc0.7493；Vessel 0.8130/0.6573/0.7269、loc0.7897。定位和state-on-GT诊断不替代joint主分。

#### 关键诊断

1. **Farmland状态漂移主导全局下降。** 其GT7402722像素，全部state0（no-change），占前景50.1213%；Building占38.2130%，T3仅0.9007%。Farm joint从epoch7的0.7160降至0.4492，定位反从0.9127升至0.9396；GT农田区state微F1/准确率从0.7374降至0.4826，final错误中约1968470像素预测reduced、1012779预测added。joint混淆最大项为正确target0但state2（1857366）及state3（945807）。预测目标错误并不是这两项的原因。
2. **直接状态loss与严格评分不一致。** 启动配置及snapshot保留state_ignore_targets=[0]；V7Loss排除Farm的直接state CE/Dice，但strict要求Farm target+state均正确。辅助GNN validity监督仍存在，不能说Farm完全没有训练信号。上述是已查明的不一致及有数据支持的漂移原因候选，未进行恢复Farm监督的因果消融。当前配置/loss/core协议hash与启动manifest匹配；trainer当前文件后来增加了数值/CSV/日志格式化，已与启动snapshot做diff核对，仅见报告格式改动，snapshot本身hash匹配manifest。本次训练与结果来源以启动snapshot和原始metrics/CSV为准。
3. **不能泛称整体过拟合。** 从7到50其余七类joint F1全部上升、global macro提升，val_loss从1.7006降至1.3356（实际最低loss在39）；全局micro下降主要是Farm占比大且状态漂移。各任务后期峰值不同，单一micro选择牺牲了T1/T3后期进展。
4. **召回与稀有状态仍有瓶颈。** Building P0.9078/R0.7154、Highway P0.8717/R0.5738，主要缺失召回；Aircraft联合F10.6371，loc0.7493/state-GT0.7180，定位和状态都有损失。Aircraft replaced support715、F1=0.0022075（约0.22%）；Vessel replaced support8460、F1约0.3725；Taxiway reduced support182、F1=0。宏平均比micro更能暴露小状态问题，原始support为64网格像素数，不是实例数。

结果产物 `results_analysis_20261009.json` 保存best/last全表、全50轮曲线、任务峰值、状态混淆与核验边界。来源原文件为metrics/epoch_007.json、epoch_050.json、train_log.csv、status.txt及checkpoint元数据。当前结果是val，不是test；旧split独立性风险和与旧BDD/外部0.7851不可直接横比的限制仍保留。没有由本run单独证明审计修复的增益。

下一步优先在同T1–T3/同严格协议下对齐Farm直接state监督与评分，再处理飞机replaced、公路/建筑漏检；不以去掉Farm或事后强制GT区域state0抬分。正式test按预先选择的best7权重评估；如果比较其他checkpoint，明确各自选择准则，不拼接任务峰值。当前请求为看结果，故本轮不自行改配置或启动新训练/长test。

### 2026-10-09 最新农田语义澄清与独立解码复算

用户明确：“农田我记得只有不变吧，我们理解的是无论是不是被破坏了都能种，所以应该都按不变算”。这是本项目标签约定：Farmland仅no-change，不把农田的损毁解释为其状态分类变化；不将这句话推广为现实世界所有损毁农田均可耕作的结论。50轮val GT农田均只出现state0已核实。

**撤回上段‘恢复Farm直接状态CE/Dice监督’作为首要修复建议。** state_ignore_targets=[0]符合只对其他目标学习可变状态的意图；缺口在预测语义解码未实施Farm固定状态，原始自由state输出产生了项目中不允许的Farm状态。图辅助监督仍在；不因raw漂移改GT或放宽其他类别的严格匹配。

本线新增opt-in `evaluation/farmland_semantics.py`：`decode_farmland_state(pred_target,pred_state)`仅依据**预测**target0把state设0；其他预测不变。`FarmlandNoChangeModelEvaluator`在现有strict流式计数上生成等价解码结果；04原core/adapter和已有raw报告不改。错把Building/其他类/背景判Farm仍计FP/FN，GT和ignore范围不变。计分公式仍hicd.strict_joint.v1，新增独立预测语义版本hicd.predicted_farmland_nochange.v1。

利用每轮联合混淆矩阵合并所有预测Farm列到(Farm,no-change)，同时精确更新state-on-GT矩阵；不读取GT来决定如何解码。4项回归通过：3000随机像素直接解码vs计数复算完全一致（含各目标状态诊断）、错目标/背景FP保留、GT存在非0农田状态时拒绝而非重标、Torch helper和实际模型batch adapter。复算50轮、验证非Farm的joint与全部目标loc均不变、valid/ignore计数不变。

| 第50轮验证指标 | 原自由state输出 | Farm固定no-change解码 |
|---|---:|---:|
| 全前景joint micro | 0.6261505 | 0.8786285 |
| 联合类别macro | 0.5204461 | 0.6185618 |
| Farm joint | 0.4492441 | 0.9395602 |
| T1 joint | 0.8001874 | 0.8001874 |
| T2 joint | 0.5225332 | 0.9272358 |
| T3 joint | 0.7069399 | 0.7069399 |

新规则下第7轮micro0.8218201；全50轮回顾峰值为46轮0.8789693，但没有46轮独立权重。在已保存的7/10/20/30/40/50轮权重中50轮最好（0.8786285）。原best_model仍保存raw规则选择的7轮，不覆盖/改名；后续使用新语义评估时应明确选择50轮latest/checkpoint_epoch50及这条预测规则，不能声称训练当时已按该政策选出46轮。

结果目录 `checkpoints/v12_strict_joint_t123_trainfix_20261008/farmland_nochange_review_20261009/` 保存每轮独立decoded验证JSON、源文件hash/identity和summary.json。它是用户语义澄清后的**回顾性val复算**，数学上等价于同一批预测先解码再评分；没有新推理、test或重训，也不是新的学习增益/SOTA证据。T1–T3范围不变、T4 GT仍0。原CSV/metrics/checkpoint/manifest不动；原raw分析作为历史诊断保留，后续结论必须标明预测策略版本。

后续重点改为：正式推理/验证显式采用该Farm语义解码，其他类别保留原始细状态并严格评估；处理Aircraft replaced以及建筑/公路漏检。Farm不再作为多状态损毁分类能力评价对象，不将其fixed状态正确率称为学会损毁分类。本轮仅新规则实现、数值复算与本线记录；没有改公共记忆、数据或启动任务。

### 2026-10-09 用户授权第50轮完整test与随机GT/Pred可视化：已完成

用户明确要求第50轮跑测试集及随机GT/Pred图。本轮新脚本 `script/eval_epoch50_strict_test.py` 真正加载checkpoint_epoch50并对test全量推理，不是重复val计数复算。epoch零基49、协议v1、run_identity、checkpoint dataset_config与冻结snapshot验证；当前models/datasets/core hash匹配训练manifest，state_ce动态buffer恢复后strict=True载入。float32、raw state_logits、GT nearest0.25→64网格，预测Farm固定state0策略保持。

- GPU已有BiSRNet PID144481训练，未终止/修改；测试batch8/workers2，显存资源并行可容纳。model.eval推理投票模块对raw无影响，以单样本原/跳过路径allclose及argmax完全相同验证后跳过该后处理，未改变用于指标的raw头。无targets前向仅推理、不恢复optimizer或更新参数。
- test实载13418/13418、processed13418、无样本丢弃；2026-10-09 15:27:11 +08:00正常完成，约801.54秒（13分22秒），PyTorch峰值allocated约819.37MiB（不含reserved/其他进程/驱动）。status complete、FULL_TEST_COMPLETE，测试进程已退出，基线训练继续存活。

| 解码后test指标 | P | R | joint micro-F1 | joint类别macro-F1 |
|---|---:|---:|---:|---:|
| 全前景 | 0.9517749 | 0.8289321 | 0.8861163 | 0.6827043 |
| T1 | 0.9182210 | 0.7204813 | 0.8074207 | 0.7372190 |
| T2 | 0.9705850 | 0.8914617 | 0.9293423 | 0.7407817 |
| T3 | 0.7612633 | 0.6400190 | 0.6953959 | 0.5799463 |
| T4 | N/A | N/A | N/A | N/A |

全前景TP12667376/FP641838/FN2614184/support15281560；T4 GT/pred support均0。保留未应用Farm规则的raw test对照micro0.6255719，不把它与语义解码结果混表。

test各目标joint P/R/F1：Farmland0.9822/0.9084/0.9438；Runway0.9718/0.8961/0.9325；Taxiway0.9167/0.8648/0.8900；Apron0.9027/0.8106/0.8541；Highway0.8760/0.6728/0.7611；Building0.9182/0.7205/0.8074；Aircraft0.7183/0.6911/0.7045；Vessel0.7768/0.6246/0.6924。全部target/state同对，不是仅定位。Aircraft replaced support1452px、F10.2055974、R0.13154；Vessel replaced support17537px、F10.3831930、R0.26470，稀有状态仍弱。

**原划分的同名重叠必须随结果解释：** test/train同名197张、test/val同名341张，未去重重划。按同名分组另报告而不替换完整test：无train同名的13221张micro0.8859314，T1=0.8068319/T2=0.9293637/T3=0.6453032；有train同名197张T3=0.8784704。整体micro受影响小不能掩盖T3差异，同名不重叠也不证明场景/空间独立。用户关于实例有意过拟合的策略不自动视为像素结果已严格独立。

可视化按推理前seed20261009均匀无放回抽6张，固定dataset indices3972/5557/6343/8283/8665/12239，不按结果替换。保存两时相model-input RGB、GT64、pred target/state、raw state数组及样本级strict计数。sample2双输入全黑、GT/Pred全background、前景support0、F1 N/A；sample4 GT Taxiway而pred Runway、前景support41格点、jointF1=0，均原样保留。其他随机sampleF1：1=.9694169，3=.7130834，5=.7442509，6=.5511811。

使用nature-figure的Python流程：PNG300dpi、SVG/PDF可编辑文字；单图8panel为pre/post、GT/pred target、GT/pred foreground state、joint错误、错误叠加；另6行总览。掩膜nearest显示，无额外crop/contrast；Pred background仅在状态图显示为none，原state诊断仍保留真实头输出；GT-ignore斜线，Pred底色不隐藏。错误色：青绿正确前景，紫漏检，橙错目标/背景FP，蓝错状态。source QA无FAIL，3个WARN（报告宽度/PNG300而非600/未请求TIFF）按诊断报告契约记录；实际图检查后修正总览标题重叠，全6图导出。

结果目录 `checkpoints/v12_strict_joint_t123_trainfix_20261008/test_epoch50_farmland_nochange_20261009/`：test_metrics.json包含raw/decoded/同名分组；per_target.csv、per_target_state.csv、sampling/sample_details、manifest/checkpoint hash、status、6份source NPZ、figures总览和6个单图。全量计数与分组TP/FP/FN/support加和一致，F1公式复核、抽样名单和6图对应验证完成。原训练/val产物和best文件未覆盖，无重训/改划分/停止基线/自动排程。

### 2026-10-09 用户要求撤回T4过拟合划分共享：原名单恢复完成

用户最新明确：现只训练T1–T3，应修正先前为T4过拟合留下的样本共享；随后补充“没有换名字的情况”。采用**恢复原有真实名单**，不再扩大为改名/内容去重或重新随机/按城市划分。

#### 追溯与修复范围

- source `D:/CD/0617final_256_v2` 当前names61029/13418/13418。同名交集train/val197、train/test197、val/test341。
- 最早names.txt.bak：train64481含3452条synth前缀，剔除这些后真实61029与当前train集合完全相等；val/test备份均13077、无synth。当前val/test各多341条，原备份条目均仍在当前集合，没有原样本丢失。恢复train保持当前bytes及顺序，val/test用原备份bytes直接复制。
- 原名单按准确母样本键（仅去末尾_NN_NN切片坐标，不误剥长日期/对象ID）互斥。旧当前母样本交集103/103/144，恢复后0/0/0。原切片脚本（HEAD历史cut_0617_256.py）证明1024裁256、stride192（相邻切片重叠64），所以同母样本不能跨split；恢复方案已保持这一点。
- 用户澄清后仅停止本会话自己的CPU内容审计PID151764，未完成/未采用内容hash去重；其源脚本移至datasets/split_audits/disjoint_20261009/abandoned_content_audit_source.py，状态标为取消，不把未完成检查记成结论。采用的流程是script/restore_original_real_splits.py。

#### 已发布的数据版本

新view `D:/CD/0617final_256_v2_disjoint_20261009`（WSL `/mnt/d/CD/0617final_256_v2_disjoint_20261009`），版本hicd.original_real_membership_restored.v1，identity `9f66845372df4c8730a543ac2899575b8ba40eff31b676355b183991e09db05b`。

| 集合 | 旧名单 | 新名单 | 撤回后加条目 |
|---|---:|---:|---:|
| train | 61029 | 61029 | 0 |
| val | 13418 | 13077 | 341 |
| test | 13418 | 13077 | 341 |

新root的train names与旧root逐字节一致；val/test与各自原names.txt.bak逐字节一致。同名交集、同母样本切片交集均0；解析后的pre源文件路径跨split交集0。为避免旧的兄弟目录/全split首次命中规则以后换源，新root每个split包含resolved_samples.json，固定全部87183个样本的实际pre/post/label路径；scene_state_info按原数据复制。没有复制/移动/删除原图片或改原names.txt，故旧run可以保留其原始数据身份，正在进行的任务不会未来静默改成新test。

Dataset新增catalog-only分支：验证manifest/catalog/name SHA、schema、唯一且一致的membership，转换Windows/WSL路径并限定数据父目录；失败即报错，不fallback到其他目录。旧root无catalog时使用原逻辑，旧实验/基线的冻结loader不被热改。

#### 后续入口与已执行验收

- 新配置datasets/configs/0617final_t123_disjoint_20261009.yaml继续T1–T3、全部T4/Tower ignore；显式声明Farm预测固定no-change。新配置的TrainingEvaluator启用已验证的Farm解码后strict计分，原loss反馈保持；旧配置无该flag时保留raw行为，不重写旧结果。
- launch_v12_strict.sh后续默认指向新view/new config，新实验默认名v12_strict_joint_t123_disjoint_20261009。prepare_run将split_manifest/每split路径catalog一并hash和snapshot，checkpoint记录split identity与预测语义版本；epoch50测试入口改为从checkpoint args读取配置/使用manifest实际test数量，不再硬编码13418。
- 51项CPU测试全部通过（含3个catalog完整性/拒绝静默fallback测试）、语法/bash-n/diff检查通过；实际new loader装载61029/13077/13077，三split各读取首/中/末共9样本，图像/GT shape、T123-only解码、Farm state0通过。另用小数组确认new TrainingEvaluator joint固定Farm规则且仍输出原IoU反馈；新目录未用GPU训练。
- 审计产物datasets/split_audits/original_restored_20261009/summary.json、before原名单、removed_members.json、loader_acceptance.json、acceptance.json；active_split.json登记后续默认版本。源图片可变且未逐文件内容hash/复制为immutable snapshot；按用户范围未做额外内容去重。城市前缀仍跨split共享74个（原设计），不是城市级/OOD重划，也未做地理footprint审计，不能扩大称所有城市/空间因素已独立。
- 本轮最初发现BiSRNet旧run在训练，未停止它。最后核查其同级status已由其他用户授权工作记录USER_TERMINATED 2026-10-09 15:52:54 +08:00 reason=sample_leakage、automatic_test cancelled，无训练screen/Python；本会话没有操作其进程/文件。基线后续需与02一起更新data root、冻结catalog loader、reference/config/hash，再按新协议从头训练；不沿用旧val的best作为新数据正式对比。

#### 旧结果与下一步

v12训练/50轮权重/旧test88.61及图像不删除或改数，另加data_split_superseded_20261009.json明确其是旧划分结果。旧train集合虽未变，但旧val反馈/选best含共享条目；正式新划分比较须重新训练/验证，不能靠改文件路径把旧实验追认为新实验。新test固定13077；本轮数据恢复完成，未自动启动新HICD/基线训练或重评，仍只T1–T3。公共记忆/注册表及04/02工作流未编辑，主会话/相关线可读取本记录和active_split。

**同轮最终推广更新（优先于上文初始发布时原names不动的描述）：** 再次确认无HICD/基线训练或测试Python，旧BiSRNet已由用户终止后，原目录D:/CD/0617final_256_v2的val/test names也使用临时文件+原子替换恢复到13077；train bytes不动。三个原目录names与新版逐字节相同、原旧名单已保存在audit/before，图片/权重/原指标不变。canonical_transition.json及新版canonical_source_transition_20261009.json记录推广和检查条件；新版原split_manifest保持其发布时身份，不覆盖。新root仍为主训练默认的固定路径版本；旧baseline启动若沿用旧reference/名单hash会拒绝不匹配，须重新建立新reference而非修改旧manifest。没有重新启动GPU任务。

### 2026-10-09 最新授权：恢复划分后从头100轮训练已启动

用户明确要求数据集处理完开始新训练，并将预算加到100轮。本轮在恢复完成并确认无其他训练后实际执行，不恢复旧50轮任务权重。新实验 `v12_strict_joint_t123_disjoint_e100_20261009`，run_identity `d1584ee10d43b2df6e2b2e6922535cb2451a9f44941bedd8bcb23dba4ea93d75`。

- 数据版本及T1–T3、T4/Tower ignore固定：train61029/val13077/test13077，names与母样本tile跨split交集0，catalog固定实际来源；split identity仍9f66845372df4c8730a543ac2899575b8ba40eff31b676355b183991e09db05b。没有重洗牌或附加内容去重/城市留出。
- 模型/损失/采样/batch32/img256/LR1e-4/AMP/workers4/freeze0沿用修复版；正常DINO预训练，其余任务/AdamW/scheduler/AMP从零。CosineAnnealingLR T_max改100，因此学习率下降铺到100轮；这并不是用旧50轮末的极低学习率追加50轮，也不是完全单因素的旧划分50vs新划分100对照。
- 农田预测固定no-change已在训练每轮strict报告和primary best中执行，raw head/nearest64仍保留，主状态loss继续ignoreFarm。主best_model按val全前景joint micro最大；新增best_macro.pth独立保留joint类别macro最大及best_macro_score/epoch，latest每轮、每10轮编号。secondary criterion写manifest，官方主准则不改、不以test选模型。
- 当前旧50轮解码后46/50 micro仅87.90/87.86，相对已平稳，不能保证加轮次一定提升；100用于检验召回/稀有状态继续学习空间。修正划分与新预算同时变化，收益须以新val/test实际观察，不用旧88.61宣称新协议baseline。没有设置提前停止或自动改变其他方法参数。
- 51项CPU测试、bash-n/py_compile通过；独立real train4/val2 GPU smoke通过且30个频域/融合参数有有限非零梯度和更新、train/val strict update通路2/1。两val smoke样本仅背景，foreground n/a如实保留，仅通路验收，非性能估计。正式worker重新初始化。
- 2026-10-09 17:11:55 +08:00 STARTED，screen157237.hicd_v12_strict_joint_t123_disjoint_e100_20261009（Detached），主Python157244、worker157239；FRESH_START epoch1/100 checkpointNone/optimizer0/scheduler0，日志同时声明semantic policy、split身份和scheduler_T_max100。
- 17:16:40 +08:00 health：epoch1至少480/1907batch，运行平均loss7.6386，连续推进、finite、fatal_patterns空，GPU约12879MiB/利用率83%瞬时快照。主进程与screen存活，无重复启动、OOM或Traceback；尚无epoch1完整验证。early_health.json和training_receipt.json保存证据。

目录 `checkpoints/v12_strict_joint_t123_disjoint_e100_20261009/` 已保存原始参数/当前源码/config/三个names/catalog及split_manifest快照和hash、独立smoke、status/PID/command。active_split.json更新预算及运行身份。看train.log或screen -r hicd_v12_strict_joint_t123_disjoint_e100_20261009（Ctrl-A D脱离）。100轮在后台执行，本轮仅启动/早期健康完成，没有提前称训练/验证/test完成，也未创建额外定时监控。

回复前第二次检查已连续推进超过1000/1907batch（先前480），loss有限、未见上述异常，主进程持续存活；最新精确时间/批次/loss以early_health_followup.json为证据，不当完整轮次成绩。

### 2026-10-10 当前结果可信度核查（训练继续，未改代码/数据）

用户询问指标高是否可信。本轮只读复核，不停止/调整100轮任务。初读42轮时micro-best epoch40=0.8810079；核查期间43轮val完成，主PID157244仍运行，开始44轮；每轮val13077。最新micro-best更新为epoch43=0.8831862，其macro=.5678663；macro-best仍epoch39=.6025228。不是完整100轮或新test结果。

- 新view三个names和resolved catalogs的live SHA均与run manifest一致；same-name及实际pre源路径train/val/test两两交集0，已知人为名单共享已消除；source split身份与固定Farm预测规则均正确。
- 对全部已完成43轮的保存106×106联合矩阵独立计算TP/FP/FN及micro/macro，与原JSON精确匹配；矩阵总数等于valid_pixels，valid+ignored等于13077×64×64。Farm预测非0状态列、GT Farm非0状态行均0；GT用于评分/ignore，不用于决定预测target或Farm解码。正确背景bin不进前景TP，背景错前景仍计FP。
- 第43轮训练/验证mobile差异：Aircraft train0.8493258/val0.6673715，Vessel train0.8884361/val0.6310640；log混合输出两个集合，不能把train表当val。当前最佳micro40轮val T1=.8127987/T2=.9243045/T3=.6578302，macro=.5530660；最新val及全表见credibility_review_20261010.json和epoch_043。
- 最终43轮快照：micro=.8831862、macro=.5678663、T1=.8086347/T2=.9297337/T3=.6436613，成为此时micro-best；上项40轮为初始读取快照，不覆盖新43轮结果。
- 验证前景中Farm约50.8218%，no-change约84.7535%；Farm固定no-change是用户明确的类别定义，不是模型学会Farm损毁分类。高pixel micro可以与低小状态macro共存，当前高分支持同分布像素识别，不等于全部损毁state有88%效果。第42轮示例Aircraft replaced F1=0（694px），Vessel replaced约.0229（5957px），不能被总体分数掩盖。
- val是开发集：既选best，又通过train_full_v7.validate的update_adaptive_weights反馈target/state IoU更新下一轮loss权重；当前val不是独立最终测试。当前100轮尚未跑新test。新版test是旧test撤回341条后的子集，上一轮已看过多数样本，不能称全新外部盲测；后续冻结预先选模规则，避免用test反复调参。
- 原城市/场景前缀共享74不是同名泄漏的再现，但城市/地理footprint独立未建立；本次是恢复原分布，不是跨城市泛化证明。按用户澄清未做改名/像素内容去重，不能谎称全面内容/空间审计完成。单seed、同协议基线与消融仍待结果。

判断：指标实现、数据身份、已知名单泄漏修复可核验；当前同分布validation结果有可信证据，但全部损毁状态能力、最终独立test/跨场景泛化、SOTA/模块因果收益尚不能由这些高分证明。仅新增credibility_review_20261010.json及本线记录，训练继续原计划。

### 2026-10-11 用户要求查看结果并做好实验记录：100轮完成

**已核实状态**：`v12_strict_joint_t123_disjoint_e100_20261009`已于2026-10-11 06:13:10 +08:00正常完成100/100（python_exit=0、tee_exit=0），开始于10/09 17:11:55，历时37小时1分15秒。所有100轮train/val指标齐全；本轮进程/GPU查询无该HICD训练/测试任务。10/10“进行中”记录为历史快照。

- 主micro-best：第90轮，val micro=.895640499834862、macro=.5782007805694767，P=.9542820777224115、R=.8437888406901437；T1=.825335223374906/T2=.9395697396895657/T3=.6478249664642406；保存best_model.pth。
- 独立macro-best：第56轮，val macro=.6226105183775097、micro=.8875011299395743；T1=.8182102018717871/T2=.9310162489601078/T3=.6381178993934113；保存best_macro.pth。原主选择准则保持micro，未采用此前macro建议热改已运行协议。
- 最终100：micro=.8942938288614153、macro=.5780676742646743；T1=.8212818747883126/T2=.9394593161357572/T3=.6445628629519874；train loss=.8486165818457961、val loss=1.2801047280740476。latest和编号100均正确。
- 同run前50轮峰值：micro49=.8856567037366926、macro39=.6025228321513919；全程各提高约1.00/2.01个百分点。编号50自身micro=.8795338708422991、macro=.5928851103381544。增加轮数主要改善T1/T2和召回，T3没有同步改善；最后10轮micro89.3132–89.5267%近平台，不能保证继续加轮收益。
- 目标F1（micro-best90）：Farm95.03/Runway97.41/Taxiway91.69/Apron87.88/Highway72.08/Building82.53/Aircraft66.99/Vessel63.67%。replaced在56/90/100：Aircraft22.38/12.32/7.22%，Vessel21.44/4.83/4.39%；100训练T3 micro90.93%、val64.46%，后期小状态泛化仍弱。
- macro支持边界已查明：验证GT非零支持组合恒26；第56/90/100有效组合27/29/29，额外无GT但有预测组合按0进入既定macro。仅固定GT组合的诊断macro64.6557/64.4916/64.4768%，明确不能替换正式macro、重选best或声称原定义计算错误。
- 验收：独立计数重算全部200份train/val报告，micro/macro和TP/FP/FN精确一致，像素闭合、Farm0规则、finite loss及指定致命日志标记通过。训练drop_last，每轮实际61024张；验证完整13077。live数据名单/catalog等SHA与manifest一致。5份checkpointCPU mmap读回epoch/身份/score/scheduler，best90/bestmacro56/编号50/100/latest正确；未做大checkpoint内容SHA或新模型推理。

**记录产物**：`docs/experiments/v12_t123_disjoint_e100_completion_20261011.md`完整报告；`docs/experiment_log.md`更新已完成并追加结果；run下`completion_review_20261011/summary.json`全精度与100份指标hash、`epoch_history.csv`100轮曲线数据、50/56/90/100的target_state CSV和可重跑audit_completion.py。原指标、权重、源码/config不覆盖；公共记忆/注册表及02/04等专题未编辑。

**下一步与边界**：本实验新test尚未跑，本轮只读核查与记录。建议固定既定主best90和独立补充macro-best56，在同一test13077各评一次另记，不能根据test择优重定义选模。val仍是开发集（选best与adaptive反馈），恢复test是此前旧test的子集，非新外部盲测。单次运行/未齐备同协议基线消融、城市前缀共享和未完成内容/地理审计限制保留；旧split50 vs新版100不可称单因素因果提升。



### 2026-10-11 第90轮完整测试集已完成

用户明确指定第90轮权重跑test并记录。加载当前run的best_model.pth（内部epoch89），strict=True，run/split/config/源码输入hash和实际轮次校验通过；checkpoint SHA256 `dd8b2300c7f2f6208bea3af19be7bb8869d15f3e379a5c05b7b0695ced5be98e`。采用固定第90轮，不根据test另选模型；没有评估第56轮或更新训练参数。

2026-10-11T16:41:46.834415+08:00完整13,077张test完成，耗时544.65s；float32/raw state/nearest64、batch8/workers2、T1–T3-only及Farm预测固定不变。train/test与val/test同名交集均0，未删样本。本run新test状态由未评估更新为已完成，之前记录保留为历史。

test strict joint P=95.5402%、R=85.0587%、micro-F1=89.9953%、macro-F1=62.2192%；T1=83.1818%/T2=93.8324%/T3=62.5648%。TP=12809293/FP=597933/FN=2250058；T4返回N/A，其GT忽略，不作能力结论。相同90轮val micro89.5640%/macro57.8201%，val是选模/adaptive反馈开发集，测试另列。

目标及所有状态P/R/F1/support见完整报告与CSV；replaced：Aircraft F1=33.1322%（GT1431px）、Vessel=20.1583%（GT15034px）。官方test macro有效组合28、GT支持组合25，继续把无GT但有误预测的组合按0纳入，不更换公式。仅GT支持macro诊断=69.6855%，不能替换正式分数。

独立矩阵复算raw/decoded micro/macro/P/R和全部目标/状态计数精确一致，样本/像素闭合、组别全量一致、Farm合并与身份核验通过；证据verification.json。产物`checkpoints/v12_strict_joint_t123_disjoint_e100_20261009/test_epoch90_20261011/`、目录旁log和`docs/experiments/v12_t123_epoch90_test_20261011.md`。examples0，未额外渲染图片；原权重/训练数据/指标不覆盖，公共及其他专题记忆未改。

边界：恢复test是此前已看过旧test的子集，非新外部盲测；城市/内容独立性、单次运行、同协议基线/消融仍待证据。不能用旧划分50轮test与本结果归因预算或模块增益。下一步以固定协议结果作为本run test记录，保留各小状态诊断。

### 2026-10-12 方法重构诊断与方案（用户明确暂不改模型）

用户认为基线44轮全面领先，提出方法重构；结构化回答选择“先完成诊断和重构方案，暂不改模型”。本轮仅CPU读取/计数复算与方案记录，未改模型/训练配置/指标，未新建GPU训练/测试，未停止正在进行的Bi-SRNet PID366723。使用academic-research-suite的experiment-agent plan指导整理证据、混杂和分阶段控制；方案未实施，不能当新方法结果。

- 对齐已核实：新版Bi-SRNet `bisrnet_t1t3_20261011_164416`与HICD disjoint e100相同split identity/strict协议/Farm规则，8项共同数据文件hash一致，5份比较val报告的完整GT联合类别行边际相同。45轮基线指标独立重算micro/macro通过；训练仍进行，基线test未完成。主比较固定44轮，不按后续进度改数。
- “全面超越”不成立：Bi-SRNet44 val micro88.5944/macro43.2430/T1 81.3342/T2 93.5127/T3 59.0247%；HICD44为88.4808/56.3079/80.6106/93.3330/62.3969%；HICD主best90为89.5640/57.8201/82.5335/93.9570/64.7825%。基线45快照micro88.7524、macro45.3217、T3 56.6290。只val对val，不与HICD test90.00混比，不提前做最终等预算优劣结论。
- 行为差异：基线44 P87.4440/R89.7755%，HICD44 P94.4166/R83.2472%。GT前景→bg分别1180783/2243839；bg→前景1568654/522896。HICD90联合FN中93.21%是bg漏检，T3 FN中bg漏检64.88%、同目标错状态23.83%、错目标11.29%。证实更保守、更少误报/更多漏检，不能单凭计数认定背景loss为唯一原因。
- 代码与启动SHA相符的接入事实：dataset仅RGB/255、DINO wrapper未做mean/std；官方预训练要求匹配LVD或SAT的Normalize，本地权重来源/transform元数据待确认。dinov3 extract[2,5,8,11]发生在block执行前，实际返回前一block输出；最后block11及final norm未进入multi_scale。修正收益需独立训练，不热改旧权重输入再做因果宣称。
- 架构事实：HICD主头只用TFM fused，丢弃独立pre/post输出；ViT四层都原生16×16，64/32来自插值，频域/NFA再加工。候选是保留DINO语义+原生stride4 RGB细节，显式pre/post/abs/signed方向保留、目标条件状态或T3 adapter；先通过简单控制验证，不先断定骨干无效。
- GNN检查：CPU读best90及macro-best56，25个有val GT支持、非Farm组合validity最小1.0，低于0.05门限0个；没有看到这两个checkpoint门控删GT监督。图合法表仍Farm0/Farm2，而项目Farm仅0，辅助语义待同步；不把存在门控逻辑说成已证实性能原因。
- 方案顺序：A0权重/Normalize/after-block接口及梯度核验；A1/A2分别控制输入与抽取修正；B固定weighted CE对照后分别研究Dice/背景抑制/自适应；C保留时序、细节、条件状态逐项增加；D图/频域单独加回。旧16/7错误空间与strict计数不改；如果未来限制合法输出，需对控制方法同样施加并单列其贡献。新macro选模建议/2pp阈值仅提案，未成为运行规则；最终候选多seed、固定test，不靠反复test调参。

产物：`docs/experiments/method_reconstruction_20261012/diagnosis_and_plan.md`、comparison.json、target_comparison.csv、target_state_comparison.csv、gnn_validity_probe.json及可重跑CPU分析脚本。台账已追加；公共记忆/注册表、02专题和基线代码未改。下一步由用户确定实施范围后先A0，不在本轮擅自落实架构或抢GPU。

### 2026-10-12 新授权：DINO接入校正完成，基线结束后自动接续主训练

用户明确要求先校正DINO，等对比实验跑完（预计明早）开始重训主实验。此授权覆盖本轮接入实施与条件式未来启动，优先于上一轮暂不改模型；没有授权实施损失/时序/细节/GNN全面重构。

- 独立新adapter `models/backbone/dinov3_corrected.py`，版本hicd.dinov3_input_features.v1：DINO内部RGB[0,1]→LVD ImageNet mean/std，频域原输入不变；真实timm1.0.28 Eva.forward_intermediates取after-block2/5/8/11，norm=True、正确RoPE/5prefix。projection/原金字塔、其余结构/损失不变。旧dinov3.py/HICD factory/数据loader/strict evaluator/旧配置源码SHA保持，独立builder及train/eval新入口显式启用版本。
- 本地pretrained SHA c70aebcfbe6ad4b504aadf274feb4eebaf9af6b649c66de6c6e1670c81810fc9；从固定HF revision c6a5fb7d12bbd3cf3b0079253141c3332aaed7da下载对照timm/vit_base_patch16_dinov3.lvd1689m，远端SHA1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332。实际模型162个张量全部torch.equal，确定LVD归一化；本地额外24个q_bias/v_bias键不被当前timm模型使用，未替换原文件。weight_provenance.json记录。
- CPU接口验收：四层token与独立after-block钩子一致，原图未修改，mean输入归一化0，形状64/32/16/8正确，last-block qkv与final norm梯度finite非零。真实CPU2增强train一次AdamW更新末block实际变化+2 real val strict8192px闭合；两val仅bg，foreground N/A仅管线验收。初次CPU报告脚本误用update返回值，修复后重跑通过，未涉及模型变更或正式run。
- 54项CPU测试通过（原51+3门控）、bash-n/py_compile通过。GPU smoke未运行，待基线释放后由launcher执行原4real train/2val smoke，成功后正式进程重新初始化。既有Bi-SRNet训练继续，未抢GPU或终止它。
- 新run唯一名v13_dino_corrected_t123_disjoint_e100_20261012；配置0617final_t123_dino_corrected_20261012.yaml只有新增dino版本字段，与原cfg其余解析一致。原数据61029/13077/13077/T123/T4ignore/Farm0、batch32/img256/raw64/LR1e-4/AdamW/cosine100/AMP/workers4/冻结解冻策略/损失/采样/micro主best与macro次best保留，无resume。新checkpoint/config/source/provenance由新prepare显式snapshot/hash。
- 条件入口launch_dino_corrected.sh + check_dino_handoff.py/check_dino_ready.py：基线100轮与完整test必须均exit0、producer身份及13077覆盖正确，其他train/eval和GPU空闲，共享baseline/HICD锁可取得；17份关键源码/配置与ready_manifest一致。check-only当前WAIT。未建立正式run目录或提前称训练已启动。异常或部分run保留现场、不覆盖重试；已启动run不重复启动。
- 已实际创建本聊天heartbeat automation，name“基线完成后启动校正 DINO 主实验”，id=dino，ACTIVE，每小时检查。读取READY后按条件启动，不靠固定明早时刻抢占；未变时静默。成功且两次batch推进/finite健康后记录early_health与台账并暂停automation；失败需报告并暂停，不停其他进程/不自动重试。使用openai-docs调度说明配置；用户无需重复授权这次新run。需要电脑及Codex保持运行、项目可访问。

验收与未来接续的权威入口：`docs/experiments/dino_integration_20261012/READY.md`、ready_manifest.json、cpu_acceptance.json、real_cpu_smoke.json、weight_provenance.json；目录旁CPU日志。新训练后评估必须用eval_dino_corrected_test.py，旧run仍用旧入口。只维护本线/台账，未改公共/02记忆或运行基线。当前状态为CPU_READY_GPU_PREFLIGHT_DEFERRED_UNTIL_BASELINE_COMPLETE。
