# HICD 方法诊断与重构方案

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-12，Asia/Shanghai
- Verification Status: UNVERIFIED（重构方案未执行；下列已有指标已独立计数复核，代码事实为只读检查）
- Version Label: hicd_reconstruction_proposal_v1
- Human Scope: 用户明确选择“先完成诊断和重构方案，暂不改模型”。没有改模型/训练配置/指标，没有启动新训练或测试，未停止正在运行的Bi-SRNet。
- Sources: comparison.json、gnn_validity_probe.json、两run manifest/每轮原始metrics、当前与HICD启动SHA相符的源码；外部仅DINOv3官方输入与特征接口说明。
- Data State: 已有结果为单次运行的开发集比较；不是重构后的新性能，不代表机制因果证据。未上传实验数据。

## 1. 研究问题与核实结论

问题：在固定T1–T3数据、严格目标—状态协议和预测农田不变语义下，能否减少HICD的背景漏检，并改善移动目标的具体状态识别，同时保住较低误报？

用户关于基线44轮“全面超越”的判断需要收窄。核查时基线完成45轮、仍在训练：44轮micro略超HICD同轮，但macro、T3较低；没有超过HICD micro-best90的总体、T1/T2/T3严格联合F1。基线100轮和test尚未完成，不能提前宣称最终优劣。

对比对象：

- HICD：v12_strict_joint_t123_disjoint_e100_20261009，100轮完成；主best90，独立macro-best56。
- 基线：bisrnet_t1t3_20261011_164416，Bi-SRNet-HICD-T1T3-adapted，并非原SECOND任务的原样复现。
- 两者protocol、split identity、Farm语义均一致；8项共同数据源hash一致；5份比较报告的完整GT联合类别行边际、有效/ignore像素计数一致，证明比较的是同一验证像素总体。
- 基线45轮已有指标全部从106×106矩阵独立重算micro/macro通过。源码、指标hash和全精度结果保存在comparison.json。固定使用用户指出的44轮，不追逐训练过程中最新数字重写本次结论。

## 2. 同口径验证集比较

单位%；每列来自一份真实checkpoint轮次，不混拼分数，也不将HICD test与基线val比较。

| 指标 | HICD44 | Bi-SRNet44 | HICD主best90 | Bi-SRNet45快照 |
|---|---:|---:|---:|---:|
| micro-F1 | 88.4808 | 88.5944 | **89.5640** | 88.7524 |
| macro-F1 | **56.3079** | 43.2430 | **57.8201** | 45.3217 |
| Precision | 94.4166 | 87.4440 | **95.4282** | 87.8622 |
| Recall | 83.2472 | **89.7755** | 84.3789 | 89.6609 |
| T1 joint micro-F1 | 80.6106 | 81.3342 | **82.5335** | 81.6593 |
| T2 joint micro-F1 | 93.3330 | 93.5127 | **93.9570** | 93.5253 |
| T3 joint micro-F1 | **62.3969** | 59.0247 | **64.7825** | 56.6290 |

44vs44的基线优势：micro +0.1136个百分点、T1 +0.7236、T2 +0.1797、Recall +6.5283；不足：macro -13.0650、T3 -3.3722、Precision -6.9726。HICD100轮与基线44轮不同进度，不用这个表声称等预算最终优势。

类别joint F1：

| 目标 | HICD44 | Bi-SRNet44 | HICD90 |
|---|---:|---:|---:|
| Farmland | 94.56 | 95.34 | 95.03 |
| Runway | 96.84 | 97.25 | 97.41 |
| Taxiway | 90.58 | 87.23 | 91.69 |
| Apron | 86.71 | 82.25 | 87.88 |
| Highway | 68.22 | 69.29 | 72.08 |
| Building | 80.61 | 81.33 | 82.53 |
| Aircraft | 65.03 | 61.54 | 66.99 |
| Vessel | 61.07 | 57.96 | 63.67 |

因此44轮基线在8类中4类joint F1更高、4类更低，不是逐类全面领先。所有P/R及定位诊断在target_comparison.csv，105组合在target_state_comparison.csv。

宏平均继续用既定2TP+FP+FN>0政策。44轮HICD有效联合类别29，基线36；额外无GT但有预测的组合计0。固定26个GT支持组合的**诊断macro**分别为HICD62.8050%、基线59.8749%，诊断仍显示HICD较高，但该数不替换正式macro，不据此更改选模。

## 3. 错误分解：先解决什么

下面是像素数，来自相同GT范围的联合矩阵；互斥三项前景错误相加等于联合FN，背景误报单列。

| 错误类型 | HICD44 | Bi-SRNet44 | HICD90 |
|---|---:|---:|---:|
| GT前景→预测背景 | 2,243,839 | **1,180,783** | 2,118,167 |
| GT前景→另一目标 | 38,494 | 60,727 | 29,564 |
| 目标正确、具体状态错误 | 154,758 | 245,889 | 124,736 |
| GT背景→预测前景 | **522,896** | 1,568,654 | **433,770** |

**已观察**：HICD90的联合FN中约93.2%是被判成背景；移动目标T3的FN中背景漏检占约64.9%，同目标错状态约23.8%，错目标约11.3%。基线44的T3背景漏检4,600px，而HICD44为23,399px，但基线整体背景误报约为HICD44的3.00倍。

**解释边界**：可以确定现有HICD具有更低误报、更低召回的行为；不能由这一观察直接证明BackgroundSuppressionLoss是唯一原因。阈值校准、Focal/Dice、归一化、骨干与冻结策略、预训练、特征融合和训练权重均是混杂因素。

**优化目标建议**：提高召回时监测背景FP是否失控；提高T3定位与状态联合成绩，而不是只降低loss或提高大面积农田micro。两方法损失公式及归一化不同，train_loss绝对大小不用于判定谁学得更好。

## 4. 代码诊断：事实、假设与优先级

### P0-A：DINO输入与预训练接口

已核实：dataset_v7.py:687–688仅将RGB除255；DINO wrapper在patch embedding前没有mean/std变换。该wrapper和dataset代码SHA与本run启动一致，属于实际训练路径，不是后来未使用的代码。

DINOv3官方对LVD权重要求mean=(0.485,0.456,0.406)、std=(0.229,0.224,0.225)，对SAT权重另给遥感均值/标准差。[官方输入变换](https://github.com/facebookresearch/dinov3#image-transforms)

待验证：本地model.safetensors的具体来源与预训练transform身份；当前仅文件名，没有在此轮找到配套config。不能在没有确认权重来源时盲选归一化常数。输入约定未对齐是明确接入问题候选，但提升幅度未测。

建议：将数据集输出继续保持RGB[0,1]，在DINO分支内部执行匹配该权重的Normalize；频域/未来原生RGB细节支路仍读取原始[0,1]，避免把全dataset直接归一化而暗改所有基线或DWT输入。记录backbone权重SHA、transform版本与预训练来源。

这些输入变更必须用于独立新训练；不能给已经在未Normalize输入上训练的旧90轮权重临时换输入分布，再把重评差值称为受控修复收益。旧run的测试与身份保持不变。

### P0-B：特征抽取在block前发生

已核实：dinov3.py:161–182先`extracted[i]=x`，再执行第i个block。配置[2,5,8,11]实际取得block1/4/7/10输出（均为零基），不是对应block2/5/8/11输出。最后block11及随后final norm的输出没有进入返回multi_scale（spatial_tokens变量未被后续投影使用）。

建议：对照当前安装timm的对应模型接口，采用after-block特征或经验证的forward_intermediates/get_intermediate_layers；不能将官方facebook参数名直接套到timm。逐层与reference前向比对，并验证最后可训练block获得梯度。[官方中间层接口参考](https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/vision_transformer.py)

边界：抽取偏移与末层未贡献是静态数据流事实；实际性能影响需独立训练对照。各层projection已有LayerNorm，不能声称所有特征完全没有norm；final norm未用不单独证明特征失效。

### P1：监督与误报—漏检取舍

已核实：HICD使用自适应权重Focal、2×target Dice、0.5×BackgroundSuppression、state CE+Dice、图有效性辅助；基线使用静态weighted target CE+2×state CE。baseline CE按权重和归一化，HICD Focal按有效像素数归一化，梯度尺度不相等。HICD还用val IoU更新loss权重；基线不这样反馈。

建议优先控制：建立同一DINO架构下的静态weighted CE对照，再单独恢复Dice、背景抑制、自适应策略；背景项做0/0.1/0.5的逐级对照，比较同权重类别下的R与背景FP。该系数列表是待讨论实验变量，不是本轮已修改配置。新的权重统计仅来自train，开发val用于选模；先冻结一份对照，禁止同步换骨干/解码器/采样。

### P1：双时相证据被过早混合

已核实：当前HICD_v7.py中`fused_features, _, _ = self.tfm(...)`，解码器只有fused_features；TFM保留的独立t1/t2输出被丢弃。频域支路保留变化信息，但主语义路径没有显式保留pre/post/change三路。基线则将SR/CotSR后的pre/post与resCD change共同送入融合头。

假设：目标定位需要存在性/外观，reduced/added/replaced等状态需要有方向的时序证据；早期混合可能限制状态判断。需要与simple concat和显式abs/signed difference同预算消融，不先假定attention模块一定更好。

### P1：同patch网格构造的金字塔

已核实：256输入、patch16原生16×16；四层Transformer特征都来自同一16×16网格。C2/C3是插值至64/32，C5池化至8；这是一种同分辨率多深度特征的尺度适配，不能称已有原生stride4纹理。后续NFA/频域会进一步学习局部信息，因此也不能简单断言“输出没有细节”。

假设：原生RGB stride4细节支路与DINO语义的残差融合可能提高小目标召回。应优先做轻量卷积stem，而非直接增大输入/骨干；以T3和背景FP为验证依据。当前HICD飞机/船定位F1仍高于基线，不能从模型类型 alone 认定DINO不适用。

### P2：图约束与标签语义

已核实：当前VALID_COMBINATIONS仍把Farmland/reduced记合法，与本项目“农田仅不变”的新语义不同。预测端强制Farm0已避免该冲突直接污染正式Farm评分，但图辅助仍在学习旧合法表。

CPU读取best90与macro-best56的GNN矩阵：25个验证有支持、非农田组合的最小有效性均1.0，低于0.05门限的组合0个。Farm0/Farm2均1.0。**这两个checkpoint没有看到图门控误删这些监督**；不把存在门控逻辑扩大成已证实退化原因。训练早期门控变化尚未记录，本轮没有补造早期历史。

建议：合法关系从显式任务schema生成，含Farmland仅state0、背景none、T4忽略；先将GNN作为可拔插软先验，与无图/固定先验分别消融。不能用val/test某组合恰好缺失推导其“非法”，不能隐藏纯FP组合。

实例投票目前不参与训练、正式指标采用raw支路；移除其调用主要影响运行开销，不应作为提高raw精度的解释。先保持频域分支为可消融项，不声称此前修复optimizer就证明它已提升T3。

## 5. 推荐重构方向（设计建议，未实施）

先保留DINO作为语义骨干，完成可单独验证的接入修正，再建立一个更简单且可解释的像素T1–T3核心：

1. **语义与细节分工**：DINO提供深层语义；共享的轻量RGB stem提供原生stride4细节。仅在模型内部为各分支使用各自输入约定，保持共享数据协议。
2. **保留时序方向**：各尺度保留F_pre、F_post，并构造绝对差与有符号差；目标分支侧重目标存在性，状态分支读取完整时序。先比较concat/差分，再决定是否需要cross-attention。
3. **目标条件状态**：以共享视觉特征+目标embedding构造p(s|t)，避免一个全局7态分类器独自混合所有目标。建议先比较共享条件头与仅T3使用独立adapter，保持输出全局ID与错误计分。
4. **监督清晰化**：以target CE和conditional state CE为核心。联合NLL `-log p(t_gt)-log p(s_gt|t_gt)`等价于相应两项CE，不另加一个数值重复的“joint loss”制造无意义loss叠加。农田保持定义上的固定不变，GT仅用于训练监督，推理不输入GT目标/状态。
5. **图/频域作为后加项**：基础核心稳定后分别测试静态合法先验、GNN软先验、频域支路；只有可重复提升时保留，并说明提升在哪类/哪种错误。

输出约束的公平性：如果新头改为只输出schema合法联合类，可能减少纯FP组合而提高macro；必须同时给控制模型施加相同预测约束，单独报告约束贡献。第一版建议保留16/7全局目标—状态错误空间，不通过删列/输出mask隐去错预测。现有strict evaluator不修改。

## 6. 分阶段实验矩阵

独立变量逐项改变，所有正式对照固定train/val/test版本、raw64、增强/采样、Farm语义、有效batch32、100轮cosine周期、初始化来源与seed列表。优化器学习率分组/冻结块数须显式记录；当前HICD骨干LR0.1×、前6块与patch embed冻结，而基线ResNet常规LR，不能以“基础LR相同”宣称优化完全一致。

| 阶段/编号 | 单项变化或控制 | 要回答的问题 | 继续条件 |
|---|---|---|---|
| A0 | P0输入约定与after-block的接口/梯度核验；不训练正式run | 权重来源、归一化、层输出和梯度路径正确吗 | 与对应reference接口一致，末层梯度有效；无旧源码覆盖 |
| A1 | 旧架构+正确权重归一化；其余不变 | 输入接入是否限制能力 | 记录micro/macro/T3、R与背景FP，不预设改善 |
| A2 | A1+after-block抽取；其余不变 | 特征偏移/末层缺失是否影响能力 | 单独估计抽取修正贡献；历史v12保留 |
| B0 | A2架构+静态weighted CE（匹配基线监督约定） | 复杂loss是否限制召回 | 对照相同归一化、类别权重、优化器；必要时校准系数 |
| B1 | B0+target Dice | Dice的独立贡献 | 不把B0→B1归因骨干 |
| B2 | B1+背景抑制0/0.1/0.5控制 | 降误报的召回代价 | 同时报背景FP、GT→bg、P/R；不只看micro |
| C0 | 最佳固定监督控制+pre/post/abs/signed时序保留 | 状态是否受早混合限制 | 关注T3、replaced，且P/R取舍可接受 |
| C1 | C0+原生stride4 RGB细节stem | 小目标定位召回能否提升 | 类别尺度分组分析；背景FP不能失控 |
| C2 | C1+目标条件状态或T3 adapter（二选一逐项） | 能否改善T3具体状态 | 目标正确下的状态错误与完整joint都改善 |
| D | 固定schema先验/GNN/频域逐个加回 | 每个复杂模块是否值得保留 | 跨seed稳定收益或可解释类别收益 |

这是候选矩阵，不是全部任务已启动。先完成A0；A1/A2用于拆分接入问题，B/C分支仅在前阶段证据不足或明确需要时推进，避免一次排满许多100轮。短工程smoke不当正式成绩；若做100轮schedule内的早期观察，明确是截断训练，不混成独立短cosine公平结果。

### 预先固定选择和验收

- 旧run继续保留micro主best90与macro-best56，原分数/权重不改。当前基线按其100轮既定计划完成，等待同split test后再做最终方法比较。
- 新方法的建议主目标：macro与T3联合表现；是否以macro选择主best应在启动前明确，所有控制/基线按同一选择规则比较，不能每个方法挑更有利的标准。现有基线同时存macro-best可供后续预先统一规则的补充研究。
- 候选验收阈值（供决策，不是结果承诺）：相对经接入修正的固定控制，macro及T3各提高至少2个百分点，micro下降不超过0.5个百分点；同时监测Recall和背景FP。该阈值尚未获得用户确认，不作为既有实验规则。
- 筛选后对控制与最终候选至少3个相同seed重复，报告均值/离散度；置信区间若使用bootstrap应按母图/场景组抽样，不能把数千万相关像素当独立样本制造显著性。
- 只用val筛选；最终固定候选一次test，不能根据已看过test反复设计状态采样/类别阈值。当前test曾被查看，任何重构仍属于研究开发环境；额外新场景测试若可获得，另行建立，不擅改现有split。
- GPU启动前重新确认基线进程/结束状态；当前训练PID366723仍在运行。本轮没有GPU新任务，CPU GNN探针仅读取旧权重。

## 7. 输出、执行与下一步

当前产物目录：`docs/experiments/method_reconstruction_20261012/`。

- diagnosis_and_plan.md：本诊断和候选方案。
- comparison.json：冻结的44/45与HICD44/90/56比较、误差分解、来源hash和代码hash。
- target_comparison.csv / target_state_comparison.csv：全部目标/状态P/R/F1与支持。
- gnn_validity_probe.json：56/90轮CPU有效性矩阵，不是训练/新推理结果。
- analyze_comparison.py：可重跑的只读CPU指标分析；运行会更新同目录衍生比较，不修改run或原始数据。

模型实现、参数选择及新训练入口尚未建立。未来实施第一步应是A0的权重/输入/特征接口核验，而不是先把所有模块换掉；再依照逐项控制结果决定保留DINO、加入细节/时序分工或扩大编码器重构范围。当前范围只完成方案，后续实作按用户下一次明确任务推进。
