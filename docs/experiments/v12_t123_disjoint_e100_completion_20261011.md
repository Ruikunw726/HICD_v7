# T1–T3 恢复划分后100轮训练完成记录

记录日期：2026-10-11，Asia/Shanghai。实验：`v12_strict_joint_t123_disjoint_e100_20261009`。状态：100/100轮训练及验证完成。本报告16:27训练核查时尚未测试；随后用户授权的第90轮完整test已于16:41完成，micro89.9953%、macro62.2192%，详见同目录`v12_t123_epoch90_test_20261011.md`。下文“尚未测试/下一步测试”保留为训练核查时点的历史状态。

## 完成与来源

- 开始：2026-10-09 17:11:55 +08:00；结束：2026-10-11 06:13:10 +08:00；总历时37小时1分15秒。`status.txt`记录python_exit=0、tee_exit=0，日志明确TRAINING_COMPLETED、best_epoch=90。
- 本轮只读核查：无本实验train_strict_v7/train_full_v7/eval_epoch50_strict_test进程；nvidia-smi计算进程列表为空。进程结论仅为查询时点快照；审计产物记录时间为16:27 +08:00。
- 主证据：实验目录的manifest.json、status.txt、train.log、metrics/epoch_001.json至epoch_100.json与保存的检查点。
- 独立审计：`checkpoints/v12_strict_joint_t123_disjoint_e100_20261009/completion_review_20261011/summary.json`；可重跑脚本为同目录audit_completion.py。100轮完整序列、200份train/val报告的106×106计数复算通过，micro/macro与原JSON精确一致，TP/FP/FN与像素计数闭合，loss有限；日志未发现指定Traceback、CUDA OOM或非有限loss致命标记。
- 检查点元数据读回：best_model=第90轮、best_macro=第56轮、编号50/100及latest=各自正确轮次；5份checkpoint的run/split/协议/预测语义身份和分数对应正确，scheduler last_epoch及T_max=100正确。未计算大权重文件的内容SHA；本审计不等同于重新推理或全参数逐项数值扫描。
- 新数据root下manifest登记的names/catalog/split manifest/scene信息实时SHA均与训练开始相同；代码版本由原run的git_head、git_diff、snapshot和文件hash共同追溯，不能只用Git HEAD代表有未提交改动的实现。

## 实际协议与配置

| 项目 | 实际值 |
|---|---|
| 数据根 | D:/CD/0617final_256_v2_disjoint_20261009 |
| 数据组成 | train 61,029；val 13,077；test 13,077；真实样本原始成员恢复 |
| 训练计分样本 | 每轮61,024：batch32且train drop_last=True；尾部5张随shuffle变化，非固定剔除 |
| 验证计分样本 | 每轮完整13,077，409个batch |
| 范围 | T1 Building；T2 Farmland/Runway/Taxiway/Apron/Highway；T3 Aircraft/Vessel |
| 忽略 | 全部T4及Tower GT忽略；背景上的错误前景仍计FP |
| 主协议 | hicd.strict_joint.v1；目标与具体原始状态均正确才计TP |
| 农田 | hicd.predicted_farmland_nochange.v1；依据预测target=Farmland固定state0，非依据GT解码 |
| 网格 | 输入256×256；原raw state头；nearest GT对齐64×64计分 |
| 初始化 | DINOv3 L1预训练骨干；任务权重/AdamW/scheduler/AMP从零，无resume |
| 预算 | 100轮；初始LR1e-4；weight_decay0.0001；CosineAnnealingLR T_max100 |
| 其他 | batch32；AMP；workers4；state-aware；freeze_backbone_epochs0；无稀有类过采样 |
| 主best | 验证全前景strict joint micro最大，保存best_model.pth |
| 独立次best | 验证strict joint macro最大，保存best_macro.pth |

run_identity：`d1584ee10d43b2df6e2b2e6922535cb2451a9f44941bedd8bcb23dba4ea93d75`。

data_split_identity：`9f66845372df4c8730a543ac2899575b8ba40eff31b676355b183991e09db05b`。

## 验证集主要结果

下表均为同一新划分、同一农田语义、同一严格联合协议，单位%。每列是一份实际模型，不能跨列拼接分数。

| 指标 | 第50轮 | micro-best第90轮 | macro-best第56轮 | 最终第100轮 |
|---|---:|---:|---:|---:|
| micro-F1 | 87.9534 | **89.5640** | 88.7501 | 89.4294 |
| macro-F1 | 59.2885 | 57.8201 | **62.2611** | 57.8068 |
| micro Precision | 95.3467 | 95.4282 | 94.8985 | 95.6196 |
| micro Recall | 81.6241 | 84.3789 | 83.3499 | 83.9919 |
| T1 micro-F1 | 79.7360 | 82.5335 | 81.8210 | 82.1282 |
| T2 micro-F1 | 92.9319 | 93.9570 | 93.1016 | 93.9459 |
| T3 micro-F1 | 64.6415 | 64.7825 | 63.8118 | 64.4563 |
| macro参与联合类别数 | 28 | 29 | 27 | 29 |

T4没有有效GT/预测支持，成绩为N/A，不能写成0或归入能力结论。全量精确值及P/R/TP/FP/FN/support见审计summary.json。

### 增加轮数的实际收益

- 同一run的前50轮micro峰值是第49轮88.5657%；全100轮峰值是第90轮89.5640%，提高**0.9984个百分点**。
- 前50轮macro峰值是第39轮60.2523%；全100轮峰值是第56轮62.2611%，提高**2.0088个百分点**。第56轮已保存独立macro-best。
- 单看第50→100轮：micro提高1.4760个百分点，T1提高2.3922个百分点，T2提高1.0140个百分点；T3降低0.1852个百分点，macro降低1.4817个百分点。后期改善主要体现在T1/T2与整体召回，不能称所有类别/状态均受益。
- 最后10轮micro范围89.3132%–89.5267%，macro范围57.7256%–60.1227%；最终micro比best90低0.1347个百分点。出现平台迹象，但单次训练不能证明继续延长预算无收益。
- 这是100轮cosine计划内前后阶段观察，不是一个独立50轮cosine训练与100轮训练的随机重复因果对照。
- 原旧划分50轮val（农田复算）micro87.8628%、macro61.8562%，旧test88.6116%/68.2704%保留为历史。新旧split及学习率周期不同，禁止把旧test与新val相减当作方法提升；旧共享条目已撤回，旧test不是本实验test。

### 目标类别诊断

严格联合F1，单位%。

| 目标 | micro-best第90轮 | macro-best第56轮 | 最终第100轮 |
|---|---:|---:|---:|
| Farmland | 95.03 | 94.14 | 95.01 |
| Runway | 97.41 | 97.25 | 97.33 |
| Taxiway | 91.69 | 91.44 | 91.80 |
| Apron | 87.88 | 87.51 | 87.82 |
| Highway | 72.08 | 69.95 | 72.21 |
| Building | 82.53 | 81.82 | 82.13 |
| Aircraft | 66.99 | 63.29 | 68.31 |
| Vessel | 63.67 | 64.09 | 62.48 |

第90轮micro P95.43%而R84.38%，主要还有漏检空间：Building R75.12%、Highway R60.24%、Aircraft R60.25%、Vessel R55.86%。农田按项目定义固定不变，其高分主要反映定位，不能作为农田损毁识别证据。

### 小类与状态瓶颈

| 联合类别 | GT像素数 | 第50轮F1 | macro-best56 F1 | micro-best90 F1 | 第100轮F1 |
|---|---:|---:|---:|---:|---:|
| Aircraft / replaced | 694 | 10.71% | 22.38% | 12.32% | 7.22% |
| Vessel / replaced | 5,957 | 10.16% | 21.44% | 4.83% | 4.39% |

macro-best56对上述两类更好，但仍较低；后期整体micro上升没有转化成替换状态改善。Vessel/damaged仅12个GT像素，以上选定轮次F1均0，支持太小，不应单独做稳健泛化主张。

第100轮训练micro91.2741%、macro76.3205%，验证micro89.4294%、macro57.8068%；训练T3 micro90.93%、验证64.46%。不同集合的macro参与类别和support不同，不能把macro差直接当严格的同支持配对效应，但T3训练与验证差距提示泛化仍弱。未开展因果消融，不能认定唯一原因。

### Macro口径边界

当前既定协议：联合类别满足`2TP+FP+FN>0`便参与macro；GT与预测均无支持者为N/A，GT无支持但被误预测者按F1=0参与。这是既有规则，本轮未更改。

验证集中GT非零支持的联合组合恒为26个；第56轮macro共纳入27个（另1个只有误预测的组合），第90/100轮各29个（另3个）。因此macro变化包含真实类别识别变化，也包含误预测到无GT组合产生的零分与分母变化。

为解释这一点，审计另列**仅固定26个GT有支持组合的诊断macro**：第56轮64.6557%、第90轮64.4916%、第100轮64.4768%。它排除了无GT组合的额外惩罚，**不能替换官方macro或用来重选本run best**。后续正式基线必须明确并共用类别/N/A/背景规则。

## 记录产物与后续

- 本文：完成报告、协议、结果与边界。
- 实验台账：docs/experiment_log.md更新为已完成，并保留启动/途中检查的历史快照。
- 专题记忆：docs/memory/workstreams/01_training_and_method.md追加10/11完成状态；公共记忆与其他专题文件未改。
- completion_review_20261011/summary.json：全精度汇总、数据身份、100份源指标hash、5份checkpoint元数据与验收。
- completion_review_20261011/epoch_history.csv：100轮全精度train/val/loss/T1–T3与macro支持诊断；原训练CSV未改。
- 同目录epoch_050/056/090/100_target_state.csv：对应全部105个联合类别的P/R/F1与支持计数；原指标/权重未覆盖。

本次只读分析既有模型并保存记录，没有新GPU推理/训练。下一步评估建议固定best_model90作为既定主选模结果、best_macro56作为预先声明的补充模型，在同一13077张test按相同协议各评一次，不根据test更换选择规则；测试执行结果另记。

验证集既用于选best，也向自适应loss权重反馈，属于开发集。本实验的新test尚未推理；恢复test也是此前旧test的子集，并非新外部盲测。此前已核查同名/母样本tile/pre路径跨split互斥，但城市前缀共享、未做完整内容/地理footprint审计、单次运行和未齐备同协议基线/消融等限制继续保留。
