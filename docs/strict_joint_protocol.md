# HICD 严格目标—状态评估协议 v1

冻结日期：2026-10-08（Asia/Shanghai）。版本：`hicd.strict_joint.v1`。

本协议实现用户的要求：“严格的什么类别就是什么类别”“类别 F1 也按照目标、状态全对才算对”。主成绩是前景目标与原始具体状态同时正确的像素 F1。没有 D/ND 合并，没有 0.3/0.7 加权，没有把旧 BDD 分数改名为 strict。

## 1. 标签与评估单位

单位为 **pixel**；本协议不计算实例匹配、实例检测 AP 或实例 F1。

| target ID | 名称 | 任务 |
|---:|---|---|
| 0 | Farmland | T2 |
| 1 | Runway | T2 |
| 2 | Taxiway | T2 |
| 3 | Apron | T2 |
| 4 | Bridge | T4 |
| 5 | Highway | T2 |
| 6 | Building | T1 |
| 7 | Shelter | T4 |
| 8 | Tower | 不属于当前四任务；若有有效 GT/预测仍计入全前景 |
| 9 | Pier | T4 |
| 10 | Dock | T4 |
| 11 | Tank | T4 |
| 12 | Aircraft | T3 |
| 13 | Vessel | T3 |
| 14 | Crater | T4 |
| 15 | Background | 仅诊断，排除主分 |

| state ID | 原始状态名称 |
|---:|---|
| 0 | no-change |
| 1 | damaged |
| 2 | reduced |
| 3 | added |
| 4 | extended |
| 5 | replaced |
| 6 | none |

输入必须为已解码的整数 `pred_target, pred_state, gt_target, gt_state`，形状一致，ID 语义固定如上。不能直接把 TIFF 的原始值 0–68 当 target ID，也不能将单任务 local ID 当全局 target ID。原始编码先按明确的数据配置解码，配置与名单版本必须随实验保存。协议精确区分 **解码后的 target-state 对**；若多个原始编码经上游解码成同一对，本模块不能恢复其原始编码差异。

定义所有 15×7=105 个前景联合类别，联合 ID 为 `target_id*7+state_id`。第 106 个混淆矩阵 bin（ID=105）为背景。类别清单不依据 batch 中有没有 GT 而改变，也不采用手工“合法状态组合”删掉错误预测。

当前数据配置 `raw 0 -> target 15, state 6`，即 none 是背景的状态。预测 target=15 时不论 state 为何都视为背景；背景没有需要识别的前景状态。**预测 target 为前景、state=none(6) 时仍是一个前景联合预测**，一般对真实非 none 状态造成 FP+FN，绝不据此剔除像素。若外部有效 GT 明确为前景+none，也按字面精确评估；GT none 不是 ignore。GT 状态含义不完整/缺字段时应报错，不能伪造 no-change 或 none。

## 2. 唯一的有效像素掩码

`valid = (gt_target != 255) & (gt_state != 255)`。

GT 任一分量 ignore 则该像素同时排除于主联合指标、定位诊断、状态诊断、背景诊断与混淆摘要。有效像素上 target 只能为 0–15、state 只能为 0–6；非法 GT/预测（含预测 255）、浮点标签和形状不一致均报错，不通过 mask 悄悄丢弃。GT ignore 区的预测无需合法，因为该区域没有监督，且不会贡献任何统计。

不按预测 target、预测 state、任务归属、当前类是否有 GT 或 `state_ignore_targets` 再筛像素。配置中的 `state_ignore_targets=[0]` 属于历史 loss 策略，不允许因此从 strict 评估删除 Farmland。

## 3. 计数与主分

流式联合整数混淆矩阵 `Q[GT, prediction]`。对于前景联合类别 c：

- `TP_c = Q[c,c]`。
- `FP_c = sum(Q[:,c]) - TP_c`。
- `FN_c = sum(Q[c,:]) - TP_c`。
- `support_c = TP_c + FN_c`；`predicted_support_c = TP_c + FP_c`。
- `precision_c = TP_c/(TP_c+FP_c)`；`recall_c = TP_c/(TP_c+FN_c)`。
- `F1_c = 2 TP_c/(2 TP_c+FP_c+FN_c)`。

同目标的状态预测错误会同时给真实联合类 FN、预测联合类 FP。错目标也一样。背景上的假前景、其他任务目标上的错预测均计入预测联合类 FP；前景预测为背景产生真实联合类 FN，不给前景类别增加 FP。背景 TP 不贡献主分，因此大量正确背景不能冲高全前景 F1。

**全前景主指标 `strict_joint_f1`**：对全部 105 个前景联合类分别统计，再求 TP/FP/FN 总和计算 **micro F1**。不是联合 accuracy，不是各任务均值，不是按 batch 平均 F1。

**各任务 micro**：选择该任务的联合类别后求和 TP/FP/FN，统计仍来自全体有效像素。跨任务误预测给真实任务 FN、预测任务 FP；不能只截取 GT 属于该任务的区域。各任务的分母不同，不能把任务 F1 简单平均当全局主分。

**macro**：对相应范围内 `support+predicted_support > 0` 的联合类 F1 等权平均。GT 和预测都没有的空类 F1=null，不计入 macro；无 GT 但有 FP 的类 F1=0，必须计入 macro。所有类均空时 micro、macro 为 null，不强制为 0 或 1。单状态完美类别的 micro/macro 均为 1。

单个 precision 或 recall 的分母为零则该比率返回 JSON null；例如仅 FP 类 precision=0、recall=null、F1=0。输出保留整数计数与 support，读者可以区分“没有监督”和“预测失败”。不输出 NaN/Infinity。

## 4. 每目标与具体状态诊断

每目标的 **主类别 P/R/F1** 来自该目标七个状态联合类的 TP/FP/FN 求和；因此目标正确但状态错误仍是 FP+FN。

每目标并列两个诊断：

1. **`target_localization`**：只判断 target 是否正确，跨该目标七个状态的混淆子矩阵均算定位 TP；所有有效区域的其他 target→该 target 算 FP，该 target→其他 target 算 FN。它不是主类别 F1。
2. **`state_on_gt_target`**：在有效 GT 属于该目标的全部像素上，独立评估七状态分类；完全不按预测 target 正确与否筛选，预测背景和预测 none 的像素也保留。输出每状态 P/R/F1/support、micro、macro、样本像素数及完整 7×7 状态矩阵。该诊断用于识别状态头质量，即使定位全错，状态诊断仍可能高；不能将其冒充完整系统成绩。

状态诊断需要独立保存每目标 7×7 矩阵：联合矩阵把预测背景压成一个 bin，无法恢复背景预测原来选的状态。实现保留这部分信息，但不保留逐像素预测。

| JSON 路径/字段 | 含义 |
|---|---|
| `strict_joint_f1` | 全前景精确 target+state micro F1，主分 |
| `strict_joint_micro` | 全前景 P/R/F1、TP/FP/FN、support、predicted_support |
| `strict_joint_macro_f1` | 全前景活跃联合类等权 F1 |
| `tasks.T1..T4.strict_joint_micro` / `strict_joint_f1` / `strict_joint_macro_f1` | 各任务联合 micro/macro |
| `per_target.<ID>.strict_joint_precision/recall/f1` | 每目标主类别 P/R/F1 |
| `per_target.<ID>.strict_joint_micro` | 同一主类别的整数计数和 support |
| `per_target.<ID>.target_localization_precision/recall/f1` | 仅目标定位 P/R/F1，诊断 |
| `per_target.<ID>.target_localization` | 定位整数计数及 support |
| `per_target.<ID>.state_on_gt_target.micro/macro_f1/per_state` | GT 目标区域内状态诊断，含每状态 P/R/F1/support |
| `per_target_state[]` | 全部 105 行，每 target-state 的独立 P/R/F1、计数、support 和 evaluable 标记 |
| `joint_confusion.nonzero_entries[]` | 联合矩阵的全部非零项，GT/预测联合 ID 与 count；可定位具体状态、错目标和背景错误 |
| `background` | 背景诊断；不进入前景 micro/macro |
| `valid_pixels/ignored_pixels/updates` | 覆盖范围及流式更新次数 |
| `prediction_policy` | 模型包装器额外记录支路、GT/评估网格、样本数 |

`per_target` 的键为字符串 ID，不依赖名称排序。联合矩阵和状态矩阵均为 GT 行、预测列。完整矩阵包括无 GT 但有预测的错误类别。

## 5. 可直接训练调用的 opt-in 包装器

```python
from evaluation.strict_joint_evaluator import StrictJointModelEvaluator

strict = StrictJointModelEvaluator(dataset_config)  # 默认 raw state_logits
strict.reset()
for predictions, targets in validation_batches:
    # predictions: target_logits [B,16,H,W], state_logits [B,7,H,W]
    # targets: target_gt [B,h,w], change_state 或 state_gt [B,h,w]
    strict.update(predictions, targets)
scores = strict.compute()
print(strict.format_results(scores))
primary_f1 = scores['strict_joint_f1']
```

包装器同时接受原评估调用的第三参数 `loss_dict=None`（不参与指标）。`reset()` 清零本轮统计，`compute()` 不清零。只有显式创建此类才启用新协议；旧 `ICDEvaluatorV6`、`bdd_protocol.py` 和 `evaluation/__init__.py` 不作默认替换。

默认使用 **raw `state_logits`**，与现训练主 BDD 的状态支路一致。GT 用 nearest 直接对齐到 target/state logits 的实际 H×W，通常 256→64；不硬编码 0.25，也不把 logits 上采样后宣称成原来的分辨率。若训练入口已准备 64×64 GT，包装器不再缩放，记录的是所收到的输入网格。只有显式 `state_head='refined'` 才改用 `refined_state_logits`，报告会标明支路；raw/refined 必须分别比较。非有限 logits、缺 GT、支路缺失或 target/state 输出尺寸不一致均报错。

空间采样会改变小目标/ignore 像素覆盖。跨实验必须同时对齐 **GT 解码、名单、输出分辨率、nearest 采样、raw/refined 支路**；仅协议版本相同不足以证明实验完全可比。v1 冻结指标语义，不声称不同网格的分数相同。

纯 NumPy 核心 `StrictJointEvaluator` 也可直接调用 `update(pred_target, pred_state, gt_target, gt_state)`；支持 `merge(other)` 汇总分片，禁止自合并。内存仅保留 106×106 int64 联合矩阵及 15×7×7 int64 状态矩阵（95,768 字节），外加当前批次临时数组。不能对各 batch 的 F1 求平均替代矩阵累计。

## 6. CPU 缓存重评入口

`script/eval_strict_joint.py` 直接导入 NumPy 核心，避开已有 evaluation 包初始化中的 torch/scipy；无需模型、PyTorch 或 GPU。每个 NPZ 分片必须包含四个同形状整数数组：`pred_target, pred_state, gt_target, gt_state`，GT 与预测已经按同一网格对齐。

```powershell
F:/python/python.exe F:/mambacd/home/HICD_v7/script/eval_strict_joint.py `
  --predictions F:/mambacd/home/HICD_v7/prediction_cache/v11_test `
  --split test `
  --provenance F:/mambacd/home/HICD_v7/prediction_cache/v11_test_metadata.json `
  --output F:/mambacd/home/HICD_v7/reports/v11_test_strict_v1.json
```

上述 cache/metadata/output 是未来重评的示例路径，本轮未创建，也不代表已经有真实预测。`--provenance` 可选，建议包含 checkpoint 路径/哈希、配置/名单哈希、GT 解码与 mask、支路、GT/输出网格、代码版本、预测生产方式和完成样本数。文件夹按排序读取顶层 *.npz，可给多个路径；拒绝重复文件路径、空目录、缺字段与覆写已有输出。每个源文件记录 SHA256，检查重评期间没有改变。

入口只保证提供分片的统计，**不自动证明完整覆盖 test、样本无重复或缓存 GT 解码正确**。producer provenance 单独存放，不得覆盖计算字段。NPZ 是按文件解压，建议每 batch 一个分片；单个包含全量数组的大 NPZ 仍会占用该文件解压内存。

当前交付为训练包装器和 CPU 缓存重评入口，尚未提供独立 checkpoint 推理脚本。权重重评应按 checkpoint args 与配置快照建模、严格恢复动态 buffer 和模型键，在资源空闲时进行；不能照搬旧 `strict=False`/硬编码模型的脚本。本次没有加载真实权重或产生真实 strict 模型分数。

## 7. 验收反例与测试

4 像素全 Building，GT 状态 `[0,3,1,2]`，预测 `[3,0,2,1]`：旧 BDD 在 ND/D 折内都正确，可得到 F_overall=1；strict v1 必须 **TP=0、FP=4、FN=4，micro=macro=0，Building 定位 F1=1**。这说明旧汇总分数不能反推出新分数。

另一个固定例：

```text
GT target   = [6,6,6,15,12,15,255]
GT state    = [3,1,0, 6, 5, 6,  1]
pred target = [6,6,6, 6, 6,15,  6]
pred state  = [0,1,6, 3, 5, 0,  0]
```

ignore 后全局 TP=1、FP=4、FN=3，micro=2/9、macro=1/6；T1 micro=1/4、macro=1/5；T3 F1=0；Building 定位 F1=3/4。错误的前景 none 必须占一个 FP。

测试包括上述固定数值、错目标但状态正确、目标正确但状态全错、背景 FP、漏检、GT ignore、pred none、显式 GT none、replaced、单状态完美、仅 FP 无 GT、空类、全背景、Farmland loss-ignore 不影响评估、Tower 的全局/任务边界、schema 不完整/错序拒绝、随机 tuple-counter 参照、跨批次累计/merge、一致性及 CPU CLI 实际分片/覆写拒绝；模型包装器测试支路、nearest 网格、状态诊断不筛定位和非有限输入。

```powershell
Set-Location F:/mambacd/home/HICD_v7
F:/python/python.exe -m unittest discover -s tests -p test_strict_joint_protocol.py -v
```

2026-10-08 本地 Windows Python 3.14.6、NumPy 2.4.4：**27 项通过，无跳过**。测试数组均为数值验收数据，不是模型 test 成绩。

## 8. 历史结果和当前数据限制

2026-10-08 展示更新：控制台指标与新训练CSV的浮点字段固定四位小数；整数计数仍为整数，空指标显示n/a。控制台仅展示主分、任务micro/macro及非空目标的joint P/R/F1、loc F1、GT区域state F1；仅有FP的目标保留，完全空目标省略。核心计算、best选择、checkpoint分数和完整metrics JSON保留原精度。

已经运行的Python训练进程不会自动重新载入格式函数。当前 `v12_strict_joint_t123_trainfix_20261008` 使用独立CPU报告器 `script/report_strict_epochs.py` 读取已完整写出的metrics/epoch_NNN.json，将相同评估以四位小数追加到 `evaluation_summary.log`，不重启训练、不改原训练日志/CSV。报告器绑定原训练PID退出；后续新启动的训练器直接使用新的简表格式。

持久报告器使用独立screen，启动入口为 `bash launch_strict_summary.sh <实验绝对目录>`（WSL内）。入口校验该实验的训练PID、同run既有报告器，以及新screen/Python初始化是否存活；不把screen fork成功当作已稳定启动。2026-10-09已修复临时exec报告器退出导致epoch10后停更的问题，并从完整JSON补齐遗漏；启动与重复补写不会重复已有完整epoch block。

原训练 CSV 中的 `F_loc/F_clf/F_overall/F_macro/bdd_f1/bdd_sek/T*_f1` 仍是 **legacy**，其字段与历史日志保持原样；不从旧分数、单独 target/state 边际混淆矩阵反推严格联合分数。必须获得目标—状态共同预测或实际重新推理。新 strict 日志应另存新实验/评估文件，并记录协议版本。

v11 实际配置仍把大多数 T4 target 映射为 255，只有 Bridge 保留；它的像素 T4 strict 不代表六类实例系统。GT 被 ignore 的目标无法由新指标补回，若模型在其他有效区域预测这些类，仍计 FP。像素协议不代替实例评估。2026-10-02 已发现 train/test 同名交集 197、val/test 341；严格指标解决评分语义，未消除数据独立性问题。

v1 的任何标签 ID、有效范围、none/background、聚合、空类政策变动须另建版本；不得在同一协议名下默改。raw/refined、GT 解码与网格等实验条件必须随输出记录，不可仅凭 strict 字段名比较。
