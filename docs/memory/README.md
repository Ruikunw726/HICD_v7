# HICD 持久项目记忆

整理日期：2026-10-02（Asia/Shanghai）。来源：用户要求接管与分类的 `../HANDOVER.md`，交接记录截至 2026-09-28；基线部分截至 2026-09-18。这里保存可跨会话读取的项目文件记忆，未宣称写入应用级全局记忆。

## 读取顺序

- [项目背景与核实快照](project_context.md)：路径、任务划分、历史实验、当前观察。
- [决策与工作偏好](decisions.md)：按时间区分的研究决策与报告习惯。
- [冲突与待核查事项](risks_and_questions.md)：执行或用于论文前必须厘清的差异。
- [训练与方法](workstreams/01_training_and_method.md)
- [多任务基线](workstreams/02_baselines.md)
- [T4 实例分支](workstreams/03_instances.md)
- [数据与评估](workstreams/04_data_and_evaluation.md)
- [论文与研究论证](workstreams/05_paper.md)
- [实验管理与工作报告](workstreams/06_project_operations.md)
- [会话注册表](threads.json)：项目、实际会话 ID、接手完成状态与清单消息 ID。
- [2026-10-02 接收回执及最新核查汇总](intake_receipt_2026-10-02.md)：六条工作线均已接手并回复清单，续作优先读取。
- [2026-10-08 严格目标—状态协议](../strict_joint_protocol.md)：用户要求的新主P/R/F1口径；目标和具体状态同时正确，定位/状态诊断分列。实现和27项CPU测试已通过，训练状态见01与project_context。

## 记忆维护规则

每项事实标注“交接记录 / 已核实 / 待核实”，保留日期与文件来源。实验数字需要实验名、数据组成、split、任务范围、协议、checkpoint 与代码版本；缺失则标注未知。观察结果只对已核查范围成立。历史方案可保留，但不能覆盖较新决策或当前代码事实。

主会话负责公共记忆与跨线汇总；专题会话维护各自文件。首次接手以用户可见清单为交付，清单应写明负责内容、已核查事实、遗留问题、优先级、下一步和依赖。接手完成不等于训练、实验或论文已经完成。

