# v12 完成与测试结果的仓库快照

导出日期：2026-10-12（Asia/Shanghai），原始run为`v12_strict_joint_t123_disjoint_e100_20261009`。这些文件从已核验run复制，未改变分数，便于不携带大checkpoint时查阅结果。

- `v12_e100_completion.json`：100轮完成核查、micro-best90/macro-best56/50/100全精度验证结果、训练配置/身份/检查点元数据，以及100个原指标文件hash。其observed_at在test之前，`new_test_evaluated=false`是当时快照，后续test由下列文件记录。
- `v12_epoch90_test_metrics.json`：13,077张完整测试的raw/正式Farm不变decoded计数、各任务/目标/状态P/R/F1和分组诊断；正式micro89.995316%、macro62.219187%。
- `v12_epoch90_test_verification.json`：矩阵独立复算、样本/像素闭合、同名交集0、Farm解码及评估来源hash。

原始权重、图像、数据catalog、大训练日志和全部逐轮JSON不打包在Git。数据/模型身份、SHA和本机来源路径保留供追溯；复制JSON不意味着在另一环境重新推理已成功。新v13仍待基线完成后启动，这里没有其性能结果。
