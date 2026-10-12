# 内层主模型仓库更新记录

日期：2026-10-12，Asia/Shanghai。用户要求更新Git，并明确选择内层HICD_v7仓库。

- 本地Git根：F:/mambacd/home/HICD_v7。
- 远端：origin → https://github.com/Ruikunw726/HICD_v7.git；目标分支master。外层F:/mambacd与baselines_compare未提交或推送。
- 代码与实验记录提交：`40a72a1cf59a4a3bf382a622fc67e8a31181898c`，`feat: publish strict T1-T3 training and versioned DINO integration`。
- 推送成功：origin/master从e8c872f快进到40a72a1；随后git ls-remote确认远端完整SHA与本地一致。原先本地未推送的2834e23历史v9提交也作为已有父提交正常发布，没有重写历史或force push。

## 本次范围

81个相关文件，包含严格目标—状态评估、Farm预测不变规则、原真实成员划分恢复及catalog加载、训练完整性修复、DINO校正独立builder/配置/入口、条件接续检查、54项回归测试、项目/实验记录、数据审计摘要、方法诊断与重构提案。

将v12的100轮完成总结、90轮完整test及独立核验JSON复制到`docs/experiments/results_snapshot_20261012/`，总约643KB，原始结果未改。README已区分当前T1–T3协议与v9历史BDD/过采样说明。添加Python/Shell/YAML/Markdown/JSON/CSV的LF检出约定，避免Windows/WSL行尾问题。

权重、原始图像、全量catalog/名单、巨大训练日志/checkpoints、其他论文/PPT与外层基线未纳入。历史清理删除项以及未审阅旧launch_v9改动保留在本地未提交状态，不用此次发布清空或恢复它们。

## 验证与运行保护

- 待提交Git树：`ee0b8cab92d614efa812861c9f29b5773e6f5e4f`，单独导出到临时目录验证，包含Git中仍保留但本地已删除的历史文件，避免仅测试工作区而漏掉实际发布快照的差异。
- 该Git快照54项CPU测试通过；4个所发布启动脚本bash-n通过，证据`repo_update_validation_20261012.log`。
- 暂存区没有删除项；本轮及原待推送提交没有超过5MiB的blob或模型权重，特征代码和紧凑计数结果均为文本。特征凭证模式扫描未命中GitHub/OpenAI私钥标记；不将该检查扩大为全面安全审计。
- Git空白检查通过，允许既有Markdown记忆末尾空行；CSV在Git中规范为LF，数值/空值不变。
- 提交前后DINO自动接续17份源码/配置hash不变；旧主实验与运行基线依赖的源码hash仍匹配其manifest。没有checkout/rebase/reset、改训练文件内容、停止训练或启动GPU任务。
- 当前校正主实验仍为等待基线完成，heartbeat id=dino保持既有配置。Git发布不等于新100轮已经启动或方法已改善。

本记录、验证日志和01专题记忆随后另作小型文档提交。实际最新HEAD可用git log或远端master核对；记录中的40a72a1是已验证发布的代码提交，不用文档提交改写其身份。
