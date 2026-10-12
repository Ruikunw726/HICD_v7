# DINO 接入校正已就绪，等待基线完成

日期：2026-10-12，Asia/Shanghai。当前授权：先校正DINO接入，当前Bi-SRNet对比实验正常完成后，启动新的100轮T1–T3主实验。此前“暂不改模型”范围已被本次明确指令更新。本轮没有启动GPU训练/测试，也没有停止或修改正在运行的基线。

## 校正与版本

- 新版本：`hicd.dinov3_input_features.v1`，独立文件`models/backbone/dinov3_corrected.py`与显式builder `models/dino_corrected_model.py`。
- 输入：dataset继续RGB[0,1]；仅DINO内部按mean[0.485,0.456,0.406]/std[0.229,0.224,0.225]归一化；频域支路继续使用原RGB。原数据/增强不变。
- 特征：使用本机timm1.0.28的真实Eva.forward_intermediates接口，在block2/5/8/11执行后取特征，正确处理RoPE及5个前缀token；norm=True使用预训练最终LayerNorm规范特征，再接已有projection LayerNorm/Linear与尺度适配。最后block及final norm都进入输出和梯度路径。
- 旧`dinov3.py`、旧模型factory、原train_strict/train_full、损失、图、频域与严格评估核心未改。旧配置/权重/test与基线冻结代码可继续复现，旧默认入口不会悄然启用新版。
- 新训练与评估分别用`script/train_dino_corrected_v7.py`、`script/eval_dino_corrected_test.py`；新入口显式要求版本字段，checkpoint dataset_config与manifest记录身份。评估新run必须用新评估入口，不能用旧builder读新权重后产生旧输入语义。

## 预训练来源与验收

- 本地文件SHA256：c70aebcfbe6ad4b504aadf274feb4eebaf9af6b649c66de6c6e1670c81810fc9。
- 对照权重：timm/vit_base_patch16_dinov3.lvd1689m，固定HF revision c6a5fb7d12bbd3cf3b0079253141c3332aaed7da；远端SHA256 1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332已核对下载内容。
- 文件整体SHA不同，但本机模型实际使用的162个参数/缓冲张量与对照逐项torch.equal；本地额外24个q_bias/v_bias历史键不属于当前timm模型state_dict，原加载也报告unexpected24。没有用文件名推测来源或替换本地预训练文件。
- CPU真实预训练接口核验通过：独立after-block hooks与全部四层token一致，输出64×64/32×32/16×16/8×8，原输入不变，mean输入归一化为0；last block qkv梯度L1=160.8058，final norm梯度L1=0.9196，均finite且非零。
- 真实CPU smoke通过：2个增强train样本、AdamW一次更新，末block有finite非零梯度并实际变化；另2个真实val样本，target/state输出64×64、严格计数8192闭合。val这两张仅背景，foreground N/A如实保留；是管线检查，不是性能结果。正式训练会从预训练重新初始化，不加载smoke任务权重。
- 54项CPU测试通过（原51+3项条件接续门控）；bash-n与新入口py_compile通过。GPU train/val smoke尚未执行，按用户顺序等基线释放资源后由launcher执行，成功后才进入正式训练。
- 证据：weight_provenance.json、cpu_acceptance.json、real_cpu_smoke.json、ready_manifest.json；目录旁三个CPU日志。CPU smoke初次因验证wrapper返回None的调用方式而失败，已修正报告脚本并重跑通过；模型实现未因此改变，失败没有GPU占用或正式权重。

## 待启动实验

唯一实验名：`v13_dino_corrected_t123_disjoint_e100_20261012`。

配置：datasets/configs/0617final_t123_dino_corrected_20261012.yaml，除新增dino_integration_version外与原新版配置一致。保持train61029/val13077/test13077、原split身份9f66845372df4c8730a543ac2899575b8ba40eff31b676355b183991e09db05b、T1–T3-only/T4和Tower忽略、预测Farm固定不变、输入256/计分64、原模型其余结构/损失/采样与初始化预算。

正式100轮、batch32、LR1e-4、weight_decay1e-4、cosine T_max100、AMP、workers4、原冻结/解冻比例；从头任务/optimizer/scheduler/AMP，不resume旧50/100轮。仍按val micro选择主best，另存macro-best；latest每轮、编号每10轮。未接受/执行上轮方案的loss重构、细节stem、T3 adapter、图先验替换或macro主选模。

## 自动接续条件和入口

已创建当前聊天的每小时自动接续，名称“基线完成后启动校正 DINO 主实验”，automation id=`dino`；状态ACTIVE。无变化时安静等待，成功启动并确认早期健康后暂停自动接续；失败时报告并暂停，不自动重试/覆盖。

先执行一次条件检查：

```bash
cd /mnt/f/mambacd/home/HICD_v7
bash launch_dino_corrected.sh --check-only
```

必须同时满足：

1. 本READY及验收成立，17份启动关键源码/配置hash仍与ready_manifest匹配。
2. 唯一基线bisrnet_t1t3_20261011_164416的同级status记录TRAINING_FINISHED和TEST_FINISHED均python_exit=0、tee_exit=0，完整100轮指标存在。
3. 基线test_strict.json producer身份正确，完整13077×64×64计数闭合。
4. 无其他训练/测试进程，GPU无compute占用；基线/主实验共享运行锁与启动锁可取得。

满足后执行`bash launch_dino_corrected.sh`。它会重新跑CPU回归及DINO接口核验，prepare/snapshot新run，在GPU做4个真实train+2个val smoke，通过后后台screen正式训练。若已有status/latest则不重启；若有部分准备目录或失败则保留现场并要求检查。启动后需确认Python/screen存活、至少两次batch推进、loss有限、无fatal异常，记录early_health及实验台账/01专题；成功后暂停本自动接续。

基线完成时间是条件，不硬设“明早某时刻”抢占GPU。当前基线尚在运行，首次check-only已返回WAIT。本轮只完成接入校正与接续准备，不能称新100轮已经开始或效果已改善。

本地自动接续需要电脑保持开机、Codex桌面应用运行、项目可访问；应用关闭期间不会保证按时启动。[官方本地调度要求](https://learn.chatgpt.com/docs/automations?surface=app)
