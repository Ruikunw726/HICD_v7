# HICD V7 损失计算 & 评估协议 逐项排查

## 一、当前所有损失项

| # | 损失项 | 权重 | 计算方式 | 状态 |
|---|--------|------|----------|------|
| 1 | inst_loss_class | 1.0 | DETR分类CE（匈牙利匹配后的query vs GT类别） | ⚠️ 见问题1 |
| 2 | inst_loss_bbox | 5.0 | L1 loss（匹配后的query bbox vs GT bbox） | ✅ |
| 3 | inst_loss_giou | 2.0 | GIoU loss | ✅ |
| 4 | inst_loss_state | 2.0 | CE loss（匹配后的query状态 vs GT状态） | ⚠️ 见问题2 |
| 5 | loss_change_type | 1.0 | CE（per-instance变化类型） | ❌ 见问题3 |
| 6 | loss_pixel | 1.0 | CE + Dice（像素级6类分割） | ✅ 已修复 |
| 7 | loss_damage_level | 1.0 | CE（per-instance损毁等级） | ⚠️ 见问题4 |
| 8 | loss_consistency | 0.3 | Box内像素状态一致性 | ⚠️ 见问题5 |

## 二、逐项问题分析

### 问题1: inst_loss_class — 前景类权重过低
- empty_weight[-1] = 0.1（no-object类权重0.1）
- num_classes=1时，前景类权重=1.0，背景权重=0.1
- 匈牙利匹配后，未匹配的query全部算no-object loss
- **风险**：如果GT很少（SECOND每张图平均8.5个bbox），大量query学no-object，loss被背景主导
- **建议**：检查实际匹配比例，考虑提高前景权重或降低背景权重

### 问题2: inst_loss_state — 状态类别数配置问题
- SECOND配置num_states=4（no-change/unused/increase/decrease）
- 但instances.json中states只有0/1/2（无unused）
- DETR state head输出4维，GT标签最大值2
- **问题**：unused类永远没有GT，浪费一个输出维度
- **建议**：SECOND改为num_states=3

### 问题3: loss_change_type — 严重Bug！❌
`python
batch_losses.append(self.change_type_ce(ct_logits[b_idx, :n], ct_gt[:n]))
`
- 这里直接取ct_logits的前n个query和GT比较
- 但DETR用匈牙利匹配，匹配到的query可能是任意位置的！
- **这是完全错误的**：第1个query不一定匹配第1个GT
- **应该用匈牙利匹配的结果**来索引正确的query

### 问题4: loss_damage_level — 当前未激活
- active_heads中damage_level=false，所以这个loss没在算
- **暂时OK**，等需要时再开

### 问题5: loss_consistency — 实现可能有问题
- BoxConsistencyLoss用roi_align从pred_state_map中提取bbox区域
- pred_state_map实际是target_type_map（像素分支的状态输出）
- 但SECOND的change_state是0/1（二值），而target_type_map输出num_states维
- **问题**：语义不匹配，consistency loss可能没有正确约束

### 问题6: ChangeGuidedAttention — 定义了但没用！
- interaction.py中定义了ChangeGuidedAttention
- 但HICD_v7.py的forward中只是注释说要用，实际没有调用
- **像素→实例的指导通路是断的**

## 三、评估协议问题

### mAP计算
- 使用11点插值法（VOC风格）—— 标准做法 ✅
- 匈牙利匹配后的预测 vs GT比较IoU —— 但评估器不是用匈牙利匹配
- 评估器是按score排序后贪心匹配（每个GT只匹配一次）—— 标准做法 ✅
- score阈值0.1过滤低置信度预测 —— ✅

### 关键问题：评估器的pred_boxes和pred_logits格式
- pred_logits: [B, Nq, 2]（Building + no-object）
- softmax后取前景类概率作为score
- pred_boxes: [B, Nq, 4]（cx,cy,w,h归一化坐标）
- GT boxes也是归一化坐标
- **格式一致** ✅

### 为什么mAP一直是0？
可能原因：
1. 评估器修复后没部署到服务器（之前的key修复）
2. 训练轮数不够，前景score太低被0.1阈值过滤
3. bbox预测还没学好，IoU全部低于阈值
4. SECOND的instances.json刚生成，之前训练用的是空标注

## 四、需要立即修复的优先级

| 优先级 | 问题 | 影响 |
|--------|------|------|
| P0 | change_type_loss用错索引 | 实例级loss全部算错 |
| P0 | ChangeGuidedAttention没接入 | 像素→实例指导断路 |
| P1 | SECOND num_states 4→3 | 浪费一个输出维度 |
| P1 | BoxConsistencyLoss语义不匹配 | 约束loss可能无效 |
| P2 | 评估器部署确认 | mAP可能因为旧代码为0 |
