"""
Instance Separation Loss for HICD v7
训练时让同一实例内像素特征靠近质心，不同实例质心互相远离。
推理时零开销。

集成步骤（等当前实验跑完后执行）：
1. 在 v7_loss.py 的 V7Loss.__init__ 中添加 separation_loss
2. 在 v7_loss.py 的 V7Loss.forward 中添加 separation loss 计算
3. 在 HICD_v7.py 的 forward 中把 fused_features 加入 predictions
4. 在 train_full_v7.py 中添加 epoch 传递和退化监控
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InstanceSeparationLoss(nn.Module):
    """
    实例分离损失：在特征空间中，让同一实例内的像素特征靠近质心，
    不同实例的质心互相远离。只在训练时使用。
    
    实例定义：同一target class + 同一state = 一个语义实例
    """
    def __init__(self, temperature=0.1, min_instance_pixels=50):
        super().__init__()
        self.temperature = temperature
        self.min_instance_pixels = min_instance_pixels

    def forward(self, features, target_gt, state_gt):
        """
        Args:
            features: [B, C, H, W] 融合特征（TFM输出，下采样4x）
            target_gt: [B, H, W] 目标类别标注（已下采样到特征尺寸）
            state_gt: [B, H, W] 状态标注（已下采样到特征尺寸）
        Returns:
            loss: scalar, 实例分离损失
        """
        B, C, H, W = features.shape
        device = features.device
        total_loss = torch.tensor(0.0, device=device)
        instance_count = 0
        
        for b in range(B):
            target_np = target_gt[b].detach().cpu().numpy()
            state_np = state_gt[b].detach().cpu().numpy()
            feat_b = features[b]  # [C, H, W]
            
            # 按target class + state分组提取实例
            centroids = []
            for cls_id in np.unique(target_np):
                if cls_id == 0 or cls_id == 255:
                    continue
                cls_mask = (target_np == cls_id)
                
                for st_id in np.unique(state_np[cls_mask]):
                    if st_id == 6:  # skip none
                        continue
                    
                    inst_mask = cls_mask & (state_np == st_id)
                    inst_pixels = int(np.sum(inst_mask))
                    
                    if inst_pixels < self.min_instance_pixels:
                        continue
                    
                    # 提取实例特征 [C, N]
                    inst_mask_t = torch.from_numpy(inst_mask).to(device)
                    inst_feats = feat_b[:, inst_mask_t]
                    
                    # 质心 [C, 1]
                    centroid = inst_feats.mean(dim=1, keepdim=True)
                    
                    # Intra-instance: 余弦相似度 -> 像素向质心聚拢
                    inst_norm = F.normalize(inst_feats, dim=0)
                    centroid_norm = F.normalize(centroid, dim=0)
                    sim = (inst_norm * centroid_norm).sum(dim=0)  # [N]
                    intra_loss = (1.0 - sim).mean()
                    
                    total_loss = total_loss + intra_loss
                    centroids.append(centroid.squeeze(1))
                    instance_count += 1
            
            # Inter-instance: 不同实例质心互相远离（至少2个实例才计算）
            if len(centroids) >= 2:
                centroid_stack = torch.stack(centroids)  # [K, C]
                centroid_norm = F.normalize(centroid_stack, dim=1)
                sim_matrix = centroid_norm @ centroid_norm.T  # [K, K]
                
                # 只惩罚非对角线（不同实例间的相似度）
                K = len(centroids)
                mask = ~torch.eye(K, dtype=torch.bool, device=device)
                inter_loss = sim_matrix[mask].clamp(min=0).mean()
                
                total_loss = total_loss + inter_loss
        
        if instance_count > 0:
            total_loss = total_loss / instance_count
        
        return total_loss


class SeparationWeightScheduler:
    """
    权重调度：warmup + 线性ramp + 退化检测自动禁用
    
    使用方式：
        scheduler = SeparationWeightScheduler(warmup=5, ramp=10, max_weight=0.1)
        for epoch in range(max_epochs):
            w = scheduler.get_weight(epoch)
            # ... 训练 ...
            if epoch % 5 == 0:
                current_iou = {...}
                if scheduler.check_degradation(current_iou):
                    # 自动禁用separation loss
    """
    def __init__(self, warmup=5, ramp=10, max_weight=0.1, 
                 degrade_threshold=0.03, patience=3):
        self.warmup = warmup
        self.ramp = ramp
        self.max_weight = max_weight
        self.degrade_threshold = degrade_threshold
        self.patience = patience
        
        self.baseline_iou = {}
        self.degrade_count = {}
        self.enabled = True
    
    def get_weight(self, epoch):
        """返回当前epoch的separation loss权重"""
        if not self.enabled or epoch < self.warmup:
            return 0.0
        ramp_epoch = epoch - self.warmup
        if ramp_epoch >= self.ramp:
            return self.max_weight
        return self.max_weight * (ramp_epoch / self.ramp)
    
    def set_baseline(self, per_class_iou):
        """在warmup结束时设置基线IoU"""
        self.baseline_iou = dict(per_class_iou)
        self.degrade_count = {cls: 0 for cls in per_class_iou}
    
    def check_degradation(self, current_iou):
        """检查是否有已学好类退化，返回True则应禁用"""
        if not self.enabled or not self.baseline_iou:
            return False
        
        should_disable = False
        for cls, base_iou in self.baseline_iou.items():
            if base_iou < 0.1:  # 只监控已学好的类
                continue
            if cls in current_iou:
                drop = base_iou - current_iou[cls]
                if drop > self.degrade_threshold:
                    self.degrade_count[cls] = self.degrade_count.get(cls, 0) + 1
                    if self.degrade_count[cls] >= self.patience:
                        print(f"[WARNING] Class {cls} IoU dropped {drop:.3f} for {self.patience} checks, disabling separation loss")
                        self.enabled = False
                        should_disable = True
                else:
                    self.degrade_count[cls] = 0
        return should_disable


# ============================================================
# 集成代码片段（等当前实验跑完后粘贴到对应文件）
# ============================================================

# --- v7_loss.py: V7Loss.__init__ 中添加 ---
INIT_SNIPPET = '''
        # Instance Separation Loss (P0)
        from models.losses.instance_separation import InstanceSeparationLoss
        self.separation_loss = InstanceSeparationLoss(
            temperature=config.get('separation_temperature', 0.1),
            min_instance_pixels=config.get('separation_min_pixels', 50)
        )
        self.w_separation = config.get('w_separation', 0.02)
        self._sep_scheduler = None  # set externally by trainer
'''

# --- v7_loss.py: V7Loss.forward 中，在 loss_contrastive 之后添加 ---
FORWARD_SNIPPET = '''
        # === 6. Instance Separation Loss ===
        loss_separation = _zero(device)
        loss_dict['loss_separation'] = 0.0
        if 'fused_features' in predictions and 'target_gt' in targets and 'state_gt' in targets:
            w_sep = self.w_separation
            if self._sep_scheduler is not None:
                w_sep = self._sep_scheduler.get_weight(
                    self.config.get('_current_epoch', 0))
            if w_sep > 0:
                # 下采样target/state到特征尺寸
                feat = predictions['fused_features']
                fh, fw = feat.shape[2], feat.shape[3]
                tgt = targets['target_gt']
                st = targets['state_gt']
                if tgt.shape[1] != fh or tgt.shape[2] != fw:
                    tgt = F.interpolate(tgt.unsqueeze(1).float(), size=(fh, fw), mode='nearest').squeeze(1).long()
                    st = F.interpolate(st.unsqueeze(1).float(), size=(fh, fw), mode='nearest').squeeze(1).long()
                loss_separation = self.separation_loss(feat, tgt, st) * w_sep
                loss_dict['loss_separation'] = loss_separation.item()

        total = total + loss_separation
'''

# --- HICD_v7.py: forward 中，在 return 之前添加 ---
MODEL_SNIPPET = '''
        # 把融合特征传给loss计算（separation loss需要）
        predictions['fused_features'] = fused_features  # [B, hidden_dim, H/4, W/4]
'''

# --- train_full_v7.py: TrainerV7.__init__ 中添加 ---
TRAINER_INIT_SNIPPET = '''
        # Separation loss scheduler
        from models.losses.instance_separation import SeparationWeightScheduler
        self.sep_scheduler = SeparationWeightScheduler(
            warmup=config.get('separation_warmup_epochs', 5),
            ramp=config.get('separation_ramp_epochs', 10),
            max_weight=config.get('separation_max_weight', 0.1)
        )
        if hasattr(self.model, 'loss_fn'):
            self.model.loss_fn._sep_scheduler = self.sep_scheduler
'''

# --- train_full_v7.py: 每个epoch结束时添加 ---
TRAINER_EPOCH_SNIPPET = '''
        # Update separation scheduler
        self.model.config['_current_epoch'] = epoch
        if epoch == self.sep_scheduler.warmup:
            self.sep_scheduler.set_baseline(val_results.get('target_iou', {}))
        if epoch % 5 == 0 and epoch > self.sep_scheduler.warmup:
            current_iou = {i: v for i, v in enumerate(val_results.get('target_iou', []))}
            self.sep_scheduler.check_degradation(current_iou)
'''


