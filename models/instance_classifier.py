"""
Instance-level classifier for T3 (mobile) and T4 (fixed facility) targets.

Completely decoupled from pixel-level branch.
Input: dual-temporal crops (T1 + T2) of the same instance region
Output: target class (7 classes) + state (7 classes)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import tifffile
from PIL import Image


# Class definitions
INSTANCE_CLASSES = ['Shelter', 'Pier', 'Dock', 'Tank', 'Aircraft', 'Vessel', 'Crater']
STATE_NAMES = ['no-change', 'damaged', 'reduced', 'added', 'extended', 'replaced', 'none']

# Class ID mapping (original target_id -> instance class id)
CLASS_ID_MAP = {7: 0, 9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6}


class InstanceDataset(Dataset):
    """Dataset for instance-level classification with dual-temporal input."""

    def __init__(self, data_dir, instance_json, img_size=256, crop_size=64,
                 jitter=0.2, margin=0.3, augment=True):
        self.data_dir = data_dir
        self.img_size = img_size
        self.crop_size = crop_size
        self.jitter = jitter
        self.margin = margin
        self.augment = augment

        # Load instance annotations
        with open(instance_json, 'r', encoding='utf-8') as f:
            self.instance_data = json.load(f)

        # Build sample list: (pre_path, post_path, bbox, class_id, state_id)
        self.samples = []
        for sample_name, instances in self.instance_data['samples'].items():
            pre_path = os.path.join(data_dir, 'train', 'image', f'{sample_name}_pre_war.tif')
            post_path = os.path.join(data_dir, 'train', 'image', f'{sample_name}_post_war.tif')

            # Fallback: if pre doesn't exist, try post for both
            if not os.path.exists(pre_path):
                pre_path = post_path
            if not os.path.exists(post_path):
                post_path = pre_path
            if not os.path.exists(pre_path):
                continue

            for inst in instances:
                orig_class_id = inst['class_id']
                if orig_class_id not in CLASS_ID_MAP:
                    continue
                self.samples.append({
                    'pre_path': pre_path,
                    'post_path': post_path,
                    'bbox': inst['bbox'],  # [x1, y1, x2, y2]
                    'class_id': CLASS_ID_MAP[orig_class_id],
                    'state_id': inst['state'],
                    'pixels': inst['pixels'],
                })

        print(f"Loaded {len(self.samples)} instances from {len(self.instance_data['samples'])} images")

        # Class statistics for weighted sampling
        self.class_counts = np.zeros(len(INSTANCE_CLASSES))
        for s in self.samples:
            self.class_counts[s['class_id']] += 1

    def __len__(self):
        return len(self.samples)

    def _read_image(self, path):
        """Read TIF image as RGB numpy array."""
        try:
            arr = tifffile.imread(path)
            if arr.ndim == 3 and arr.shape[0] <= 4:
                arr = arr[:3].transpose(1, 2, 0)  # CHW -> HWC
            elif arr.ndim == 3 and arr.shape[2] >= 3:
                arr = arr[:, :, :3]  # HWC
            return arr.astype(np.uint8)
        except Exception:
            from PIL import Image as PILImage
            return np.array(PILImage.open(path).convert('RGB'))

    def _crop_instance(self, img, bbox, jitter_offset=None):
        """Crop instance region with margin and optional jitter."""
        x1, y1, x2, y2 = bbox
        h, w = img.shape[:2]

        # Add margin
        bw, bh = x2 - x1, y2 - y1
        mx, my = int(bw * self.margin), int(bh * self.margin)
        x1, y1 = x1 - mx, y1 - my
        x2, y2 = x2 + mx, y2 + my

        # Add jitter (same offset for T1 and T2 to keep alignment)
        if jitter_offset is not None:
            jx, jy = jitter_offset
            x1, y1 = x1 + jx, y1 + jy
            x2, y2 = x2 + jx, y2 + jy
        elif self.augment and self.jitter > 0:
            jx = int(bw * self.jitter * (np.random.rand() * 2 - 1))
            jy = int(bh * self.jitter * (np.random.rand() * 2 - 1))
            x1, y1 = x1 + jx, y1 + jy
            x2, y2 = x2 + jx, y2 + jy

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Ensure valid box
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, min(64, w), min(64, h)

        return img[y1:y2, x1:x2]

    def _augment_pair(self, crop_t1, crop_t2):
        """Apply same augmentation to both temporal crops."""
        if not self.augment:
            return crop_t1, crop_t2

        # Random rotation (same for both)
        if np.random.rand() < 0.5:
            k = np.random.randint(4)
            crop_t1 = np.rot90(crop_t1, k).copy()
            crop_t2 = np.rot90(crop_t2, k).copy()

        # Random flip (same for both)
        if np.random.rand() < 0.5:
            crop_t1 = crop_t1[:, ::-1].copy()
            crop_t2 = crop_t2[:, ::-1].copy()
        if np.random.rand() < 0.5:
            crop_t1 = crop_t1[::-1, :].copy()
            crop_t2 = crop_t2[::-1, :].copy()

        # Color jitter (slightly different for each to simulate illumination change)
        if np.random.rand() < 0.5:
            brightness = 0.7 + 0.6 * np.random.rand()
            crop_t1 = np.clip(crop_t1.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
            crop_t2 = np.clip(crop_t2.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

        return crop_t1, crop_t2

    def _resize(self, crop):
        """Resize crop to target size."""
        from PIL import Image as PILImage
        crop_pil = PILImage.fromarray(crop)
        crop_pil = crop_pil.resize((self.crop_size, self.crop_size), PILImage.BILINEAR)
        return np.array(crop_pil)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Read both temporal images
        img_t1 = self._read_image(sample['pre_path'])
        img_t2 = self._read_image(sample['post_path'])

        # Generate jitter offset (same for both to keep alignment)
        if self.augment and self.jitter > 0:
            x1, y1, x2, y2 = sample['bbox']
            bw, bh = x2 - x1, y2 - y1
            jx = int(bw * self.jitter * (np.random.rand() * 2 - 1))
            jy = int(bh * self.jitter * (np.random.rand() * 2 - 1))
            jitter_offset = (jx, jy)
        else:
            jitter_offset = None

        # Crop both temporal images with same bbox + jitter
        crop_t1 = self._crop_instance(img_t1, sample['bbox'], jitter_offset)
        crop_t2 = self._crop_instance(img_t2, sample['bbox'], jitter_offset)

        # Apply same augmentation to both
        crop_t1, crop_t2 = self._augment_pair(crop_t1, crop_t2)

        # Resize
        crop_t1 = self._resize(crop_t1)
        crop_t2 = self._resize(crop_t2)

        # To tensors: HWC -> CHW, normalize
        crop_t1_tensor = torch.from_numpy(crop_t1).permute(2, 0, 1).float() / 255.0
        crop_t2_tensor = torch.from_numpy(crop_t2).permute(2, 0, 1).float() / 255.0

        return {
            'image_t1': crop_t1_tensor,
            'image_t2': crop_t2_tensor,
            'target': sample['class_id'],
            'state': sample['state_id'],
        }


def get_class_weights(dataset):
    """Compute inverse frequency class weights."""
    counts = dataset.class_counts
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(weights)
    return torch.FloatTensor(weights)


class ConvBlock(nn.Module):
    """Basic conv block with BN and ReLU."""

    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FrequencyBranch(nn.Module):
    """Frequency domain branch using DWT."""

    def __init__(self, in_channels=3, out_dim=64, levels=2):
        super().__init__()
        self.levels = levels

        # High frequency encoder
        self.hf_encoder = nn.Sequential(
            ConvBlock(in_channels * 3, 32),  # 3 HF subbands per level
            ConvBlock(32, 64),
            nn.AdaptiveAvgPool2d(1),
        )

        # Low frequency encoder
        self.lf_encoder = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            nn.AdaptiveAvgPool2d(1),
        )

        self.proj = nn.Linear(128, out_dim)

    def dwt(self, x):
        """Haar DWT - returns LL, LH, HL, HH."""
        # Simple 2x2 average pooling for LL
        ll = F.avg_pool2d(x, 2)
        # Upsample and subtract for HF
        ll_up = F.interpolate(ll, scale_factor=2, mode='bilinear', align_corners=True)
        hf = x - ll_up
        return ll, hf

    def forward(self, x):
        B, C, H, W = x.shape

        # Multi-level DWT
        hf_features = []
        current = x
        for _ in range(self.levels):
            ll, hf = self.dwt(current)
            hf_features.append(hf)
            current = ll

        # Encode HF (concat all levels, resized to same size)
        hf_resized = []
        for hf in hf_features:
            hf_resized.append(F.interpolate(hf, size=(H, W), mode='bilinear', align_corners=True))
        hf_cat = torch.cat(hf_resized, dim=1)  # [B, C*levels, H, W]

        # Take first 3*C channels for HF encoder
        hf_cat = hf_cat[:, :C * 3] if hf_cat.shape[1] >= C * 3 else F.pad(hf_cat, (0, 0, 0, 0, 0, C * 3 - hf_cat.shape[1]))
        hf_feat = self.hf_encoder(hf_cat).flatten(1)  # [B, 64]

        # Encode LF
        lf_feat = self.lf_encoder(current).flatten(1)  # [B, 64]

        # Concat and project
        freq_feat = torch.cat([hf_feat, lf_feat], dim=1)  # [B, 128]
        return self.proj(freq_feat)  # [B, out_dim]


class InstanceClassifier(nn.Module):
    """
    Instance-level classifier for T3/T4 targets with dual-temporal input.

    Architecture:
        T1 crop ──┐
                  ├─ Shared Spatial Branch + Frequency Branch → Fusion → Classification
        T2 crop ──┘
    """

    def __init__(self, num_classes=7, num_states=7, crop_size=64, freq_dim=64):
        super().__init__()

        self.num_classes = num_classes
        self.num_states = num_states

        # Spatial branch (shared for T1 and T2)
        self.spatial_branch = nn.Sequential(
            ConvBlock(3, 32, stride=2),    # 32x32
            ConvBlock(32, 64, stride=2),   # 16x16
            ConvBlock(64, 128, stride=2),  # 8x8
            ConvBlock(128, 256, stride=2), # 4x4
            nn.AdaptiveAvgPool2d(1),       # 1x1
        )
        self.spatial_proj = nn.Linear(256, 256)

        # Frequency branch (shared for T1 and T2)
        self.freq_branch = FrequencyBranch(in_channels=3, out_dim=freq_dim)

        # Dual-temporal fusion
        # Input: [T1_spatial, T2_spatial, T1-T2_spatial, T1_freq, T2_freq, T1-T2_freq]
        # = 256*3 + freq_dim*3
        dual_dim = 256 * 3 + freq_dim * 3
        self.fusion = nn.Sequential(
            nn.Linear(dual_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # Classification heads
        self.target_head = nn.Linear(256, num_classes)
        self.state_head = nn.Linear(256, num_states)

    def _extract_features(self, x):
        """Extract spatial and frequency features from a single temporal input."""
        # Spatial features
        spatial_feat = self.spatial_branch(x).flatten(1)  # [B, 256]
        spatial_feat = self.spatial_proj(spatial_feat)

        # Frequency features
        freq_feat = self.freq_branch(x)  # [B, freq_dim]

        return spatial_feat, freq_feat

    def forward(self, x_t1, x_t2):
        """
        x_t1: [B, 3, crop_size, crop_size] - T1 (pre-war) crop
        x_t2: [B, 3, crop_size, crop_size] - T2 (post-war) crop
        Returns: target_logits [B, num_classes], state_logits [B, num_states]
        """
        # Extract features from both temporal inputs (shared weights)
        spatial_t1, freq_t1 = self._extract_features(x_t1)
        spatial_t2, freq_t2 = self._extract_features(x_t2)

        # Compute difference features (capture change)
        spatial_diff = spatial_t1 - spatial_t2
        freq_diff = freq_t1 - freq_t2

        # Concatenate: [T1, T2, Diff] for both spatial and frequency
        fused = torch.cat([
            spatial_t1, spatial_t2, spatial_diff,  # 256*3
            freq_t1, freq_t2, freq_diff,            # freq_dim*3
        ], dim=1)

        # Fusion
        fused = self.fusion(fused)

        # Classification
        target_logits = self.target_head(fused)
        state_logits = self.state_head(fused)

        return target_logits, state_logits


class InstanceLoss(nn.Module):
    """Combined loss for instance classification."""

    def __init__(self, class_weights=None, state_weights=None, w_state=1.0):
        super().__init__()
        self.w_state = w_state

        # Target loss: Focal CE
        if class_weights is not None:
            self.target_ce = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.target_ce = nn.CrossEntropyLoss()

        # State loss
        if state_weights is not None:
            self.state_ce = nn.CrossEntropyLoss(weight=state_weights)
        else:
            self.state_ce = nn.CrossEntropyLoss()

    def forward(self, target_logits, state_logits, target_gt, state_gt):
        loss_target = self.target_ce(target_logits, target_gt)
        loss_state = self.state_ce(state_logits, state_gt)
        total = loss_target + self.w_state * loss_state
        return total, {'loss_target': loss_target.item(), 'loss_state': loss_state.item()}
