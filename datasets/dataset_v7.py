"""
HICD v7 Dataset - Full image mode (no patch splitting)
Reads full images and resizes to a fixed training size.
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

try:
    from osgeo import gdal
    gdal.UseExceptions()
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False


def read_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.tif', '.tiff'] and HAS_GDAL:
        ds = gdal.Open(path)
        arr = ds.ReadAsArray()
        ds = None
        if arr.ndim == 3 and arr.shape[0] > 3:
            arr = arr[:3]
        return arr
    img = Image.open(path).convert('RGB')
    return np.array(img).transpose(2, 0, 1)


def read_label(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.tif', '.tiff'] and HAS_GDAL:
        ds = gdal.Open(path)
        arr = ds.ReadAsArray()
        ds = None
        return arr
    return np.array(Image.open(path))



def random_augment(img1, img2, label1, label2, boxes=None):
    """Apply consistent spatial augmentation to paired images, labels, and boxes."""
    if np.random.rand() < 0.5:
        img1 = img1[:, :, ::-1].copy()
        img2 = img2[:, :, ::-1].copy()
        label1 = label1[:, ::-1].copy()
        label2 = label2[:, ::-1].copy()
        if boxes is not None and len(boxes) > 0:
            boxes = boxes.clone()
            boxes[:, 0] = 1.0 - boxes[:, 0]
    if np.random.rand() < 0.5:
        img1 = img1[:, ::-1, :].copy()
        img2 = img2[:, ::-1, :].copy()
        label1 = label1[::-1, :].copy()
        label2 = label2[::-1, :].copy()
        if boxes is not None and len(boxes) > 0:
            boxes = boxes.clone()
            boxes[:, 1] = 1.0 - boxes[:, 1]
    k = np.random.randint(4)
    if k > 0:
        img1 = np.rot90(img1, k, axes=(1, 2)).copy()
        img2 = np.rot90(img2, k, axes=(1, 2)).copy()
        label1 = np.rot90(label1, k).copy()
        label2 = np.rot90(label2, k).copy()
        if boxes is not None and len(boxes) > 0:
            boxes = boxes.clone()
            for _ in range(k):
                new_cx = boxes[:, 1]
                new_cy = 1.0 - boxes[:, 0]
                new_w = boxes[:, 3]
                new_h = boxes[:, 2]
                boxes[:, 0] = new_cx
                boxes[:, 1] = new_cy
                boxes[:, 2] = new_w
                boxes[:, 3] = new_h
    return img1, img2, label1, label2, boxes


def color_jitter(img, brightness=0.2, contrast=0.2, saturation=0.1):
    if np.random.rand() > 0.5:
        return img
    img = img.copy()
    b = 1.0 + (np.random.rand() - 0.5) * 2 * brightness
    c = 1.0 + (np.random.rand() - 0.5) * 2 * contrast
    s = 1.0 + (np.random.rand() - 0.5) * 2 * saturation
    img = img * b
    mean = img.mean(axis=(1, 2), keepdims=True)
    img = (img - mean) * c + mean
    gray = img.mean(axis=0, keepdims=True)
    img = img * s + gray * (1 - s)
    return np.clip(img, 0, 1).astype(np.float32)


def random_erasing(img, p=0.3, sl=0.02, sh=0.1):
    if np.random.rand() > p:
        return img
    img = img.copy()
    C, H, W = img.shape
    area = H * W
    for _ in range(10):
        target_area = np.random.uniform(sl, sh) * area
        aspect_ratio = np.random.uniform(0.3, 3.3)
        h = int(round(np.sqrt(target_area * aspect_ratio)))
        w = int(round(np.sqrt(target_area / aspect_ratio)))
        if w < W and h < H:
            x1 = np.random.randint(0, H - h)
            y1 = np.random.randint(0, W - w)
            img[:, x1:x1+h, y1:y1+w] = np.random.rand(C, h, w).astype(np.float32)
            return img
    return img


def gaussian_noise(img, p=0.3, std=0.05):
    if np.random.rand() > p:
        return img
    noise = np.random.randn(*img.shape).astype(np.float32) * std
    return np.clip(img + noise, 0, 1).astype(np.float32)


def random_blur(img, p=0.2, kernel_range=(3, 7)):
    if np.random.rand() > p:
        return img
    from scipy.ndimage import uniform_filter
    k = np.random.choice(range(kernel_range[0], kernel_range[1]+1, 2))
    blurred = np.zeros_like(img)
    for c in range(img.shape[0]):
        blurred[c] = uniform_filter(img[c], size=k)
    return blurred.astype(np.float32)


def copy_paste_both_labels(img1, img2, label1, label2, p=0.3, target_classes=None, max_paste=3):
    """Change-aware copy-paste for bi-temporal change detection.
    Two modes: same position (no-change) and different position (creates change).
    """
    if np.random.rand() > p or target_classes is None:
        return img1, img2, label1, label2, None
    C, H, W = img1.shape
    n_paste = np.random.randint(1, max_paste + 1)
    state_mask = np.full((H, W), -1, dtype=np.int64)
    for _ in range(n_paste):
        cls = np.random.choice(target_classes)
        mask = (label2 == cls)
        if mask.sum() < 50:
            continue
        ys, xs = np.where(mask)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        margin_y = max(1, int((y_max - y_min) * 0.2))
        margin_x = max(1, int((x_max - x_min) * 0.2))
        y1 = max(0, y_min - margin_y)
        y2 = min(H, y_max + margin_y + 1)
        x1 = max(0, x_min - margin_x)
        x2 = min(W, x_max + margin_x + 1)
        patch_h, patch_w = y2 - y1, x2 - x1
        if patch_h < 5 or patch_w < 5:
            continue
        same_position = np.random.rand() < 0.5
        if same_position:
            dst_y = np.random.randint(0, H - patch_h + 1)
            dst_x = np.random.randint(0, W - patch_w + 1)
            dst_mask = label2[dst_y:dst_y+patch_h, dst_x:dst_x+patch_w] == 0
            if dst_mask.sum() < patch_h * patch_w * 0.3:
                continue
            src1 = img1[:, y1:y2, x1:x2].copy()
            src2 = img2[:, y1:y2, x1:x2].copy()
            for c in range(C):
                img1[c, dst_y:dst_y+patch_h, dst_x:dst_x+patch_w][dst_mask] = src1[c, :patch_h, :patch_w][dst_mask]
                img2[c, dst_y:dst_y+patch_h, dst_x:dst_x+patch_w][dst_mask] = src2[c, :patch_h, :patch_w][dst_mask]
            label1[dst_y:dst_y+patch_h, dst_x:dst_x+patch_w][dst_mask] = cls
            label2[dst_y:dst_y+patch_h, dst_x:dst_x+patch_w][dst_mask] = cls
            state_mask[dst_y:dst_y+patch_h, dst_x:dst_x+patch_w][dst_mask] = 0
        else:
            if np.random.rand() < 0.5:
                dst_y = np.random.randint(0, H - patch_h + 1)
                dst_x = np.random.randint(0, W - patch_w + 1)
                dst_mask = label2[dst_y:dst_y+patch_h, dst_x:dst_x+patch_w] == 0
                if dst_mask.sum() < patch_h * patch_w * 0.3:
                    continue
                src2 = img2[:, y1:y2, x1:x2].copy()
                for c in range(C):
                    img2[c, dst_y:dst_y+patch_h, dst_x:dst_x+patch_w][dst_mask] = src2[c, :patch_h, :patch_w][dst_mask]
                label2[dst_y:dst_y+patch_h, dst_x:dst_x+patch_w][dst_mask] = cls
                paste_state = np.random.choice([2, 3])  # reduced or added
                state_mask[dst_y:dst_y+patch_h, dst_x:dst_x+patch_w][dst_mask] = paste_state
            else:
                dst_y = np.random.randint(0, H - patch_h + 1)
                dst_x = np.random.randint(0, W - patch_w + 1)
                dst_mask = label1[dst_y:dst_y+patch_h, dst_x:dst_x+patch_w] == 0
                if dst_mask.sum() < patch_h * patch_w * 0.3:
                    continue
                src1 = img1[:, y1:y2, x1:x2].copy()
                for c in range(C):
                    img1[c, dst_y:dst_y+patch_h, dst_x:dst_x+patch_w][dst_mask] = src1[c, :patch_h, :patch_w][dst_mask]
                label1[dst_y:dst_y+patch_h, dst_x:dst_x+patch_w][dst_mask] = cls
                paste_state = np.random.choice([1, 5])  # damaged or replaced
                state_mask[dst_y:dst_y+patch_h, dst_x:dst_x+patch_w][dst_mask] = paste_state
    return img1, img2, label1, label2, state_mask

class FullImageDataset(Dataset):
    """
    Full-image dataset for V7. No patch splitting.
    
    Reads full images, resizes to `img_size`, and provides all
    instance annotations in normalized coordinates.
    """
    
    def __init__(self, data_dir, config, split='train', scenes=None,
                 img_size=1024, boxes_dir=None):
        super().__init__()
        self.data_dir = data_dir
        self.config = config
        self.split = split
        self.img_size = img_size
        self.boxes_dir = boxes_dir
        
        self.pre_suffix = config.get('pre_suffix', '_pre')
        self.post_suffix = config.get('post_suffix', '_post')
        self.label_suffix = config.get('label_suffix', '_label')
        self.image_ext = config.get('image_ext', '.tif')
        self.label_ext = config.get('label_ext', '.tif')
        
        self.label_decode_map = config.get('label_decode_map', None)
        self.state_decode_map = config.get('state_decode_map', None)
        self.augment = (split == 'train')
        self.is_flat = self._detect_flat_structure()
        
        if scenes is None:
            scenes = self._get_scenes()
        self.scenes = scenes
        
        self.instances_json = self._load_instances_json()
        self.samples = self._collect_samples()
        
        self._img_cache = {}
        self._cache_max = 32
        
        print(f"FullImageDataset: {config.get('dataset', 'unknown')}")
        print(f"  Split: {split}, Scenes: {len(self.scenes)}, Samples: {len(self.samples)}")
        print(f"  Image size: {img_size}x{img_size}, Augment: {self.augment}")
    
    def _detect_flat_structure(self):
        return os.path.isdir(os.path.join(self.data_dir, self.split, 'image'))
    
    def _get_scenes(self):
        if self.is_flat:
            return ['']
        scenes = []
        for item in sorted(os.listdir(self.data_dir)):
            item_path = os.path.join(self.data_dir, item)
            if os.path.isdir(item_path) and item not in ['train', 'val', 'test']:
                if os.path.exists(os.path.join(item_path, self.split)):
                    scenes.append(item)
        return sorted(scenes)
    
    def _load_instances_json(self):
        json_path = os.path.join(self.data_dir, 'instances.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)
        return None
    
    def _get_split_dir(self, scene):
        if self.is_flat or scene == '':
            return os.path.join(self.data_dir, self.split)
        return os.path.join(self.data_dir, scene, self.split)
    
    def _collect_samples(self):
        samples = []
        for si, scene in enumerate(self.scenes):
            split_dir = self._get_split_dir(scene)
            img_dir = os.path.join(split_dir, 'image')
            if not os.path.exists(img_dir):
                continue
            lbl_dir = os.path.join(split_dir, 'label')
            
            files = sorted(os.listdir(img_dir))
            for fname in files:
                if not fname.endswith(self.pre_suffix + self.image_ext):
                    continue
                base_name = fname.replace(self.pre_suffix + self.image_ext, '')
                pre_path = os.path.join(img_dir, fname)
                post_path = os.path.join(img_dir, base_name + self.post_suffix + self.image_ext)
                label_path = os.path.join(lbl_dir, base_name + self.label_suffix + self.label_ext)
                
                if not os.path.exists(post_path):
                    continue
                
                inst_key = None
                if self.instances_json is not None:
                    if self.is_flat or scene == '':
                        inst_key = f"{self.split}/{base_name}{self.pre_suffix}{self.image_ext}"
                    else:
                        inst_key = f"{scene}/{self.split}/{base_name}{self.pre_suffix}{self.image_ext}"
                    if inst_key not in self.instances_json:
                        alt_key = f"{self.split}/{base_name}{self.pre_suffix}{self.image_ext}"
                        if alt_key in self.instances_json:
                            inst_key = alt_key
                
                samples.append({
                    'scene': scene,
                    'base_name': base_name,
                    'pre_path': pre_path,
                    'post_path': post_path,
                    'label_path': label_path if os.path.exists(label_path) else None,
                    'inst_key': inst_key,
                })
        
        print(f"  [{self.split}] Total samples: {len(samples)}")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _load_boxes_from_json(self, inst_key):
        if self.instances_json is None or inst_key is None:
            return None, None, None
        entry = self.instances_json.get(inst_key)
        if entry is None:
            return None, None, None
        instances = entry.get('instances', [])
        if not instances:
            return None, None, None
        
        instance_class_names = self.config.get('branch_routing', {}).get('instance', [])
        target_names = self.config.get('target_names', [])
        instance_global_indices = []
        for cls_name in instance_class_names:
            if cls_name in target_names:
                instance_global_indices.append(target_names.index(cls_name))
        
        boxes, classes, states = [], [], []
        for inst in instances:
            bbox = inst.get('bbox')
            target_idx = inst.get('target_idx', 0)
            state_idx = inst.get('state_idx', 0)
            
            if instance_global_indices and target_idx not in instance_global_indices:
                continue
            
            if bbox is not None and len(bbox) == 4:
                boxes.append(bbox)
                if target_idx in instance_global_indices:
                    classes.append(instance_global_indices.index(target_idx))
                else:
                    classes.append(0)
                states.append(state_idx)
        
        if not boxes:
            return None, None, None
        return boxes, classes, states
    
    def _load_boxes_from_txt(self, box_path, img_h, img_w):
        boxes, classes, states = [], [], []
        with open(box_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    # If coordinates are in pixel format, normalize
                    if cx > 1.0 or cy > 1.0:
                        cx /= img_w
                        cy /= img_h
                        w /= img_w
                        h /= img_h
                    boxes.append([cx, cy, w, h])
                    classes.append(cls_id)
                    states.append(int(parts[5]) if len(parts) > 5 else 0)
        if not boxes:
            return None, None, None
        return boxes, classes, states
    
    def _decode_state(self, label):
        if self.state_decode_map:
            decoded = np.zeros_like(label)
            for src, dst in self.state_decode_map.items():
                decoded[label == src] = dst
            return decoded
        return label

    def _decode_label(self, label):
        if self.label_decode_map is None:
            return label
        decoded = np.zeros_like(label)
        for src, dst in self.label_decode_map.items():
            decoded[label == src] = dst
        return decoded
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        S = self.img_size
        
        # Load images with cache
        cache_key = sample['pre_path']
        if cache_key in self._img_cache:
            pre_img, post_img = self._img_cache[cache_key]
        else:
            pre_img = read_image(sample['pre_path'])
            post_img = read_image(sample['post_path'])
            if len(self._img_cache) < self._cache_max:
                self._img_cache[cache_key] = (pre_img, post_img)
        
        C, H, W = pre_img.shape
        
        # Resize to training size
        pre_pil = Image.fromarray(pre_img.transpose(1, 2, 0).astype(np.uint8))
        post_pil = Image.fromarray(post_img.transpose(1, 2, 0).astype(np.uint8))
        pre_resized = np.array(pre_pil.resize((S, S), Image.BILINEAR)).transpose(2, 0, 1).astype(np.float32) / 255.0
        post_resized = np.array(post_pil.resize((S, S), Image.BILINEAR)).transpose(2, 0, 1).astype(np.float32) / 255.0
        
        result = {
            'img_t1': torch.from_numpy(pre_resized).float(),
            'img_t2': torch.from_numpy(post_resized).float(),
            'scene': sample['scene'],
            'base_name': sample['base_name'],
            'orig_h': H,
            'orig_w': W,
            'change_state': torch.zeros((S, S), dtype=torch.long),
        }
        
        # Load label
        if sample['label_path']:
            try:
                label = read_label(sample['label_path'])
                if len(label.shape) == 3:
                    label = label[0]
                # Decode state from raw label BEFORE decoding target class
                if self.state_decode_map is not None:
                    if H != S or W != S:
                        raw_pil = Image.fromarray(label.astype(np.uint16))
                        raw_resized = np.array(raw_pil.resize((S, S), Image.NEAREST))
                    else:
                        raw_resized = label
                    state_map = self._decode_state(raw_resized)
                    result['change_state'] = torch.from_numpy(state_map).long()
                label = self._decode_label(label)
                lbl_pil = Image.fromarray(label.astype(np.uint8))
                lbl_resized = np.array(lbl_pil.resize((S, S), Image.NEAREST))
                result['label'] = torch.from_numpy(lbl_resized).long()
            except Exception:
                pass
        
        # Data augmentation (training only)
        if self.augment and 'label' in result:
            pre_img_np = result['img_t1'].numpy()
            post_img_np = result['img_t2'].numpy()
            pre_lbl = result['label'].numpy()
            post_lbl = pre_lbl.copy()
            boxes_aug = result.get('instance_boxes', None)

            pre_img_np, post_img_np, pre_lbl, post_lbl, boxes_aug = random_augment(
                pre_img_np, post_img_np, pre_lbl, post_lbl, boxes_aug)
            pre_img_np = color_jitter(pre_img_np)
            post_img_np = color_jitter(post_img_np)
            pre_img_np = random_erasing(pre_img_np)
            post_img_np = random_erasing(post_img_np)

            # Copy-paste rare classes
            rare_classes = [1, 2, 4, 7, 8, 9, 10, 12, 13, 14]
            pre_img_np, post_img_np, pre_lbl, post_lbl, cp_state_mask = copy_paste_both_labels(
                pre_img_np, post_img_np, pre_lbl, post_lbl,
                p=0.8, target_classes=rare_classes, max_paste=8)

            pre_img_np = gaussian_noise(pre_img_np, p=0.3, std=0.03)
            post_img_np = gaussian_noise(post_img_np, p=0.3, std=0.03)
            pre_img_np = random_blur(pre_img_np, p=0.2)
            post_img_np = random_blur(post_img_np, p=0.2)

            result['img_t1'] = torch.from_numpy(pre_img_np.copy()).float()
            result['img_t2'] = torch.from_numpy(post_img_np.copy()).float()
            result['label'] = torch.from_numpy(post_lbl.copy()).long()

            # Recompute state after augmentation
            if 'change_state' in result:
                state_map = result['change_state'].numpy()
                if self.state_decode_map is not None:
                    raw_lbl = read_label(sample['label_path'])
                    if len(raw_lbl.shape) == 3:
                        raw_lbl = raw_lbl[0]
                    if H != S or W != S:
                        raw_pil_aug = Image.fromarray(raw_lbl.astype(np.uint16))
                        raw_resized_aug = np.array(raw_pil_aug.resize((S, S), Image.NEAREST))
                    else:
                        raw_resized_aug = raw_lbl
                    _, _, raw_pre_aug, raw_post_aug, _ = random_augment(
                        pre_img_np, post_img_np, raw_resized_aug, raw_resized_aug, None)
                    state_map = self._decode_state(raw_post_aug)
                # Override for copy-paste regions
                if cp_state_mask is not None:
                    cp_valid = cp_state_mask >= 0
                    state_map[cp_valid] = cp_state_mask[cp_valid]
                result['change_state'] = torch.from_numpy(state_map).long()

            if boxes_aug is not None:
                result['instance_boxes'] = boxes_aug

        # Load instance boxes (normalized coordinates)
        boxes, classes, states = None, None, None
        
        if self.instances_json is not None:
            boxes, classes, states = self._load_boxes_from_json(sample.get('inst_key'))
        
        if boxes is None and self.boxes_dir:
            box_path = os.path.join(
                self.boxes_dir, sample['scene'], self.split, 'boxes',
                sample['base_name'] + self.label_suffix + '.txt'
            )
            if os.path.exists(box_path):
                boxes, classes, states = self._load_boxes_from_txt(box_path, H, W)
        
        if boxes is not None and len(boxes) > 0:
            result['instance_boxes'] = torch.tensor(boxes, dtype=torch.float32)
            result['instance_classes'] = torch.tensor(classes, dtype=torch.long)
            result['instance_states'] = torch.tensor(states, dtype=torch.long)
        
        return result
    
    @staticmethod
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return {}
        result = {
            'img_t1': torch.stack([b['img_t1'] for b in batch]),
            'img_t2': torch.stack([b['img_t2'] for b in batch]),
            'scene': [b.get('scene', '') for b in batch],
            'base_name': [b.get('base_name', '') for b in batch],
            'orig_h': [b.get('orig_h', 0) for b in batch],
            'orig_w': [b.get('orig_w', 0) for b in batch],
        }
        if all('label' in b for b in batch):
            result['label'] = torch.stack([b['label'] for b in batch])
        if all('change_state' in b for b in batch):
            result['change_state'] = torch.stack([b['change_state'] for b in batch])
        result['instance_boxes'] = [b.get('instance_boxes') for b in batch]
        result['instance_classes'] = [b.get('instance_classes') for b in batch]
        result['instance_states'] = [b.get('instance_states') for b in batch]
        return result


def create_dataloader(data_dir, config, split='train', batch_size=4,
                      num_workers=4, scenes=None, img_size=1024,
                      boxes_dir=None):
    dataset = FullImageDataset(
        data_dir=data_dir, config=config, split=split,
        scenes=scenes, img_size=img_size, boxes_dir=boxes_dir
    )
    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers, pin_memory=True,
        drop_last=(split == 'train'),
        collate_fn=FullImageDataset.collate_fn
    )














