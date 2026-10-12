"""
HICD v7 Dataset - Full image mode (no patch splitting)
Reads full images and resizes to a fixed training size.
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
                new_cx = boxes[:, 1].clone()
                new_cy = 1.0 - boxes[:, 0]
                new_w = boxes[:, 3].clone()
                new_h = boxes[:, 2].clone()
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



class ClassAwareCropSampler:
    """
    Class-aware cropping: bias crop center toward rare classes.
    Increases rare class frequency by 5-10x in training patches.
    """
    def __init__(self, crop_size=256, rare_prob=0.7):
        self.crop_size = crop_size
        self.rare_classes = [1, 2, 4, 7, 8, 9, 10, 12, 13, 14]
        self.rare_prob = rare_prob

    def get_crop_center(self, label, H, W):
        crop_h = min(self.crop_size, H)
        crop_w = min(self.crop_size, W)
        if np.random.rand() < self.rare_prob:
            rare_mask = np.zeros_like(label, dtype=bool)
            for cls in self.rare_classes:
                rare_mask |= (label == cls)
            rare_pixels = np.where(rare_mask)
            if len(rare_pixels[0]) > 0:
                idx = np.random.randint(len(rare_pixels[0]))
                cy = int(rare_pixels[0][idx])
                cx = int(rare_pixels[1][idx])
                cx = np.clip(cx, crop_w // 2, W - crop_w // 2)
                cy = np.clip(cy, crop_h // 2, H - crop_h // 2)
                return cx, cy
        cx = np.random.randint(crop_w // 2, max(crop_w // 2 + 1, W - crop_w // 2))
        cy = np.random.randint(crop_h // 2, max(crop_h // 2 + 1, H - crop_h // 2))
        return cx, cy


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


def copy_paste_both_labels(img1, img2, label1, label2, p=0.3, target_classes=None, max_paste=3,
                           state_map=None, background_class=15, legal_states=None):
    """Paste a paired source object into paired background, retaining its state.

    No state is invented from a random mode: both timestamps and their original
    target/state supervision travel together. Farmland (0) is never a vacancy.
    Without source states this augmentation is disabled rather than fabricating
    damaged/replaced/added labels. Returns an explicit state override mask.
    """
    if np.random.rand() > p or target_classes is None or state_map is None:
        return img1, img2, label1, label2, None
    C, H, W = img1.shape
    n_paste = np.random.randint(1, max_paste + 1)
    state_mask = np.full((H, W), -1, dtype=np.int64)
    updated_states = state_map.copy()
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
        dst_y = np.random.randint(0, H - patch_h + 1)
        dst_x = np.random.randint(0, W - patch_w + 1)
        source_mask = (label1[y1:y2, x1:x2] == cls) & (label2[y1:y2, x1:x2] == cls)
        source_states = updated_states[y1:y2, x1:x2].copy()
        allowed = (legal_states or {}).get(int(cls), range(6))
        source_mask &= np.isin(source_states, list(allowed))
        dst_slice = np.s_[dst_y:dst_y+patch_h, dst_x:dst_x+patch_w]
        vacant = (label1[dst_slice] == background_class) & (label2[dst_slice] == background_class)
        paste = source_mask & vacant
        if not paste.any():
            continue
        src1 = img1[:, y1:y2, x1:x2].copy()
        src2 = img2[:, y1:y2, x1:x2].copy()
        img1[:, dst_y:dst_y+patch_h, dst_x:dst_x+patch_w][:, paste] = src1[:, paste]
        img2[:, dst_y:dst_y+patch_h, dst_x:dst_x+patch_w][:, paste] = src2[:, paste]
        label1[dst_slice][paste] = cls
        label2[dst_slice][paste] = cls
        updated_states[dst_slice][paste] = source_states[paste]
        state_mask[dst_slice][paste] = source_states[paste]
    return img1, img2, label1, label2, state_mask

class FullImageDataset(Dataset):
    """
    Full-image dataset for V7. No patch splitting.
    
    Reads full images, resizes to `img_size`, and provides all
    instance annotations in normalized coordinates.
    """
    
    def __init__(self, data_dir, config, split='train', scenes=None,
                 img_size=1024, boxes_dir=None, pin_memory=True):
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
        self.crop_sampler = ClassAwareCropSampler(crop_size=img_size) if self.augment else None
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
        # Names.txt mode: always prefer names.txt if it exists
        names_file = os.path.join(self.data_dir, self.split, 'names.txt')
        if os.path.isfile(names_file):
            return self._collect_samples_from_names(names_file)

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
    
    def _collect_samples_from_names(self, names_file):
        """Load samples from names.txt, resolving image/label paths from source dirs."""
        with open(names_file) as f:
            names = [l.strip() for l in f if l.strip()]

        catalog_path = os.path.join(self.data_dir, self.split, 'resolved_samples.json')
        if os.path.isfile(catalog_path):
            # Versioned clean views pin exact source files; never silently search
            # siblings or substitute a differently labelled image on mismatch.
            import hashlib
            import re
            manifest_path = os.path.join(self.data_dir, 'split_manifest.json')
            with open(manifest_path, encoding='utf-8') as f:
                manifest = json.load(f)
            with open(catalog_path, 'rb') as f:
                catalog_bytes = f.read()
            if hashlib.sha256(catalog_bytes).hexdigest() != manifest['catalog_sha256'][self.split]:
                raise ValueError('Resolved split catalog changed')
            catalog = json.loads(catalog_bytes)
            if catalog.get('schema_version') != 'hicd.resolved_split.v1' or catalog.get('split') != self.split:
                raise ValueError('Unsupported or mismatched resolved split catalog')
            with open(names_file, 'rb') as f:
                names_hash = hashlib.sha256(f.read()).hexdigest()
            if names_hash != catalog['names_sha256'] or names_hash != manifest['names_sha256'][self.split]:
                raise ValueError('Clean split names changed')
            records = {record['base_name']: record for record in catalog['samples']}
            if len(names) != len(set(names)) or len(records) != len(catalog['samples']) or set(records) != set(names):
                raise ValueError('Names/catalog membership is not unique and identical')
            samples = []
            allowed_root = os.path.normcase(os.path.abspath(os.path.dirname(self.data_dir)))
            for name in names:
                record = dict(records[name])
                for key in ('pre_path', 'post_path', 'label_path'):
                    value = record.get(key)
                    if not isinstance(value, str) or not value:
                        raise ValueError(f'Incomplete pinned source path: {name}/{key}')
                    if os.name == 'nt':
                        match = re.match(r'^/mnt/([a-zA-Z])/(.*)$', value)
                        if match: value = match[1].upper() + ':/' + match[2]
                    else:
                        match = re.match(r'^([a-zA-Z]):[/\\](.*)$', value)
                        if match: value = '/mnt/' + match[1].lower() + '/' + match[2].replace('\\', '/')
                    if not os.path.isabs(value):
                        raise ValueError('Pinned source path must be absolute')
                    value = os.path.abspath(value)
                    if os.path.commonpath((allowed_root, os.path.normcase(value))) != allowed_root:
                        raise ValueError('Pinned source path escaped the data parent directory')
                    record[key] = value
                samples.append(record)
            print(f'  [{self.split}] Loaded {len(samples)} pinned samples from clean split catalog')
            return samples
        
        # Build path index once: map (split, filename) -> full path
        # Search self.data_dir and sibling dirs
        search_dirs = [self.data_dir]
        parent = os.path.dirname(self.data_dir)
        if os.path.isdir(parent):
            for d in os.listdir(parent):
                candidate = os.path.join(parent, d)
                if candidate != self.data_dir and os.path.isdir(os.path.join(candidate, self.split, 'image')):
                    search_dirs.append(candidate)
        
        # Build filename -> path lookup for images and labels
        # Search ALL original splits (train/val/test) since patches may have been reassigned
        img_map = {}
        lbl_map = {}
        for sd in search_dirs:
            for orig_split in ["train", "val", "test"]:
                img_dir = os.path.join(sd, orig_split, 'image')
                lbl_dir = os.path.join(sd, orig_split, 'label')
                if os.path.isdir(img_dir):
                    for fname in os.listdir(img_dir):
                        if fname not in img_map:
                            img_map[fname] = os.path.join(img_dir, fname)
                if os.path.isdir(lbl_dir):
                    for fname in os.listdir(lbl_dir):
                        if fname not in lbl_map:
                            lbl_map[fname] = os.path.join(lbl_dir, fname)
        
        samples = []
        for name in names:
            pre_fname = name + self.pre_suffix + self.image_ext
            post_fname = name + self.post_suffix + self.image_ext
            label_fname = name + self.label_suffix + self.label_ext
            
            pre_path = img_map.get(pre_fname)
            post_path = img_map.get(post_fname)
            label_path = lbl_map.get(label_fname)
            
            if pre_path is None:
                continue
            
            samples.append({
                'scene': '',
                'base_name': name,
                'pre_path': pre_path,
                'post_path': post_path or '',
                'label_path': label_path,
                'inst_key': None,
            })
        
        print(f"  [{self.split}] Loaded {len(samples)}/{len(names)} samples from names.txt")
        return samples

    def _compute_state_sampling_weights(self, rare_states=None, boost=10.0):
        """Compute per-sample weights using pre-computed scene_state_info.json.
        Samples from scenes with rare state classes get higher weight.
        No label file scanning needed - instant computation.
        """
        if rare_states is None:
            rare_states = [2, 4, 5]  # reduced, extended, replaced

        # Try to load pre-computed scene info
        info_path = os.path.join(self.data_dir, "scene_state_info.json")
        scene_info = {}
        if os.path.isfile(info_path):
            import json
            with open(info_path) as f:
                scene_info = json.load(f)

        weights = np.ones(len(self.samples), dtype=np.float64)
        n_boosted = 0
        for i, sample in enumerate(self.samples):
            scene = sample.get('base_name', '')
            # Extract scene key: remove last _XX_XX patch coords
            if '_' in scene:
                scene_key = scene.rsplit('_', 2)[0] if scene.count('_') >= 2 else scene
            else:
                scene_key = scene

            info = scene_info.get(scene_key, {})
            if info.get('has_rare', False):
                weights[i] = boost
                n_boosted += 1

        # Normalize so mean weight = 1
        weights /= weights.mean()
        print(f"  State-aware sampling: {n_boosted}/{len(weights)} samples boosted (from scene_state_info.json)")
        return weights

    def _compute_rare_class_sampling_weights(self, rare_class_ids=None, boost=50.0):
        """Compute per-sample weights for rare class oversampling.
        Samples containing rare classes (Bridge/Dock/Shelter) get much higher weight.

        Args:
            rare_class_ids: list of class IDs to oversample (decoded IDs)
            boost: weight multiplier for samples containing rare classes

        Returns:
            weights: numpy array of sampling weights
        """
        if rare_class_ids is None:
            rare_class_ids = [4, 7, 10]  # Bridge, Shelter, Dock (decoded IDs)

        target_names = self.config.get('pixel_target_names', self.config.get('target_names', []))
        rare_names = [target_names[i] if i < len(target_names) else str(i) for i in rare_class_ids]

        # Raw label values that map to rare classes (from label_decode_map)
        # Bridge(4): 18-22, Shelter(7): 33-37, Dock(10): 48-52
        raw_label_map = {
            4: set(range(18, 23)),   # Bridge
            7: set(range(33, 38)),   # Shelter
            10: set(range(48, 53)),  # Dock
        }

        # Cache file for rare class sample indices
        cache_path = os.path.join(self.data_dir, f"rare_class_samples_{self.split}.json")

        # Try to load cached results
        if os.path.isfile(cache_path):
            with open(cache_path) as f:
                cache = json.load(f)
            rare_sample_indices = set(cache.get('rare_indices', []))
            class_sample_counts = cache.get('class_counts', {})
        else:
            # Scan label files to find samples containing rare classes
            print(f"  Scanning labels for rare classes {rare_names}...")
            print(f"  Raw label ranges: {[(n, sorted(raw_label_map.get(i, []))) for i, n in zip(rare_class_ids, rare_names)]}")
            rare_sample_indices = set()
            class_sample_counts = {name: 0 for name in rare_names}

            for i, sample in enumerate(self.samples):
                label_path = sample.get('label_path')
                if label_path is None or not os.path.isfile(label_path):
                    continue

                try:
                    label = read_label(label_path)
                    unique_vals = set(np.unique(label).tolist())
                    for cls_id, cls_name in zip(rare_class_ids, rare_names):
                        raw_vals = raw_label_map.get(cls_id, set())
                        if unique_vals & raw_vals:  # Intersection
                            rare_sample_indices.add(i)
                            class_sample_counts[cls_name] += 1
                except Exception:
                    continue

                # Progress indicator
                if (i + 1) % 5000 == 0:
                    print(f"    Scanned {i+1}/{len(self.samples)} samples...")

            # Cache results
            cache = {
                'rare_indices': list(rare_sample_indices),
                'class_counts': class_sample_counts,
                'rare_class_ids': rare_class_ids,
                'rare_names': rare_names,
            }
            with open(cache_path, 'w') as f:
                json.dump(cache, f)

        # Print statistics
        print(f"  Rare class sampling ({self.split}):")
        for name in rare_names:
            count = class_sample_counts.get(name, 0)
            print(f"    {name}: {count} samples")
        print(f"    Total rare samples: {len(rare_sample_indices)}/{len(self.samples)}")

        # Compute weights
        weights = np.ones(len(self.samples), dtype=np.float64)
        for i in rare_sample_indices:
            weights[i] = boost

        # Normalize so mean weight = 1
        weights /= weights.mean()
        print(f"  Rare class boost: {boost}x for {len(rare_sample_indices)} samples")
        return weights

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
            decoded = np.full(label.shape, 255, dtype=np.int64)
            for src, dst in self.state_decode_map.items():
                decoded[label == src] = dst
            return decoded
        return label

    def _decode_label(self, label):
        if self.label_decode_map is None:
            return label
        decoded = np.full(label.shape, 255, dtype=np.int64)
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
        if post_img.shape != pre_img.shape:
            raise ValueError(f'Paired image shape mismatch: {sample["base_name"]}')
        crop_coords = None
        raw_label = None
        if sample.get('label_path'):
            raw_label = read_label(sample['label_path'])
            if raw_label.ndim == 3:
                raw_label = raw_label[0]
            if raw_label.shape != (H, W):
                raise ValueError(f'Image/label shape mismatch: {sample["base_name"]}')
        
        # Class-aware cropping (before resize)
        if self.crop_sampler is not None and raw_label is not None and (H > S or W > S):
            cx, cy = self.crop_sampler.get_crop_center(self._decode_label(raw_label), H, W)
            crop_h = min(S, H)
            crop_w = min(S, W)
            y1 = min(max(0, cy - crop_h // 2), H - crop_h)
            y2 = y1 + crop_h
            x1 = min(max(0, cx - crop_w // 2), W - crop_w)
            x2 = x1 + crop_w
            crop_coords = (y1, y2, x1, x2)
            pre_img = pre_img[:, y1:y2, x1:x2]
            post_img = post_img[:, y1:y2, x1:x2]
            raw_label = raw_label[y1:y2, x1:x2]
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
        if raw_label is not None:
            raw_resized = np.array(Image.fromarray(raw_label.astype(np.int32)).resize((S, S), Image.NEAREST))
            result['label'] = torch.from_numpy(self._decode_label(raw_resized).copy()).long()
            if self.state_decode_map is not None:
                result['change_state'] = torch.from_numpy(self._decode_state(raw_resized).copy()).long()
        
        # Data augmentation (training only)
        if self.augment and 'label' in result:
            pre_img_np = result['img_t1'].numpy()
            post_img_np = result['img_t2'].numpy()
            pre_lbl = result['label'].numpy()
            # One geometry draw for both images, target and state supervision.
            state_map = result['change_state'].numpy()
            boxes_aug = result.get('instance_boxes', None)

            pre_img_np, post_img_np, pre_lbl, state_map, boxes_aug = random_augment(
                pre_img_np, post_img_np, pre_lbl, state_map, boxes_aug)
            post_lbl = pre_lbl.copy()
            pre_img_np = color_jitter(pre_img_np)
            post_img_np = color_jitter(post_img_np)
            pre_img_np = random_erasing(pre_img_np)
            post_img_np = random_erasing(post_img_np)

            # Copy-paste rare classes
            rare_classes = [1, 2, 4, 7, 8, 9, 10, 12, 13, 14]
            legal_states = {}
            for raw, target in (self.label_decode_map or {}).items():
                state = (self.state_decode_map or {}).get(raw, 255)
                if target != 255 and 0 <= state < 6:
                    legal_states.setdefault(target, set()).add(state)
            target_names = self.config.get('pixel_target_names', self.config.get('target_names', []))
            background_class = next((i for i, name in enumerate(target_names) if name.lower() == 'background'), None)
            pre_img_np, post_img_np, pre_lbl, post_lbl, cp_state_mask = copy_paste_both_labels(
                pre_img_np, post_img_np, pre_lbl, post_lbl,
                p=0.8 if background_class is not None else 0.0, target_classes=rare_classes, max_paste=8,
                state_map=state_map, background_class=background_class, legal_states=legal_states)

            pre_img_np = gaussian_noise(pre_img_np, p=0.3, std=0.03)
            post_img_np = gaussian_noise(post_img_np, p=0.3, std=0.03)
            pre_img_np = random_blur(pre_img_np, p=0.2)
            post_img_np = random_blur(post_img_np, p=0.2)

            result['img_t1'] = torch.from_numpy(pre_img_np.copy()).float()
            result['img_t2'] = torch.from_numpy(post_img_np.copy()).float()
            result['label'] = torch.from_numpy(post_lbl.copy()).long()

            if cp_state_mask is not None:
                cp_valid = cp_state_mask >= 0
                state_map[cp_valid] = cp_state_mask[cp_valid]
            result['change_state'] = torch.from_numpy(state_map.copy()).long()

            result['_crop_coords'] = crop_coords
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
                      boxes_dir=None, pin_memory=True, state_aware=False,
                      rare_class_oversample=False, rare_class_boost=50.0):
    dataset = FullImageDataset(
        data_dir=data_dir, config=config, split=split,
        scenes=scenes, img_size=img_size, boxes_dir=boxes_dir
    )
    sampler = None
    shuffle = (split == 'train')
    if split == 'train':
        if rare_class_oversample:
            # Oversample rare classes (Bridge/Dock/Shelter)
            weights = dataset._compute_rare_class_sampling_weights(
                rare_class_ids=[4, 7, 10],  # Bridge, Shelter, Dock
                boost=rare_class_boost
            )
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(weights).double(),
                num_samples=len(weights),
                replacement=True
            )
            shuffle = False
        elif state_aware:
            # Original state-aware sampling
            weights = dataset._compute_state_sampling_weights(
                rare_states=[2, 4, 5], boost=10.0)
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(weights).double(),
                num_samples=len(weights),
                replacement=True
            )
            shuffle = False
    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=(split == 'train'),
        collate_fn=FullImageDataset.collate_fn
    )














