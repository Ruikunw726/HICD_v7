"""
Cut XBD 1024x1024 images into 256x256 patches (stride=192, 25 patches per image).
XBD structure: disaster/split/{image,label}/
Images: PNG, Labels: TIF
Instances: per-disaster instances.json
"""
import os, json, numpy as np
from PIL import Image

SRC_DIR = '/mnt/d/CD/xbd'
DST_DIR = '/mnt/d/CD/XBD_256'
PATCH_SIZE = 256
STRIDE = 192

def get_patches(size):
    positions = []
    pos = 0
    while pos + PATCH_SIZE <= size:
        positions.append(pos)
        pos += STRIDE
    if positions[-1] + PATCH_SIZE < size:
        positions.append(size - PATCH_SIZE)
    return positions

def merge_instances(src_dir):
    """Merge all disaster-level instances.json into one dict."""
    merged = {}
    for disaster in os.listdir(src_dir):
        inst_path = os.path.join(src_dir, disaster, 'instances.json')
        if os.path.exists(inst_path):
            with open(inst_path, 'r') as f:
                data = json.load(f)
            for key, val in data.items():
                merged[f"{disaster}/{key}"] = val
    return merged

# Merge all instances
print("Merging instances.json from all disasters...")
all_instances = merge_instances(SRC_DIR)
print(f"Total instances: {len(all_instances)}")

new_instances = {}
total_patches = 0
total_boxes = 0

for split in ['train', 'val', 'test']:
    dst_split = os.path.join(DST_DIR, split)
    os.makedirs(os.path.join(dst_split, 'image'), exist_ok=True)
    os.makedirs(os.path.join(dst_split, 'label'), exist_ok=True)

    split_patches = 0
    split_boxes = 0

    for disaster in sorted(os.listdir(SRC_DIR)):
        disaster_dir = os.path.join(SRC_DIR, disaster, split)
        if not os.path.isdir(disaster_dir):
            continue

        img_dir = os.path.join(disaster_dir, 'image')
        lbl_dir = os.path.join(disaster_dir, 'label')
        if not os.path.exists(img_dir):
            continue

        pre_files = sorted([f for f in os.listdir(img_dir) if f.endswith('_pre_war.png')])

        for fi, pre_name in enumerate(pre_files):
            base = pre_name.replace('_pre_war.png', '')
            post_name = base + '_post_war.png'

            pre_path = os.path.join(img_dir, pre_name)
            post_path = os.path.join(img_dir, post_name)
            if not os.path.exists(post_path):
                continue

            pre_arr = np.array(Image.open(pre_path))
            post_arr = np.array(Image.open(post_path))

            if pre_arr.ndim == 2:
                pre_arr = np.stack([pre_arr]*3, axis=-1)
                post_arr = np.stack([post_arr]*3, axis=-1)
            H, W = pre_arr.shape[:2]

            # Label: single target (post-war)
            lbl_name = base + '_target.tif'
            lbl_path = os.path.join(lbl_dir, lbl_name)
            if os.path.exists(lbl_path):
                lbl_raw = Image.open(lbl_path)
                if lbl_raw.size != (W, H):
                    lbl_raw = lbl_raw.resize((W, H), Image.NEAREST)
                lbl_arr = np.array(lbl_raw)
            else:
                lbl_arr = np.zeros((H, W), dtype=np.uint8)

            y_pos = get_patches(H)
            x_pos = get_patches(W)

            # Instance boxes
            inst_key = f"{disaster}/{split}/{pre_name}"
            inst_data = all_instances.get(inst_key, {'boxes': [], 'classes': [], 'states': []})
            gt_boxes = np.array(inst_data.get('boxes', []))
            gt_classes = inst_data.get('classes', [])
            gt_states = inst_data.get('states', [])

            if len(gt_boxes) > 0:
                abs_boxes = np.zeros_like(gt_boxes)
                abs_boxes[:, 0] = (gt_boxes[:, 0] - gt_boxes[:, 2] / 2) * W
                abs_boxes[:, 1] = (gt_boxes[:, 1] - gt_boxes[:, 3] / 2) * H
                abs_boxes[:, 2] = (gt_boxes[:, 0] + gt_boxes[:, 2] / 2) * W
                abs_boxes[:, 3] = (gt_boxes[:, 1] + gt_boxes[:, 3] / 2) * H
            else:
                abs_boxes = np.zeros((0, 4))

            for yi, yp in enumerate(y_pos):
                for xi, xp in enumerate(x_pos):
                    patch_name = f"{disaster}_{base}_{yi}_{xi}"

                    pre_patch = pre_arr[yp:yp+PATCH_SIZE, xp:xp+PATCH_SIZE]
                    post_patch = post_arr[yp:yp+PATCH_SIZE, xp:xp+PATCH_SIZE]
                    lbl_patch = lbl_arr[yp:yp+PATCH_SIZE, xp:xp+PATCH_SIZE]

                    Image.fromarray(pre_patch).save(os.path.join(dst_split, 'image', patch_name + '_pre_war.png'))
                    Image.fromarray(post_patch).save(os.path.join(dst_split, 'image', patch_name + '_post_war.png'))
                    Image.fromarray(lbl_patch).save(os.path.join(dst_split, 'label', patch_name + '_target.tif'))

                    patch_boxes, patch_classes, patch_states = [], [], []
                    for bi in range(len(abs_boxes)):
                        bx1, by1, bx2, by2 = abs_boxes[bi]
                        ix1, iy1 = max(bx1, xp), max(by1, yp)
                        ix2, iy2 = min(bx2, xp+PATCH_SIZE), min(by2, yp+PATCH_SIZE)
                        if ix1 < ix2 and iy1 < iy2:
                            box_area = (bx2-bx1) * (by2-by1)
                            overlap = (ix2-ix1) * (iy2-iy1)
                            if overlap / max(box_area, 1e-6) >= 0.5:
                                ncx = np.clip(((ix1+ix2)/2 - xp) / PATCH_SIZE, 0, 1)
                                ncy = np.clip(((iy1+iy2)/2 - yp) / PATCH_SIZE, 0, 1)
                                nw = np.clip((ix2-ix1) / PATCH_SIZE, 0, 1)
                                nh = np.clip((iy2-iy1) / PATCH_SIZE, 0, 1)
                                if nw > 0.01 and nh > 0.01:
                                    patch_boxes.append([float(ncx), float(ncy), float(nw), float(nh)])
                                    patch_classes.append(gt_classes[bi])
                                    patch_states.append(gt_states[bi])

                    new_instances[f"{split}/{patch_name}_pre_war.png"] = {
                        'boxes': patch_boxes, 'classes': patch_classes, 'states': patch_states
                    }
                    split_boxes += len(patch_boxes)
                    split_patches += 1

        if (fi+1) % 50 == 0:
            print(f"  [{split}] {disaster}: {fi+1}/{len(pre_files)} images")

    total_patches += split_patches
    total_boxes += split_boxes
    print(f"[{split}] Done: {split_patches} patches, {split_boxes} boxes")

with open(os.path.join(DST_DIR, 'instances.json'), 'w') as f:
    json.dump(new_instances, f)

print(f"\nTotal: {total_patches} patches, {total_boxes} boxes")
print(f"Output: {DST_DIR}")


