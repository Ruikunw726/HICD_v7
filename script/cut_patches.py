"""
Cut 512x512 SECOND images into 256x256 patches with stride=128 (9 patches per image).
Outputs to a new folder, original data untouched.
Also transforms instance boxes to patch coordinates.
"""
import os, json, shutil, numpy as np
from PIL import Image

try:
    from osgeo import gdal
    gdal.UseExceptions()
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False

SRC_DIR = '/mnt/c/SECOND_6class_v2'
DST_DIR = '/mnt/c/SECOND_256'
PATCH_SIZE = 256
STRIDE = 128

def read_tif(path):
    if HAS_GDAL:
        ds = gdal.Open(path)
        arr = ds.ReadAsArray()
        ds = None
        return arr
    return np.array(Image.open(path))

def write_tif(path, arr):
    """Write array as GeoTIFF (or PNG as fallback)."""
    if HAS_GDAL:
        driver = gdal.GetDriverByName('GTiff')
        if arr.ndim == 2:
            h, w = arr.shape
            ds = driver.Create(path, w, h, 1, gdal.GDT_Byte)
            ds.GetRasterBand(1).WriteArray(arr)
            ds = None
        else:
            c, h, w = arr.shape
            ds = driver.Create(path, w, h, c, gdal.GDT_Byte)
            for i in range(c):
                ds.GetRasterBand(i+1).WriteArray(arr[i])
            ds = None
    else:
        if arr.ndim == 2:
            Image.fromarray(arr.astype(np.uint8)).save(path)
        else:
            Image.fromarray(arr.transpose(1, 2, 0).astype(np.uint8)).save(path)

def get_patches(shape_h, stride):
    """Get patch start positions for given image dimension."""
    positions = []
    pos = 0
    while pos + PATCH_SIZE <= shape_h:
        positions.append(pos)
        pos += stride
    # If last patch doesn't reach end, add overlapping patch at the end
    if positions[-1] + PATCH_SIZE < shape_h:
        positions.append(shape_h - PATCH_SIZE)
    return positions

# Load instances.json
instances_path = os.path.join(SRC_DIR, 'instances.json')
if os.path.exists(instances_path):
    with open(instances_path, 'r') as f:
        instances_json = json.load(f)
else:
    instances_json = {}

new_instances = {}

for split in ['train', 'val']:
    src_split = os.path.join(SRC_DIR, split)
    dst_split = os.path.join(DST_DIR, split)
    os.makedirs(os.path.join(dst_split, 'image'), exist_ok=True)
    os.makedirs(os.path.join(dst_split, 'label'), exist_ok=True)

    img_dir = os.path.join(src_split, 'image')
    lbl_dir = os.path.join(src_split, 'label')

    if not os.path.exists(img_dir):
        continue

    # Find all pre_war images
    pre_files = sorted([f for f in os.listdir(img_dir) if f.endswith('_pre_war.tif')])
    print(f"\n[{split}] Found {len(pre_files)} image pairs")

    total_patches = 0
    total_boxes = 0

    for fi, pre_name in enumerate(pre_files):
        base = pre_name.replace('_pre_war.tif', '')
        post_name = base + '_post_war.tif'
        pre_path = os.path.join(img_dir, pre_name)
        post_path = os.path.join(img_dir, post_name)

        if not os.path.exists(post_path):
            continue

        # Read images
        pre_arr = read_tif(pre_path)  # [C, H, W] or [H, W]
        post_arr = read_tif(post_path)

        # Ensure [C, H, W]
        if pre_arr.ndim == 2:
            pre_arr = np.stack([pre_arr] * 3, axis=0)
            post_arr = np.stack([post_arr] * 3, axis=0)
        if pre_arr.shape[0] > 3:
            pre_arr = pre_arr[:3]
            post_arr = post_arr[:3]

        _, H, W = pre_arr.shape

        # Read labels
        pre_lbl_name = base + '_pre_target.tif'
        post_lbl_name = base + '_post_target.tif'
        pre_lbl_path = os.path.join(lbl_dir, pre_lbl_name)
        post_lbl_path = os.path.join(lbl_dir, post_lbl_name)

        pre_lbl = read_tif(pre_lbl_path) if os.path.exists(pre_lbl_path) else np.zeros((H, W), dtype=np.uint8)
        post_lbl = read_tif(post_lbl_path) if os.path.exists(post_lbl_path) else np.zeros((H, W), dtype=np.uint8)

        # Get patch positions
        y_positions = get_patches(H, STRIDE)
        x_positions = get_patches(W, STRIDE)

        # Get instance boxes for this image
        inst_key = f"{split}/{base}_pre_war.tif"
        inst_data = instances_json.get(inst_key, {'boxes': [], 'classes': [], 'states': []})
        gt_boxes = np.array(inst_data.get('boxes', []))  # [N, 4] in [cx, cy, w, h] normalized
        gt_classes = inst_data.get('classes', [])
        gt_states = inst_data.get('states', [])

        if len(gt_boxes) > 0:
            # Convert to absolute [x1, y1, x2, y2]
            abs_boxes = np.zeros_like(gt_boxes)
            abs_boxes[:, 0] = (gt_boxes[:, 0] - gt_boxes[:, 2] / 2) * W
            abs_boxes[:, 1] = (gt_boxes[:, 1] - gt_boxes[:, 3] / 2) * H
            abs_boxes[:, 2] = (gt_boxes[:, 0] + gt_boxes[:, 2] / 2) * W
            abs_boxes[:, 3] = (gt_boxes[:, 1] + gt_boxes[:, 3] / 2) * H
        else:
            abs_boxes = np.zeros((0, 4))

        # Cut patches
        for yi, yp in enumerate(y_positions):
            for xi, xp in enumerate(x_positions):
                patch_name = f"{base}_{yi}_{xi}"

                # Extract image patches
                pre_patch = pre_arr[:, yp:yp+PATCH_SIZE, xp:xp+PATCH_SIZE]
                post_patch = post_arr[:, yp:yp+PATCH_SIZE, xp:xp+PATCH_SIZE]

                # Extract label patches
                pre_lbl_patch = pre_lbl[yp:yp+PATCH_SIZE, xp:xp+PATCH_SIZE]
                post_lbl_patch = post_lbl[yp:yp+PATCH_SIZE, xp:xp+PATCH_SIZE]

                # Save
                write_tif(os.path.join(dst_split, 'image', patch_name + '_pre_war.tif'), pre_patch)
                write_tif(os.path.join(dst_split, 'image', patch_name + '_post_war.tif'), post_patch)
                write_tif(os.path.join(dst_split, 'label', patch_name + '_pre_target.tif'), pre_lbl_patch)
                write_tif(os.path.join(dst_split, 'label', patch_name + '_post_target.tif'), post_lbl_patch)

                # Transform boxes to patch coordinates
                patch_boxes = []
                patch_classes = []
                patch_states = []

                for bi in range(len(abs_boxes)):
                    bx1, by1, bx2, by2 = abs_boxes[bi]

                    # Check overlap with patch
                    ix1 = max(bx1, xp)
                    iy1 = max(by1, yp)
                    ix2 = min(bx2, xp + PATCH_SIZE)
                    iy2 = min(by2, yp + PATCH_SIZE)

                    if ix1 < ix2 and iy1 < iy2:
                        # Compute overlap ratio
                        box_area = (bx2 - bx1) * (by2 - by1)
                        overlap_area = (ix2 - ix1) * (iy2 - iy1)
                        overlap_ratio = overlap_area / max(box_area, 1e-6)

                        if overlap_ratio >= 0.5:  # Keep if >=50% in patch
                            # Convert to patch-relative normalized coordinates
                            ncx = ((ix1 + ix2) / 2 - xp) / PATCH_SIZE
                            ncy = ((iy1 + iy2) / 2 - yp) / PATCH_SIZE
                            nw = (ix2 - ix1) / PATCH_SIZE
                            nh = (iy2 - iy1) / PATCH_SIZE

                            # Clamp to [0, 1]
                            ncx = np.clip(ncx, 0, 1)
                            ncy = np.clip(ncy, 0, 1)
                            nw = np.clip(nw, 0, 1)
                            nh = np.clip(nh, 0, 1)

                            if nw > 0.01 and nh > 0.01:
                                patch_boxes.append([float(ncx), float(ncy), float(nw), float(nh)])
                                patch_classes.append(gt_classes[bi])
                                patch_states.append(gt_states[bi])

                patch_key = f"{split}/{patch_name}_pre_war.tif"
                new_instances[patch_key] = {
                    'boxes': patch_boxes,
                    'classes': patch_classes,
                    'states': patch_states
                }
                total_boxes += len(patch_boxes)
                total_patches += 1

        if (fi + 1) % 100 == 0:
            print(f"  Processed {fi+1}/{len(pre_files)} images, {total_patches} patches, {total_boxes} boxes")

    print(f"  Done: {total_patches} patches, {total_boxes} boxes")

# Save new instances.json
new_instances_path = os.path.join(DST_DIR, 'instances.json')
with open(new_instances_path, 'w') as f:
    json.dump(new_instances, f)
print(f"\nSaved {new_instances_path} with {len(new_instances)} entries")

# Summary
print(f"\n{'='*60}")
print(f"Patch extraction complete!")
print(f"Source: {SRC_DIR}")
print(f"Output: {DST_DIR}")
print(f"Patches per image: {len(y_positions) * len(x_positions)}")
print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}, stride: {STRIDE}")
print(f"{'='*60}")
