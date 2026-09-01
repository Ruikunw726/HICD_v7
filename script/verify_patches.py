"""Verify SECOND_256 patch dataset integrity."""
import os, json, numpy as np
from PIL import Image

try:
    from osgeo import gdal
    gdal.UseExceptions()
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False

DST = '/mnt/c/SECOND_256'

def read_tif(path):
    if HAS_GDAL:
        ds = gdal.Open(path)
        arr = ds.ReadAsArray()
        ds = None
        return arr
    return np.array(Image.open(path))

issues = []

for split in ['train', 'val']:
    img_dir = os.path.join(DST, split, 'image')
    lbl_dir = os.path.join(DST, split, 'label')

    pre_files = sorted([f for f in os.listdir(img_dir) if f.endswith('_pre_war.tif')])
    post_files = sorted([f for f in os.listdir(img_dir) if f.endswith('_post_war.tif')])
    pre_lbl_files = sorted([f for f in os.listdir(lbl_dir) if f.endswith('_pre_target.tif')])
    post_lbl_files = sorted([f for f in os.listdir(lbl_dir) if f.endswith('_post_target.tif')])

    print(f"\n[{split}]")
    print(f"  Pre-war images: {len(pre_files)}")
    print(f"  Post-war images: {len(post_files)}")
    print(f"  Pre-target labels: {len(pre_lbl_files)}")
    print(f"  Post-target labels: {len(post_lbl_files)}")

    # Check pre/post pairing
    pre_bases = set(f.replace('_pre_war.tif', '') for f in pre_files)
    post_bases = set(f.replace('_post_war.tif', '') for f in post_files)
    missing_post = pre_bases - post_bases
    missing_pre = post_bases - pre_bases
    if missing_post:
        issues.append(f"  {split}: {len(missing_post)} pre without post")
    if missing_pre:
        issues.append(f"  {split}: {len(missing_pre)} post without pre")

    # Check label pairing
    pre_lbl_bases = set(f.replace('_pre_target.tif', '') for f in pre_lbl_files)
    post_lbl_bases = set(f.replace('_post_target.tif', '') for f in post_lbl_files)
    missing_post_lbl = pre_bases - post_lbl_bases
    if missing_post_lbl:
        issues.append(f"  {split}: {len(missing_post_lbl)} pre without post label")

    # Sample check: read a few files and verify shapes
    sample_files = pre_files[:3]
    for f in sample_files:
        img = read_tif(os.path.join(img_dir, f))
        base = f.replace('_pre_war.tif', '')
        lbl = read_tif(os.path.join(lbl_dir, base + '_pre_target.tif'))

        if img.ndim == 2:
            issues.append(f"  {split}/{f}: grayscale (expected RGB)")
        elif img.shape[0] != 3:
            issues.append(f"  {split}/{f}: {img.shape[0]} channels (expected 3)")

        if lbl.ndim != 2:
            issues.append(f"  {split}/{base}_pre_target.tif: shape {lbl.shape} (expected 2D)")

        if img.shape[1:] != (256, 256):
            issues.append(f"  {split}/{f}: size {img.shape[1:]} (expected 256x256)")
        if lbl.shape != (256, 256):
            issues.append(f"  {split}/{base}_pre_target.tif: size {lbl.shape} (expected 256x256)")

# Check instances.json
inst_path = os.path.join(DST, 'instances.json')
with open(inst_path, 'r') as f:
    instances = json.load(f)

print(f"\nInstances.json: {len(instances)} entries")
train_entries = sum(1 for k in instances if k.startswith('train/'))
val_entries = sum(1 for k in instances if k.startswith('val/'))
print(f"  Train: {train_entries}, Val: {val_entries}")

# Check box stats
total_boxes = 0
empty_boxes = 0
for k, v in instances.items():
    n = len(v.get('boxes', []))
    total_boxes += n
    if n == 0:
        empty_boxes += 1

print(f"  Total boxes: {total_boxes}")
print(f"  Entries with 0 boxes: {empty_boxes}")

# Check box coordinate ranges
all_boxes = []
for v in instances.values():
    all_boxes.extend(v.get('boxes', []))
if all_boxes:
    arr = np.array(all_boxes)
    print(f"  Box stats: cx=[{arr[:,0].min():.3f},{arr[:,0].max():.3f}], "
          f"cy=[{arr[:,1].min():.3f},{arr[:,1].max():.3f}], "
          f"w=[{arr[:,2].min():.3f},{arr[:,2].max():.3f}], "
          f"h=[{arr[:,3].min():.3f},{arr[:,3].max():.3f}]")

if issues:
    print(f"\n{'='*40}")
    print("ISSUES FOUND:")
    for i in issues:
        print(i)
else:
    print(f"\n{'='*40}")
    print("ALL CHECKS PASSED!")

print(f"\nDataset summary:")
print(f"  Train: {len(pre_files) if 'pre_files' in dir() else '?'} patches")
print(f"  Val: {val_entries} patches")
print(f"  Expected train: 2077*9={2077*9}, got {len(pre_files)}")
