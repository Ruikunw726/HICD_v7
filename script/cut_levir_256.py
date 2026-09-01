"""
Cut LEVIR-CD 1024x1024 into 256x256 patches (stride=192, 5x5=25 patches per image).
"""
import os, numpy as np
from PIL import Image

SRC_DIR = '/mnt/d/CD/LEVIR-CD'
DST_DIR = '/mnt/d/CD/LEVIR-CD_256'
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

total_patches = 0

for split in ['train', 'val', 'test']:
    src_split = os.path.join(SRC_DIR, split)
    dst_split = os.path.join(DST_DIR, split)
    os.makedirs(os.path.join(dst_split, 'A'), exist_ok=True)
    os.makedirs(os.path.join(dst_split, 'B'), exist_ok=True)
    os.makedirs(os.path.join(dst_split, 'label'), exist_ok=True)

    a_files = sorted([f for f in os.listdir(os.path.join(src_split, 'A')) if f.endswith('.png')])
    split_patches = 0

    for fi, fname in enumerate(a_files):
        base = fname.replace('.png', '')
        a_arr = np.array(Image.open(os.path.join(src_split, 'A', fname)))
        b_arr = np.array(Image.open(os.path.join(src_split, 'B', fname)))
        lbl_arr = np.array(Image.open(os.path.join(src_split, 'label', fname)))

        H, W = a_arr.shape[:2]
        y_pos = get_patches(H)
        x_pos = get_patches(W)

        for yp in y_pos:
            for xp in x_pos:
                patch_name = f"{base}_{yp}_{xp}"
                Image.fromarray(a_arr[yp:yp+PATCH_SIZE, xp:xp+PATCH_SIZE]).save(
                    os.path.join(dst_split, 'A', patch_name + '.png'))
                Image.fromarray(b_arr[yp:yp+PATCH_SIZE, xp:xp+PATCH_SIZE]).save(
                    os.path.join(dst_split, 'B', patch_name + '.png'))
                Image.fromarray(lbl_arr[yp:yp+PATCH_SIZE, xp:xp+PATCH_SIZE]).save(
                    os.path.join(dst_split, 'label', patch_name + '.png'))
                split_patches += 1

        if (fi+1) % 100 == 0:
            print(f"  [{split}] {fi+1}/{len(a_files)}")

    total_patches += split_patches
    print(f"[{split}] Done: {split_patches} patches ({len(y_pos)}x{len(x_pos)} per image)")

print(f"\nTotal: {total_patches} patches")
