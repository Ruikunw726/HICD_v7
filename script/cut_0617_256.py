"""
Cut 0617final 1024x1024 images into 256x256 patches (5x5 = 25 patches per image).
Original data untouched. Saves to a new folder.
"""
import os, numpy as np
from PIL import Image

SRC = "/mnt/d/CD/0617final"
DST = "/mnt/d/CD/0617final_256"
PATCH = 256
STRIDE = 192  # (1024-256)/(5-1) = 192

os.makedirs(DST, exist_ok=True)

for split in ["train", "val", "test"]:
    total_patches = 0
    total_with_change = 0
    for scene in ["Airports", "Ports", "Urban-Rural Areas"]:
        src_split = f"{SRC}/{scene}/{split}"
        if not os.path.exists(src_split):
            continue
        img_dir = f"{src_split}/image"
        lbl_dir = f"{src_split}/label"
        if not os.path.exists(img_dir):
            continue

        # Output dirs
        dst_split = f"{DST}/{split}"
        os.makedirs(f"{dst_split}/image", exist_ok=True)
        os.makedirs(f"{dst_split}/label", exist_ok=True)

        pre_files = sorted([f for f in os.listdir(img_dir) if f.endswith("_pre_war.tif")])
        print(f"[{split}/{scene}] {len(pre_files)} images")

        for fi, pre_name in enumerate(pre_files):
            base = pre_name.replace("_pre_war.tif", "")
            post_name = base + "_post_war.tif"
            lbl_name = base + "_target.tif"

            pre_path = f"{img_dir}/{pre_name}"
            post_path = f"{img_dir}/{post_name}"
            lbl_path = f"{lbl_dir}/{lbl_name}"

            if not os.path.exists(post_path):
                continue

            pre_arr = np.array(Image.open(pre_path))  # [H, W, C]
            post_arr = np.array(Image.open(post_path))
            H, W = pre_arr.shape[:2]

            if os.path.exists(lbl_path):
                lbl_arr = np.array(Image.open(lbl_path))
            else:
                lbl_arr = np.zeros((H, W), dtype=np.uint8)

            # Ensure 3 channels
            if pre_arr.ndim == 2:
                pre_arr = np.stack([pre_arr]*3, axis=-1)
                post_arr = np.stack([post_arr]*3, axis=-1)

            # Patch positions
            ys = list(range(0, H - PATCH + 1, STRIDE))
            if ys[-1] + PATCH < H:
                ys.append(H - PATCH)
            xs = list(range(0, W - PATCH + 1, STRIDE))
            if xs[-1] + PATCH < W:
                xs.append(W - PATCH)

            for yi, yp in enumerate(ys):
                for xi, xp in enumerate(xs):
                    pname = f"{base}_{yi:02d}_{xi:02d}"
                    pre_p = pre_arr[yp:yp+PATCH, xp:xp+PATCH]
                    post_p = post_arr[yp:yp+PATCH, xp:xp+PATCH]
                    lbl_p = lbl_arr[yp:yp+PATCH, xp:xp+PATCH]

                    Image.fromarray(pre_p.astype(np.uint8)).save(f"{dst_split}/image/{pname}_pre_war.tif")
                    Image.fromarray(post_p.astype(np.uint8)).save(f"{dst_split}/image/{pname}_post_war.tif")
                    Image.fromarray(lbl_p.astype(np.uint8)).save(f"{dst_split}/label/{pname}_target.tif")

                    total_patches += 1

        print(f"  {fi+1} images done")

    print(f"[{split}] Total patches: {total_patches}")

# Write split files
for split in ["train", "val", "test"]:
    split_dir = f"{DST}/{split}"
    if not os.path.exists(split_dir):
        continue
    img_dir = f"{split_dir}/image"
    if not os.path.exists(img_dir):
        continue
    pre_files = sorted([f.replace("_pre_war.tif","") for f in os.listdir(img_dir) if f.endswith("_pre_war.tif")])
    with open(f"{split_dir}/names.txt", "w") as f:
        for name in pre_files:
            f.write(name + "\n")
    print(f"[{split}] {len(pre_files)} patch names written")

# Verify
print("\n=== Verification ===")
for split in ["train", "val", "test"]:
    img_dir = f"{DST}/{split}/image"
    lbl_dir = f"{DST}/{split}/label"
    if not os.path.exists(img_dir): continue
    n_pre = len([f for f in os.listdir(img_dir) if f.endswith("_pre_war.tif")])
    n_post = len([f for f in os.listdir(img_dir) if f.endswith("_post_war.tif")])
    n_lbl = len([f for f in os.listdir(lbl_dir) if f.endswith("_target.tif")]) if os.path.exists(lbl_dir) else 0
    print(f"[{split}] pre={n_pre} post={n_post} label={n_lbl} match={n_pre==n_post==n_lbl}")

print("\nDone!")