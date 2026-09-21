import os, numpy as np
from PIL import Image
from collections import Counter

total_px = 0
state_counter = Counter()
changed_images = 0
total_images = 0

base = "/mnt/d/CD/0617final"
for scene in ["Airports", "Ports", "Urban-Rural Areas"]:
    label_dir = f"{base}/{scene}/train/label"
    if not os.path.exists(label_dir):
        continue
    files = [f for f in os.listdir(label_dir) if f.endswith("_target.tif")]
    for fname in files:
        lbl = np.array(Image.open(f"{label_dir}/{fname}"))
        total_images += 1
        total_px += lbl.size
        unique = np.unique(lbl)
        for u in unique:
            state_counter[int(u)] += (lbl == u).sum().item()
        # Check if any pixel is a changed state (encoded values 4-8 = damaged/reduced/added/extended)
        # But we need change_state, not target. Check if change_state files exist
    print(f"{scene}: {len(files)} images processed")

print(f"\nTotal images: {total_images}")
print(f"Total pixels: {total_px}")
print(f"\nTarget label distribution:")
for k in sorted(state_counter.keys()):
    pct = state_counter[k] / total_px * 100
    print(f"  Label {k}: {state_counter[k]} ({pct:.2f}%)")