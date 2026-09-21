import os, numpy as np
from PIL import Image
from collections import Counter

# State decode map from config (raw label value -> state)
state_decode = {0:6,1:0,2:0,3:0,4:1,5:2,6:3,7:4,8:0,9:1,10:2,11:3,12:4,13:0,14:1,15:2,16:3,17:4,18:0,19:1,20:2,21:3,22:4,23:0,24:1,25:2,26:3,27:4,28:0,29:1,30:2,31:3,32:4,33:0,34:1,35:2,36:3,37:4,38:0,39:1,40:2,41:3,42:4,43:0,44:1,45:2,46:3,47:4,48:0,49:1,50:2,51:3,52:4,53:0,54:1,55:2,56:3,57:0,58:1,59:2,60:3,61:5,62:0,63:1,64:2,65:3,66:5,67:6,68:6}

state_px = Counter()
changed_images = 0
total_images = 0
images_with_change = []

base = "/mnt/d/CD/0617final"
for scene in ["Airports", "Ports", "Urban-Rural Areas"]:
    label_dir = f"{base}/{scene}/train/label"
    if not os.path.exists(label_dir):
        continue
    files = sorted([f for f in os.listdir(label_dir) if f.endswith("_target.tif")])
    for fname in files:
        lbl = np.array(Image.open(f"{label_dir}/{fname}"))
        total_images += 1
        # Decode states
        states = np.zeros_like(lbl)
        for raw_val, st in state_decode.items():
            states[lbl == raw_val] = st
        for s in range(7):
            c = (states == s).sum()
            if c > 0:
                state_px[s] += int(c)
        # Check if image has any changed pixels (state 1-5)
        has_change = ((states >= 1) & (states <= 5)).any()
        if has_change:
            changed_images += 1
            images_with_change.append(fname)
    print(f"{scene}: {len(files)} images done")

total_px = sum(state_px.values())
changed_px = sum(state_px[s] for s in range(1,6))

print(f"\n=== Training Set Statistics ===")
print(f"Total images: {total_images}")
print(f"Images with change: {changed_images} ({changed_images/total_images*100:.1f}%)")
print(f"Total pixels: {total_px}")
print(f"Changed pixels (state 1-5): {changed_px} ({changed_px/total_px*100:.2f}%)")
print(f"\nState distribution:")
state_names = ["no-change","damaged","reduced","added","extended","replaced","none"]
for s in range(7):
    print(f"  {s} {state_names[s]}: {state_px[s]} ({state_px[s]/total_px*100:.2f}%)")