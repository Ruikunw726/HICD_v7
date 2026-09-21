import os
import numpy as np
from collections import defaultdict

DATA_DIR = '/mnt/d/CD/0617final'
SCENES = ['Airports', 'Ports', 'Urban-Rural Areas']
SEED = 42
np.random.seed(SEED)

LABEL_MAP = {0:255, 1:0, 2:0, 3:1, 4:1, 5:1, 6:1, 7:1, 8:2, 9:2, 10:2, 11:2, 12:2,
             13:3, 14:3, 15:3, 16:3, 17:3, 18:4, 19:4, 20:4, 21:4, 22:4, 23:5, 24:5,
             25:5, 26:5, 27:5, 28:6, 29:6, 30:6, 31:6, 32:6, 33:7, 34:7, 35:7, 36:7,
             37:7, 38:8, 39:8, 40:8, 41:8, 42:8, 43:9, 44:9, 45:9, 46:9, 47:9, 48:10,
             49:10, 50:10, 51:10, 52:10, 53:11, 54:11, 55:11, 56:11, 57:12, 58:12, 59:12,
             60:12, 61:12, 62:13, 63:13, 64:13, 65:13, 66:13, 67:14, 68:14}
TARGET_NAMES = ['Farmland','Runway','Taxiway','Apron','Bridge','Highway','Building',
                'Shelter','Tower','Pier','Dock','Tank','Aircraft','Vessel','Crater']

# Step 1: collect all sample names per scene
print("Collecting sample names...")
scene_samples = {}
for scene in SCENES:
    scene_dir = os.path.join(DATA_DIR, scene)
    all_names = set()
    for split in ['train', 'val', 'test']:
        txt = os.path.join(scene_dir, split, f'{split}.txt')
        if os.path.exists(txt):
            with open(txt) as f:
                for line in f:
                    n = line.strip()
                    if n:
                        all_names.add(n)
    scene_samples[scene] = sorted(all_names)
    print(f"  {scene}: {len(scene_samples[scene])} samples")

# Step 2: scan labels using PIL (faster for single-channel TIF)
print("\nScanning labels...")
from PIL import Image

scene_data = {}  # scene -> [(name, frozenset(targets))]
for scene in SCENES:
    data = []
    scene_dir = os.path.join(DATA_DIR, scene)
    for i, name in enumerate(scene_samples[scene]):
        lbl_path = None
        for split in ['train', 'val', 'test']:
            p = os.path.join(scene_dir, split, 'label', f'{name}_target.tif')
            if os.path.exists(p):
                lbl_path = p
                break
        if not lbl_path:
            continue
        
        try:
            img = Image.open(lbl_path)
            arr = np.array(img)
            unique = np.unique(arr)
            targets = set()
            for v in unique:
                tid = LABEL_MAP.get(int(v), 255)
                if tid != 255:
                    targets.add(tid)
            data.append((name, frozenset(targets)))
        except:
            data.append((name, frozenset()))
        
        if (i+1) % 500 == 0:
            print(f"  {scene}: {i+1}/{len(scene_samples[scene])}")
    
    scene_data[scene] = data
    print(f"  {scene}: done ({len(data)} samples)")

# Step 3: count class frequencies
class_counts = defaultdict(int)
for scene, data in scene_data.items():
    for name, targets in data:
        for t in targets:
            class_counts[t] += 1

print(f"\nClass frequency:")
for c in sorted(class_counts.keys()):
    print(f"  {c:2d} {TARGET_NAMES[c]:12s}: {class_counts[c]} samples")

# Step 4: stratified split
def get_rarest(targets):
    best, best_cnt = None, float('inf')
    for t in targets:
        if class_counts[t] < best_cnt:
            best_cnt = class_counts[t]
            best = t
    return best if best is not None else 0

new_splits = {}
for scene in SCENES:
    data = scene_data[scene]
    groups = defaultdict(list)
    for name, targets in data:
        label = get_rarest(targets)
        groups[label].append(name)
    
    print(f"\n{scene} groups:")
    for lbl in sorted(groups.keys()):
        print(f"  {TARGET_NAMES[lbl]}: {len(groups[lbl])}")
    
    train_l, val_l, test_l = [], [], []
    for lbl, names in groups.items():
        np.random.shuffle(names)
        n = len(names)
        if n >= 10:
            n_test = max(1, round(n * 0.15))
            n_val = max(1, round(n * 0.15))
            n_train = n - n_val - n_test
        elif n >= 3:
            n_train = n - 2; n_val = 1; n_test = 1
        elif n == 2:
            n_train = 1; n_val = 1; n_test = 0
        else:
            n_train = 1; n_val = 0; n_test = 0
        train_l.extend(names[:n_train])
        val_l.extend(names[n_train:n_train+n_val])
        test_l.extend(names[n_train+n_val:])
    
    np.random.shuffle(train_l)
    np.random.shuffle(val_l)
    np.random.shuffle(test_l)
    new_splits[scene] = {'train': train_l, 'val': val_l, 'test': test_l}
    print(f"  -> train:{len(train_l)} val:{len(val_l)} test:{len(test_l)}")

# Step 5: backup and write
print("\nBacking up old splits...")
for scene in SCENES:
    for split in ['train', 'val', 'test']:
        txt = os.path.join(DATA_DIR, scene, split, f'{split}.txt')
        if os.path.exists(txt):
            bak = txt + '.bak'
            if not os.path.exists(bak):
                os.rename(txt, bak)

print("Writing new splits...")
for scene in SCENES:
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(DATA_DIR, scene, split)
        with open(os.path.join(split_dir, f'{split}.txt'), 'w') as f:
            for name in new_splits[scene][split]:
                f.write(name + '\n')

# Verify
print("\n=== New split verification ===")
for split in ['train', 'val', 'test']:
    split_classes = defaultdict(int)
    total = 0
    for scene in SCENES:
        name_set = set(new_splits[scene][split])
        for name, targets in scene_data[scene]:
            if name in name_set:
                for t in targets:
                    split_classes[t] += 1
                total += 1
    print(f"\n{split} ({total} samples):")
    for c in sorted(split_classes.keys()):
        pct = 100 * split_classes[c] / max(total, 1)
        print(f"  {TARGET_NAMES[c]:12s}: {split_classes[c]:>5d} ({pct:.1f}%)")

print("\nDone!")
