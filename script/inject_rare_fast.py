"""
Fast rare class injector - scans 0617final_256 labels directly.
"""
import os
import shutil
import numpy as np
from collections import defaultdict

from osgeo import gdal
gdal.UseExceptions()

SOURCE = '/mnt/d/CD/0617final_256'
TARGET = '/mnt/d/CD/0617final_256_v2'
RARE_IDS = {4: 'Bridge', 7: 'Shelter', 10: 'Dock'}
INJECT_COPIES = 50

def read_label(path):
    ds = gdal.Open(path)
    arr = ds.ReadAsArray()
    ds = None
    return arr

print("=== Fast Rare Class Injector ===")
print(f"Source: {SOURCE}")
print(f"Target: {TARGET}")
print()

# Scan all labels in source
print("Scanning labels...")
rare_samples = defaultdict(list)  # class -> [names]
all_rare = set()
scanned = 0

for split in ['train', 'val', 'test']:
    lbl_dir = os.path.join(SOURCE, split, 'label')
    if not os.path.isdir(lbl_dir):
        print(f"  {split}/label not found")
        continue
    
    files = [f for f in os.listdir(lbl_dir) if f.endswith('.tif')]
    print(f"  {split}: {len(files)} label files")
    
    for fname in files:
        # Extract base name (remove _target.tif suffix)
        base = fname.replace('_target.tif', '').replace('.tif', '')
        path = os.path.join(lbl_dir, fname)
        
        try:
            label = read_label(path)
            for cls_id, cls_name in RARE_IDS.items():
                if (label == cls_id).any():
                    rare_samples[cls_name].append(base)
                    all_rare.add(base)
            scanned += 1
        except:
            continue
        
        if scanned % 2000 == 0:
            print(f"    Scanned {scanned}...")

print(f"\nScan complete! Scanned {scanned} files")
print(f"\nRare class samples found:")
for cls_name in sorted(set(RARE_IDS.values())):
    count = len(rare_samples.get(cls_name, []))
    print(f"  {cls_name}: {count}")
print(f"  Total unique: {len(all_rare)}")

if len(all_rare) == 0:
    print("\nNo rare class samples found! Check source data.")
    exit(0)

# Inject into target splits
print(f"\nInjecting {INJECT_COPIES} copies of each rare sample...")

for split in ['train', 'val', 'test']:
    names_file = os.path.join(TARGET, split, 'names.txt')
    if not os.path.isfile(names_file):
        print(f"  {split}: names.txt not found")
        continue
    
    # Read current
    with open(names_file) as f:
        current = [l.strip() for l in f if l.strip()]
    current_set = set(current)
    
    # Backup
    backup = names_file + '.bak'
    if not os.path.isfile(backup):
        shutil.copy2(names_file, backup)
    
    # Determine what to add
    to_add = []
    for name in sorted(all_rare):
        copies = INJECT_COPIES if name not in current_set else INJECT_COPIES // 2
        to_add.extend([name] * copies)
    
    if to_add:
        with open(names_file, 'a') as f:
            for name in to_add:
                f.write(name + '\n')
        print(f"  {split}: +{len(to_add)} samples ({len(current)} -> {len(current)+len(to_add)})")
    else:
        print(f"  {split}: nothing to add")

# Final counts
print(f"\nFinal split sizes:")
for split in ['train', 'val', 'test']:
    names_file = os.path.join(TARGET, split, 'names.txt')
    if os.path.isfile(names_file):
        with open(names_file) as f:
            count = sum(1 for l in f if l.strip())
        print(f"  {split}: {count}")

print(f"\nDone! Restart training with launch_v9.sh")
