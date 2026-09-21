"""
Rare class injector using raw label values.
"""
import os
import shutil
import numpy as np
from collections import defaultdict
from osgeo import gdal
gdal.UseExceptions()

SOURCE = '/mnt/d/CD/0617final_256'
TARGET = '/mnt/d/CD/0617final_256_v2'
INJECT_COPIES = 50

# Raw label values that map to rare classes
RAWE_LABELS = {
    'Bridge': range(18, 23),   # 18-22 -> class 4
    'Shelter': range(33, 38),  # 33-37 -> class 7
    'Dock': range(48, 53),     # 48-52 -> class 10
}

def read_label(path):
    ds = gdal.Open(path)
    arr = ds.ReadAsArray()
    ds = None
    return arr

print("=== Rare Class Injector (raw labels) ===")
print(f"Source: {SOURCE}")
print(f"Looking for:")
for cls, rng in RAWE_LABELS.items():
    print(f"  {cls}: raw labels {list(rng)}")
print()

# Scan labels
print("Scanning labels...")
rare_samples = defaultdict(list)
all_rare = set()
scanned = 0

for split in ['train', 'val', 'test']:
    lbl_dir = os.path.join(SOURCE, split, 'label')
    if not os.path.isdir(lbl_dir):
        continue
    
    files = [f for f in os.listdir(lbl_dir) if f.endswith('.tif')]
    print(f"  {split}: {len(files)} files")
    
    for fname in files:
        base = fname.replace('_target.tif', '').replace('.tif', '')
        path = os.path.join(lbl_dir, fname)
        
        try:
            label = read_label(path)
            for cls_name, rng in RAWE_LABELS.items():
                for raw_val in rng:
                    if (label == raw_val).any():
                        rare_samples[cls_name].append(base)
                        all_rare.add(base)
                        break
            scanned += 1
        except Exception as e:
            continue
        
        if scanned % 5000 == 0:
            print(f"    Scanned {scanned}...")

print(f"\nScan complete! ({scanned} files)")
print(f"\nRare samples found:")
for cls_name in RAWE_LABELS:
    print(f"  {cls_name}: {len(rare_samples[cls_name])}")
print(f"  Total unique: {len(all_rare)}")

if len(all_rare) == 0:
    print("\nNo rare samples found!")
    exit(0)

# Show examples
print(f"\nExamples:")
for cls_name, samples in rare_samples.items():
    if samples:
        print(f"  {cls_name}: {samples[:3]}")

# Inject into target splits
print(f"\nInjecting {INJECT_COPIES} copies each...")

for split in ['train', 'val', 'test']:
    names_file = os.path.join(TARGET, split, 'names.txt')
    if not os.path.isfile(names_file):
        continue
    
    with open(names_file) as f:
        current = [l.strip() for l in f if l.strip()]
    current_set = set(current)
    
    # Backup
    backup = names_file + '.bak'
    if not os.path.isfile(backup):
        shutil.copy2(names_file, backup)
    
    to_add = []
    for name in sorted(all_rare):
        copies = INJECT_COPIES if name not in current_set else INJECT_COPIES // 2
        to_add.extend([name] * copies)
    
    if to_add:
        with open(names_file, 'a') as f:
            for name in to_add:
                f.write(name + '\n')
        print(f"  {split}: +{len(to_add)} ({len(current)} -> {len(current)+len(to_add)})")

print(f"\nDone! Restart training.")
