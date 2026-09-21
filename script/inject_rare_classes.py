"""
Inject rare class samples into train/val/test splits.
Finds samples containing Bridge/Dock/Shelter and adds them to all splits.
"""
import os
import json
import numpy as np
from collections import Counter, defaultdict

try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    print("GDAL not available")
    exit(1)

DATA_DIR = '/mnt/d/CD/0617final_256_v2'
SOURCE_DIR = '/mnt/d/CD/0617final_256'  # Where labels actually live
RARE_CLASS_IDS = [4, 7, 10]  # Bridge, Shelter, Dock
RARE_CLASS_NAMES = ['Bridge', 'Shelter', 'Dock']

def read_label(path):
    ds = gdal.Open(path)
    arr = ds.ReadAsArray()
    ds = None
    return arr

print("=== Rare Class Injector ===")
print(f"Target classes: {RARE_CLASS_NAMES} (IDs: {RARE_CLASS_IDS})")
print(f"Source: {SOURCE_DIR}")
print(f"Target: {DATA_DIR}")
print()

# Step 1: Scan ALL samples across all source splits to find rare class samples
print("Step 1: Scanning source data for rare class samples...")

# Build label file index from source
label_map = {}  # base_name -> label_path
for split in ['train', 'val', 'test']:
    lbl_dir = os.path.join(SOURCE_DIR, split, 'label')
    if not os.path.isdir(lbl_dir):
        continue
    for fname in os.listdir(lbl_dir):
        if not (fname.endswith('.tif') or fname.endswith('.png')):
            continue
        # Extract base name
        base = fname
        for suffix in ['_target.tif', '_label.tif', '.tif', '.png']:
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break
        if base not in label_map:
            label_map[base] = os.path.join(lbl_dir, fname)

print(f"  Indexed {len(label_map)} label files from source")

# Scan for rare classes
rare_samples = defaultdict(list)  # class_name -> [sample_names]
sample_classes = defaultdict(set)  # sample_name -> {class_names}

for i, (base_name, label_path) in enumerate(label_map.items()):
    try:
        label = read_label(label_path)
        for cls_id, cls_name in zip(RARE_CLASS_IDS, RARE_CLASS_NAMES):
            if (label == cls_id).any():
                rare_samples[cls_name].append(base_name)
                sample_classes[base_name].add(cls_name)
    except Exception:
        continue
    if (i + 1) % 5000 == 0:
        print(f"  Scanned {i+1}/{len(label_map)}...")

print(f"\n  Scan complete!")
for cls_name in RARE_CLASS_NAMES:
    print(f"    {cls_name}: {len(rare_samples[cls_name])} samples")

all_rare = set()
for names in rare_samples.values():
    all_rare.update(names)
print(f"    Total unique rare samples: {len(all_rare)}")

# Step 2: Read current names.txt files
print("\nStep 2: Reading current splits...")
current_names = {}
for split in ['train', 'val', 'test']:
    names_file = os.path.join(DATA_DIR, split, 'names.txt')
    if os.path.isfile(names_file):
        with open(names_file) as f:
            current_names[split] = [l.strip() for l in f if l.strip()]
        print(f"  {split}: {len(current_names[split])} samples")
    else:
        current_names[split] = []
        print(f"  {split}: not found")

# Step 3: Determine which rare samples are missing from each split
print("\nStep 3: Checking coverage...")
for split in ['train', 'val', 'test']:
    current_set = set(current_names[split])
    missing = all_rare - current_set
    present = all_rare & current_set
    print(f"  {split}: {len(present)} rare samples present, {len(missing)} missing")

# Step 4: Inject missing rare samples into each split
print("\nStep 4: Injecting rare samples into splits...")
INJECT_COUNT = 50  # How many copies of each rare sample to add

for split in ['train', 'val', 'test']:
    names_file = os.path.join(DATA_DIR, split, 'names.txt')
    backup_file = names_file + '.bak_rare'
    
    # Backup original
    if os.path.isfile(names_file) and not os.path.isfile(backup_file):
        import shutil
        shutil.copy2(names_file, backup_file)
        print(f"  Backed up {split}/names.txt -> names.txt.bak_rare")
    
    current_set = set(current_names[split])
    to_add = []
    
    for sample_name in sorted(all_rare):
        # Add multiple copies for oversampling
        copies = INJECT_COUNT if sample_name not in current_set else INJECT_COUNT // 2
        for _ in range(copies):
            to_add.append(sample_name)
    
    if to_add:
        # Append to names.txt
        with open(names_file, 'a') as f:
            for name in to_add:
                f.write(name + '\n')
        
        new_total = len(current_names[split]) + len(to_add)
        print(f"  {split}: added {len(to_add)} rare samples ({len(current_names[split])} -> {new_total})")
    else:
        print(f"  {split}: no samples to add")

# Step 5: Generate summary
print("\n" + "="*60)
print("Summary:")
print("="*60)
print(f"\nRare class samples found:")
for cls_name in RARE_CLASS_NAMES:
    samples = rare_samples[cls_name]
    print(f"  {cls_name}: {len(samples)} unique samples")
    if samples:
        print(f"    Examples: {samples[:3]}")

print(f"\nInjection settings:")
print(f"  New samples: {INJECT_COUNT} copies each")
print(f"  Existing samples: {INJECT_COUNT//2} copies each")

print(f"\nNew split sizes:")
for split in ['train', 'val', 'test']:
    names_file = os.path.join(DATA_DIR, split, 'names.txt')
    if os.path.isfile(names_file):
        with open(names_file) as f:
            count = sum(1 for l in f if l.strip())
        print(f"  {split}: {count}")

print(f"\nDone! Restart training to use updated splits.")
print(f"Backup files saved as names.txt.bak_rare")
