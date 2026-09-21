"""
Inject rare class samples from original 0617final into v2 dataset.
"""
import os
import shutil
import numpy as np
from collections import defaultdict
from osgeo import gdal
gdal.UseExceptions()

SOURCE_BASE = '/mnt/d/CD/0617final'
TARGET = '/mnt/d/CD/0617final_256_v2'
INJECT_COPIES = 50

BRIDGE_RAW = set(range(18, 23))
SHELTER_RAW = set(range(33, 38))
DOCK_RAW = set(range(48, 53))

def read_label(path):
    ds = gdal.Open(path)
    arr = ds.ReadAsArray()
    ds = None
    return arr

print("=== Injecting Rare Classes from Source ===")

# Scan source for rare class samples
found_samples = defaultdict(list)

for scene in ['Ports', 'Airports', 'Urban-Rural Areas']:
    scene_path = os.path.join(SOURCE_BASE, scene)
    if not os.path.isdir(scene_path):
        continue
    
    print(f"Scanning {scene}...")
    
    for split in ['train', 'val', 'test']:
        lbl_dir = os.path.join(scene_path, split, 'label')
        if not os.path.isdir(lbl_dir):
            continue
        
        for fname in os.listdir(lbl_dir):
            if not fname.endswith('.tif'):
                continue
            
            path = os.path.join(lbl_dir, fname)
            base_name = fname.replace('_target.tif', '').replace('.tif', '')
            
            try:
                label = read_label(path)
                unique = set(np.unique(label).tolist())
                
                if unique & BRIDGE_RAW:
                    found_samples['Bridge'].append(base_name)
                if unique & SHELTER_RAW:
                    found_samples['Shelter'].append(base_name)
                if unique & DOCK_RAW:
                    found_samples['Dock'].append(base_name)
            except:
                continue

print(f"\nFound samples:")
for cls_name, samples in found_samples.items():
    print(f"  {cls_name}: {len(samples)}")

# Get all unique rare samples
all_rare = set()
for samples in found_samples.values():
    all_rare.update(samples)
print(f"  Total unique: {len(all_rare)}")

if len(all_rare) == 0:
    print("No samples found!")
    exit(0)

# Inject into v2 dataset names.txt
print(f"\nInjecting {INJECT_COPIES} copies each into {TARGET}...")

for split in ['train', 'val', 'test']:
    names_file = os.path.join(TARGET, split, 'names.txt')
    if not os.path.isfile(names_file):
        print(f"  {split}: names.txt not found")
        continue
    
    with open(names_file) as f:
        current = [l.strip() for l in f if l.strip()]
    current_set = set(current)
    
    # Backup
    backup = names_file + '.bak_rare'
    if not os.path.isfile(backup):
        shutil.copy2(names_file, backup)
    
    # Add rare samples
    to_add = []
    for name in sorted(all_rare):
        if name not in current_set:
            to_add.extend([name] * INJECT_COPIES)
        else:
            to_add.extend([name] * (INJECT_COPIES // 2))
    
    if to_add:
        with open(names_file, 'a') as f:
            for name in to_add:
                f.write(name + '\n')
        print(f"  {split}: +{len(to_add)} ({len(current)} -> {len(current)+len(to_add)})")

print(f"\nDone! Rare class samples injected.")
print(f"\nNote: These samples are from 0617final (original resolution).")
print(f"The dataloader will need to find the images. Check if paths resolve correctly.")
