"""
Fast rare class scanner - generates cache files for oversampling.
Uses GDAL directly for speed.
"""
import os
import json
import numpy as np
from collections import Counter

try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    print("GDAL not available")
    exit(1)

DATA_DIR = '/mnt/d/CD/0617final_256_v2'
RARE_CLASS_IDS = [4, 7, 10]  # Bridge, Shelter, Dock
RARE_CLASS_NAMES = ['Bridge', 'Shelter', 'Dock']

def read_label_fast(path):
    ds = gdal.Open(path)
    arr = ds.ReadAsArray()
    ds = None
    return arr

print("=== Fast Rare Class Scanner ===")
print(f"Classes: {RARE_CLASS_NAMES} (IDs: {RARE_CLASS_IDS})")
print()

for split in ['train', 'val', 'test']:
    print(f"Scanning {split}...")
    
    names_file = os.path.join(DATA_DIR, split, 'names.txt')
    if not os.path.isfile(names_file):
        print(f"  names.txt not found")
        continue
    
    with open(names_file) as f:
        names = [l.strip() for l in f if l.strip()]
    
    print(f"  Total samples: {len(names)}")

    # Build label path lookup - search common locations
    label_dirs = []
    # Search in 0617final_256 (main data location)
    for split_name in ['train', 'val', 'test']:
        lbl_dir = f'/mnt/d/CD/0617final_256/{split_name}/label'
        if os.path.isdir(lbl_dir):
            label_dirs.append(lbl_dir)
    # Also search in 0617final (original data)
    for scene in ['Airports', 'Ports', 'Urban-Rural Areas']:
        for split_name in ['train', 'val', 'test']:
            lbl_dir = f'/mnt/d/CD/0617final/{scene}/{split_name}/label'
            if os.path.isdir(lbl_dir):
                label_dirs.append(lbl_dir)
    # Search in 0617final_256_v2 if labels exist there
    for split_name in ['train', 'val', 'test']:
        lbl_dir = f'/mnt/d/CD/0617final_256_v2/{split_name}/label'
        if os.path.isdir(lbl_dir):
            label_dirs.append(lbl_dir)

    print(f"  Label dirs found: {len(label_dirs)}")
    for d in label_dirs[:5]:
        print(f"    {d}")
    
    # Build filename -> path map
    label_map = {}
    for label_dir in label_dirs:
        for fname in os.listdir(label_dir):
            if fname.endswith('.tif') or fname.endswith('.png'):
                # Extract base name
                base = fname
                for suffix in ['_target.tif', '_label.tif', '.tif', '.png']:
                    if base.endswith(suffix):
                        base = base[:-len(suffix)]
                        break
                if base not in label_map:
                    label_map[base] = os.path.join(label_dir, fname)
    
    print(f"  Label files indexed: {len(label_map)}")
    
    # Scan for rare classes
    rare_indices = []
    class_counts = Counter()
    pixel_counts = Counter()
    scanned = 0
    not_found = 0
    
    for i, name in enumerate(names):
        label_path = label_map.get(name)
        if label_path is None:
            not_found += 1
            continue
        
        try:
            label = read_label_fast(label_path)
            found_rare = False
            for cls_id, cls_name in zip(RARE_CLASS_IDS, RARE_CLASS_NAMES):
                count = (label == cls_id).sum()
                if count > 0:
                    class_counts[cls_name] += 1
                    pixel_counts[cls_name] += int(count)
                    found_rare = True
            if found_rare:
                rare_indices.append(i)
            scanned += 1
        except Exception as e:
            continue
        
        if (i + 1) % 10000 == 0:
            print(f"    Progress: {i+1}/{len(names)}, found {len(rare_indices)} rare samples")
    
    # Save cache
    cache = {
        'rare_indices': rare_indices,
        'class_counts': dict(class_counts),
        'pixel_counts': dict(pixel_counts),
        'rare_class_ids': RARE_CLASS_IDS,
        'rare_names': RARE_CLASS_NAMES,
        'total_samples': len(names),
        'scanned': scanned,
        'not_found': not_found,
        'split': split,
    }
    cache_path = os.path.join(DATA_DIR, f'rare_class_samples_{split}.json')
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    
    # Print statistics
    print(f"\n  Results for {split}:")
    print(f"  Scanned: {scanned}/{len(names)} (not found: {not_found})")
    print(f"  {'Class':<12} {'Samples':>10} {'Pixels':>15}")
    print(f"  {'-'*40}")
    for cls_name in RARE_CLASS_NAMES:
        samples = class_counts.get(cls_name, 0)
        pixels = pixel_counts.get(cls_name, 0)
        print(f"  {cls_name:<12} {samples:>10} {pixels:>15,}")
    print(f"  {'-'*40}")
    print(f"  {'Total rare':<12} {len(rare_indices):>10}")
    print(f"  Cache saved: {cache_path}")
    print()

print("Done!")
