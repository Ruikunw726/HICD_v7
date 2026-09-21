"""
Pre-scan labels to find samples containing rare classes (Bridge/Dock/Shelter).
Generates cache files for fast sampling weight computation.
"""
import os
import json
import numpy as np
from collections import Counter
from tqdm import tqdm

try:
    from osgeo import gdal
    gdal.UseExceptions()
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False

def read_label(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.tif', '.tiff'] and HAS_GDAL:
        ds = gdal.Open(path)
        arr = ds.ReadAsArray()
        ds = None
        return arr
    from PIL import Image
    return np.array(Image.open(path))

DATA_DIR = '/mnt/d/CD/0617final_256_v2'
RARE_CLASS_IDS = [4, 7, 10]  # Bridge, Shelter, Dock
RARE_CLASS_NAMES = ['Bridge', 'Shelter', 'Dock']
TARGET_NAMES = [
    'Farmland', 'Runway', 'Taxiway', 'Apron', 'Bridge', 'Highway',
    'Building', 'Shelter', 'Tower', 'Pier', 'Dock', 'Tank',
    'Aircraft', 'Vessel', 'Crater', 'Background'
]

print("=== Rare Class Scanner ===")
print(f"Target classes: {RARE_CLASS_NAMES} (IDs: {RARE_CLASS_IDS})")
print(f"Data dir: {DATA_DIR}")
print()

for split in ['train', 'val', 'test']:
    print(f"\n{'='*60}")
    print(f"Scanning {split} split...")
    print(f"{'='*60}")

    names_file = os.path.join(DATA_DIR, split, 'names.txt')
    if not os.path.isfile(names_file):
        print(f"  names.txt not found, skipping")
        continue

    with open(names_file) as f:
        names = [l.strip() for l in f if l.strip()]

    print(f"  Total samples: {len(names)}")

    # Build path lookup for labels
    label_map = {}
    search_dirs = [os.path.join(DATA_DIR, split)]
    # Also search sibling directories
    parent = os.path.dirname(DATA_DIR)
    if os.path.isdir(parent):
        for d in os.listdir(parent):
            candidate = os.path.join(parent, d, split)
            if os.path.isdir(candidate) and candidate != search_dirs[0]:
                search_dirs.append(candidate)

    for search_dir in search_dirs:
        label_dir = os.path.join(search_dir, 'label')
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if fname.endswith('.tif') or fname.endswith('.png'):
                base = fname
                for suffix in ['_target.tif', '_label.tif', '.tif', '.png']:
                    if base.endswith(suffix):
                        base = base[:-len(suffix)]
                        break
                if base not in label_map:
                    label_map[base] = os.path.join(label_dir, fname)

    print(f"  Found {len(label_map)} label files")

    # Scan labels
    rare_indices = []
    class_counts = Counter()
    pixel_counts = Counter()

    for i, name in enumerate(tqdm(names, desc=f"Scanning {split}")):
        # Find label file
        label_path = label_map.get(name)
        if label_path is None:
            # Try common patterns
            for suffix in ['_target.tif', '_label.tif', '.tif']:
                for search_dir in search_dirs:
                    candidate = os.path.join(search_dir, 'label', name + suffix)
                    if os.path.isfile(candidate):
                        label_path = candidate
                        break
                if label_path:
                    break

        if label_path is None or not os.path.isfile(label_path):
            continue

        try:
            label = read_label(label_path)
            found_rare = False
            for cls_id, cls_name in zip(RARE_CLASS_IDS, RARE_CLASS_NAMES):
                count = (label == cls_id).sum()
                if count > 0:
                    class_counts[cls_name] += 1
                    pixel_counts[cls_name] += int(count)
                    found_rare = True
            if found_rare:
                rare_indices.append(i)
        except Exception as e:
            continue

    # Save cache
    cache = {
        'rare_indices': rare_indices,
        'class_counts': dict(class_counts),
        'pixel_counts': dict(pixel_counts),
        'rare_class_ids': RARE_CLASS_IDS,
        'rare_names': RARE_CLASS_NAMES,
        'total_samples': len(names),
        'split': split,
    }
    cache_path = os.path.join(DATA_DIR, f'rare_class_samples_{split}.json')
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)

    # Print statistics
    print(f"\n  Results for {split}:")
    print(f"  {'Class':<12} {'Samples':>10} {'Pixels':>15} {'Sample%':>10}")
    print(f"  {'-'*50}")
    for cls_name in RARE_CLASS_NAMES:
        samples = class_counts.get(cls_name, 0)
        pixels = pixel_counts.get(cls_name, 0)
        pct = 100.0 * samples / len(names) if len(names) > 0 else 0
        print(f"  {cls_name:<12} {samples:>10} {pixels:>15,} {pct:>9.2f}%")
    print(f"  {'-'*50}")
    print(f"  {'Total rare':<12} {len(rare_indices):>10} {'':>15} {100.0*len(rare_indices)/len(names):>9.2f}%")
    print(f"\n  Cache saved to: {cache_path}")

print("\n" + "="*60)
print("Scan complete!")
print("="*60)
print("\nRecommendations:")
print("  - If Bridge/Dock/Shelter have 0 samples in train, they cannot be learned")
print("  - Consider moving some samples from test to train, or collecting more data")
print("  - Oversampling only helps if there are SOME training samples")
