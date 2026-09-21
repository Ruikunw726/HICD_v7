# -*- coding: utf-8 -*-
"""
Instance-level statistics for classes with independent individual objects.
Scans dataset labels, finds connected components, counts instances.
"""
import os
import sys
import numpy as np
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

import tifffile
from scipy import ndimage

# Config
DATA_DIR = "D:/CD/0617final_256"
NAMES_PATH = "D:/CD/0617final_256_v2/train/names.txt"

# Classes with independent individuals (raw label IDs from decode_map)
# Based on 0617final.yaml label_decode_map:
# raw 38-42 -> Tower(8)
# raw 43-49 -> Pier(9)
# raw 48-52 -> Dock(10)
# raw 53-56 -> Tank(11)
# raw 57-61 -> Aircraft(12)
# raw 62-66 -> Vessel(13)
# raw 67-68 -> Crater(14)
# raw 33-37 -> Shelter(7)
# raw 18-22 -> Bridge(4)

# Target class names (0-indexed after decode)
TARGET_NAMES = [
    'Farmland', 'Runway', 'Taxiway', 'Apron', 'Bridge', 'Highway',
    'Building', 'Shelter', 'Tower', 'Pier', 'Dock', 'Tank',
    'Aircraft', 'Vessel', 'Crater'
]

# Raw label to target class mapping (from config)
RAW_TO_TARGET = {}
# Farmland: 1-2
for i in range(1, 3): RAW_TO_TARGET[i] = 0
# Runway: 3-7
for i in range(3, 8): RAW_TO_TARGET[i] = 1
# Taxiway: 8-12
for i in range(8, 13): RAW_TO_TARGET[i] = 2
# Apron: 13-17
for i in range(13, 18): RAW_TO_TARGET[i] = 3
# Bridge: 18-22
for i in range(18, 23): RAW_TO_TARGET[i] = 4
# Highway: 23-27
for i in range(23, 28): RAW_TO_TARGET[i] = 5
# Building: 28-32
for i in range(28, 33): RAW_TO_TARGET[i] = 6
# Shelter: 33-37
for i in range(33, 38): RAW_TO_TARGET[i] = 7
# Tower: 38-42
for i in range(38, 43): RAW_TO_TARGET[i] = 8
# Pier: 43-47
for i in range(43, 48): RAW_TO_TARGET[i] = 9
# Dock: 48-52
for i in range(48, 53): RAW_TO_TARGET[i] = 10
# Tank: 53-56
for i in range(53, 57): RAW_TO_TARGET[i] = 11
# Aircraft: 57-61
for i in range(57, 62): RAW_TO_TARGET[i] = 12
# Vessel: 62-66
for i in range(62, 67): RAW_TO_TARGET[i] = 13
# Crater: 67-68
for i in range(67, 69): RAW_TO_TARGET[i] = 14

# Instance-level classes (have independent individuals)
INSTANCE_CLASSES = [4, 7, 8, 9, 10, 11, 12, 13, 14]  # Bridge, Shelter, Tower, Pier, Dock, Tank, Aircraft, Vessel, Crater
INSTANCE_CLASS_NAMES = [TARGET_NAMES[i] for i in INSTANCE_CLASSES]


def decode_label(raw_label):
    """Decode raw label to target class."""
    decoded = np.full_like(raw_label, 255)
    for raw, target in RAW_TO_TARGET.items():
        decoded[raw_label == raw] = target
    return decoded


def count_instances(label_mask, min_pixels=10):
    """Count connected components and return instance stats."""
    labeled, n = ndimage.label(label_mask)
    if n == 0:
        return 0, []

    instances = []
    for i in range(1, n + 1):
        inst_mask = (labeled == i)
        n_pixels = inst_mask.sum()
        if n_pixels < min_pixels:
            continue
        # Get bounding box
        rows = np.where(inst_mask.any(axis=1))[0]
        cols = np.where(inst_mask.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            continue
        h = rows[-1] - rows[0] + 1
        w = cols[-1] - cols[0] + 1
        instances.append({
            'pixels': n_pixels,
            'height': h,
            'width': w,
            'aspect_ratio': w / max(h, 1),
        })

    return len(instances), instances


def main():
    # Read names
    with open(NAMES_PATH, 'r') as f:
        names = [line.strip() for line in f if line.strip()]

    # Filter out synthetic for "real" stats, but include all for total
    real_names = [n for n in names if not n.startswith('synth_')]

    print(f"Total samples: {len(names)}")
    print(f"Real samples: {len(real_names)}")
    print(f"Instance classes: {INSTANCE_CLASS_NAMES}")
    print()

    # Stats storage
    class_instances = defaultdict(list)  # class_id -> list of instance dicts
    class_pixel_counts = defaultdict(int)
    total_pixels = 0

    # Scan labels
    print("Scanning labels...")
    for idx, name in enumerate(real_names):
        lbl_path = os.path.join(DATA_DIR, "train", "label", f"{name}_target.tif")
        if not os.path.exists(lbl_path):
            continue

        try:
            raw_label = tifffile.imread(lbl_path)
        except:
            continue

        decoded = decode_label(raw_label)
        total_pixels += decoded.size

        for cls_id in INSTANCE_CLASSES:
            mask = (decoded == cls_id)
            n_px = mask.sum()
            if n_px == 0:
                continue

            class_pixel_counts[cls_id] += n_px

            # Count instances
            n_inst, instances = count_instances(mask, min_pixels=10)
            for inst in instances:
                inst['class'] = cls_id
                inst['sample'] = name
                class_instances[cls_id].append(inst)

        if (idx + 1) % 1000 == 0:
            print(f"  {idx + 1}/{len(real_names)}")

    # Print statistics
    print("\n" + "=" * 80)
    print("Instance-Level Statistics (Real Samples Only)")
    print("=" * 80)

    print(f"\n{'Class':<12} {'Pixels':>12} {'Pixel%':>8} {'Instances':>10} {'Avg Size':>10} {'Avg H×W':>15}")
    print("-" * 80)

    for cls_id in INSTANCE_CLASSES:
        name = TARGET_NAMES[cls_id]
        px = class_pixel_counts[cls_id]
        px_pct = 100.0 * px / max(total_pixels, 1)
        instances = class_instances[cls_id]
        n_inst = len(instances)

        if n_inst > 0:
            avg_size = np.mean([i['pixels'] for i in instances])
            avg_h = np.mean([i['height'] for i in instances])
            avg_w = np.mean([i['width'] for i in instances])
            print(f"{name:<12} {px:>12,} {px_pct:>7.4f}% {n_inst:>10,} {avg_size:>10.0f} {avg_h:>6.0f}×{avg_w:<6.0f}")
        else:
            print(f"{name:<12} {px:>12,} {px_pct:>7.4f}% {n_inst:>10,} {'N/A':>10} {'N/A':>15}")

    # Detailed per-class analysis
    print("\n" + "=" * 80)
    print("Detailed Instance Size Distribution")
    print("=" * 80)

    for cls_id in INSTANCE_CLASSES:
        name = TARGET_NAMES[cls_id]
        instances = class_instances[cls_id]
        if len(instances) == 0:
            continue

        sizes = [i['pixels'] for i in instances]
        heights = [i['height'] for i in instances]
        widths = [i['width'] for i in instances]

        print(f"\n{name} ({len(instances)} instances):")
        print(f"  Pixels:  min={min(sizes)}, max={max(sizes)}, "
              f"mean={np.mean(sizes):.0f}, median={np.median(sizes):.0f}")
        print(f"  Height:  min={min(heights)}, max={max(heights)}, "
              f"mean={np.mean(heights):.1f}, median={np.median(heights):.0f}")
        print(f"  Width:   min={min(widths)}, max={max(widths)}, "
              f"mean={np.mean(widths):.1f}, median={np.median(widths):.0f}")

        # Size distribution
        size_bins = [0, 50, 100, 200, 500, 1000, 5000, 100000]
        size_labels = ['<50', '50-100', '100-200', '200-500', '500-1K', '1K-5K', '>5K']
        hist, _ = np.histogram(sizes, bins=size_bins)
        print(f"  Size distribution: ", end="")
        for label, count in zip(size_labels, hist):
            pct = 100.0 * count / len(sizes)
            print(f"{label}:{pct:.0f}% ", end="")
        print()

    # Training feasibility assessment
    print("\n" + "=" * 80)
    print("Training Feasibility Assessment")
    print("=" * 80)

    for cls_id in INSTANCE_CLASSES:
        name = TARGET_NAMES[cls_id]
        n_inst = len(class_instances[cls_id])

        if n_inst >= 1000:
            feasibility = "✅ Excellent"
        elif n_inst >= 500:
            feasibility = "✅ Good"
        elif n_inst >= 100:
            feasibility = "⚠️ Moderate"
        elif n_inst >= 50:
            feasibility = "⚠️ Limited"
        elif n_inst >= 10:
            feasibility = "❌ Very limited"
        else:
            feasibility = "❌ Insufficient"

        print(f"  {name:<12}: {n_inst:>6,} instances → {feasibility}")

    # Summary for instance-level training
    print("\n" + "=" * 80)
    print("Recommendation for Instance-Level Training")
    print("=" * 80)

    trainable = []
    borderline = []
    insufficient = []

    for cls_id in INSTANCE_CLASSES:
        name = TARGET_NAMES[cls_id]
        n_inst = len(class_instances[cls_id])
        if n_inst >= 100:
            trainable.append((name, n_inst))
        elif n_inst >= 20:
            borderline.append((name, n_inst))
        else:
            insufficient.append((name, n_inst))

    if trainable:
        print(f"\n✅ Trainable (≥100 instances):")
        for name, n in sorted(trainable, key=lambda x: -x[1]):
            print(f"   {name}: {n:,}")

    if borderline:
        print(f"\n⚠️ Borderline (20-99 instances, need augmentation):")
        for name, n in sorted(borderline, key=lambda x: -x[1]):
            print(f"   {name}: {n:,}")

    if insufficient:
        print(f"\n❌ Insufficient (<20 instances):")
        for name, n in sorted(insufficient, key=lambda x: -x[1]):
            print(f"   {name}: {n:,}")


if __name__ == "__main__":
    main()
