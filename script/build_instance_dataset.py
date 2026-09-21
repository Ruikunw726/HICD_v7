# -*- coding: utf-8 -*-
"""
Build instance dataset for T3 (mobile) and T4 (fixed facility) tasks.
Only extracts instances for classes we need.
"""
import os
import sys
import json
import numpy as np
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

import tifffile
from scipy import ndimage

DATA_DIR = "D:/CD/0617final_256"
NAMES_PATH = "D:/CD/0617final_256_v2/train/names.txt"
OUTPUT_PATH = "D:/CD/0617final_256_v2/instance_dataset.json"

# Raw label ranges for T3/T4 classes only
# Shelter: 33-37 -> target 7
# Pier: 43-47 -> target 9
# Dock: 48-52 -> target 10
# Tank: 53-56 -> target 11
# Aircraft: 57-61 -> target 12
# Vessel: 62-66 -> target 13
# Crater: 67-68 -> target 14

RAW_TO_TARGET = {}
for i in range(33, 38): RAW_TO_TARGET[i] = 7    # Shelter
for i in range(43, 48): RAW_TO_TARGET[i] = 9    # Pier
for i in range(48, 53): RAW_TO_TARGET[i] = 10   # Dock
for i in range(53, 57): RAW_TO_TARGET[i] = 11   # Tank
for i in range(57, 62): RAW_TO_TARGET[i] = 12   # Aircraft
for i in range(62, 67): RAW_TO_TARGET[i] = 13   # Vessel
for i in range(67, 69): RAW_TO_TARGET[i] = 14   # Crater

# State decode (only for relevant raw ranges)
STATE_DECODE = {
    # Shelter
    33: 0, 34: 1, 35: 2, 36: 3, 37: 4,
    # Pier
    43: 0, 44: 1, 45: 2, 46: 3, 47: 4,
    # Dock
    48: 0, 49: 1, 50: 2, 51: 3, 52: 4,
    # Tank
    53: 0, 54: 1, 55: 2, 56: 3,
    # Aircraft
    57: 0, 58: 1, 59: 2, 60: 3, 61: 5,  # 61=replaced
    # Vessel
    62: 0, 63: 1, 64: 2, 65: 3, 66: 5,  # 66=replaced
    # Crater
    67: 3, 68: 3,  # added
}

TARGET_NAMES = {
    7: 'Shelter', 9: 'Pier', 10: 'Dock', 11: 'Tank',
    12: 'Aircraft', 13: 'Vessel', 14: 'Crater'
}

STATE_NAMES = ['no-change', 'damaged', 'reduced', 'added', 'extended', 'replaced', 'none']

INSTANCE_CLASSES = {
    'T3': [12, 13],           # Aircraft, Vessel
    'T4': [7, 9, 10, 11, 14], # Shelter, Pier, Dock, Tank, Crater
}
ALL_INSTANCE_CLASSES = [7, 9, 10, 11, 12, 13, 14]


def main():
    with open(NAMES_PATH, 'r') as f:
        names = [line.strip() for line in f if line.strip()]
    real_names = [n for n in names if not n.startswith('synth_')]
    print(f"Processing {len(real_names)} real samples...")

    dataset = {
        'description': 'Instance dataset for T3/T4 tasks',
        'tasks': {
            'T3': {'classes': ['Aircraft', 'Vessel'], 'class_ids': [12, 13]},
            'T4': {'classes': ['Shelter', 'Pier', 'Dock', 'Tank', 'Crater'], 'class_ids': [7, 9, 10, 11, 14]},
        },
        'samples': {},
        'statistics': {},
    }

    class_instances = defaultdict(list)
    class_states = defaultdict(lambda: defaultdict(int))

    for idx, name in enumerate(real_names):
        lbl_path = os.path.join(DATA_DIR, "train", "label", f"{name}_target.tif")
        if not os.path.exists(lbl_path):
            continue

        try:
            raw_label = tifffile.imread(lbl_path)
        except:
            continue

        sample_instances = []

        for raw, tgt in RAW_TO_TARGET.items():
            mask = (raw_label == raw)
            if mask.sum() == 0:
                continue

            labeled, n = ndimage.label(mask)
            for i in range(1, n + 1):
                inst_mask = (labeled == i)
                n_px = inst_mask.sum()
                if n_px < 10:
                    continue

                rows = np.where(inst_mask.any(axis=1))[0]
                cols = np.where(inst_mask.any(axis=0))[0]
                if len(rows) == 0 or len(cols) == 0:
                    continue

                y1, y2 = int(rows[0]), int(rows[-1])
                x1, x2 = int(cols[0]), int(cols[-1])

                state = STATE_DECODE.get(raw, 0)

                inst = {
                    'class_id': tgt,
                    'class_name': TARGET_NAMES[tgt],
                    'bbox': [x1, y1, x2, y2],
                    'pixels': int(n_px),
                    'height': int(y2 - y1 + 1),
                    'width': int(x2 - x1 + 1),
                    'state': state,
                    'state_name': STATE_NAMES[state],
                    'sample': name,
                }
                sample_instances.append(inst)
                class_instances[tgt].append(inst)
                class_states[tgt][state] += 1

        if sample_instances:
            dataset['samples'][name] = sample_instances

        if (idx + 1) % 5000 == 0:
            total = sum(len(v) for v in class_instances.values())
            print(f"  {idx + 1}/{len(real_names)}, instances: {total}")

    # Statistics
    for cls_id in ALL_INSTANCE_CLASSES:
        instances = class_instances[cls_id]
        name = TARGET_NAMES[cls_id]
        task = 'T3' if cls_id in INSTANCE_CLASSES['T3'] else 'T4'

        if instances:
            sizes = [i['pixels'] for i in instances]
            dataset['statistics'][name] = {
                'task': task,
                'class_id': cls_id,
                'total_instances': len(instances),
                'state_distribution': {STATE_NAMES[s]: c for s, c in sorted(class_states[cls_id].items())},
                'size_stats': {
                    'mean_pixels': float(np.mean(sizes)),
                    'median_pixels': float(np.median(sizes)),
                    'min_pixels': int(min(sizes)),
                    'max_pixels': int(max(sizes)),
                },
            }
        else:
            dataset['statistics'][name] = {
                'task': task, 'class_id': cls_id,
                'total_instances': 0, 'state_distribution': {}, 'size_stats': {},
            }

    # Print summary
    total_instances = sum(len(v) for v in class_instances.values())
    print(f"\n{'='*60}")
    print(f"Instance Dataset Summary")
    print(f"{'='*60}")
    print(f"Samples with instances: {len(dataset['samples'])}")
    print(f"Total instances: {total_instances}")

    for task in ['T3', 'T4']:
        print(f"\n{task}:")
        for cls_id in INSTANCE_CLASSES[task]:
            name = TARGET_NAMES[cls_id]
            s = dataset['statistics'][name]
            print(f"  {name:<10}: {s['total_instances']:>5,} instances  {s['state_distribution']}")

    # Save
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
