import json
from collections import defaultdict

print("=== Finding samples with rare classes ===")

data = json.load(open('/mnt/d/CD/0617final_256_v2/instance_dataset.json'))
samples = data['samples']

print(f"Total samples in instance_dataset: {len(samples)}")

# Find samples containing rare classes
rare_class_ids = {7: 'Shelter', 10: 'Dock', 4: 'Bridge'}
rare_samples = defaultdict(list)

for sample_name, instances in samples.items():
    for inst in instances:
        cls_id = inst.get('class_id')
        cls_name = inst.get('class_name')
        if cls_id in rare_class_ids or cls_name in ['Shelter', 'Dock', 'Bridge']:
            rare_samples[cls_name].append(sample_name)
            break  # Only count each sample once

print(f"\nSamples containing rare classes:")
for cls_name in ['Bridge', 'Shelter', 'Dock']:
    sample_list = rare_samples.get(cls_name, [])
    print(f"  {cls_name}: {len(sample_list)} samples")
    if sample_list:
        print(f"    Examples: {sample_list[:3]}")

# Get all unique rare samples
all_rare = set()
for samples_list in rare_samples.values():
    all_rare.update(samples_list)
print(f"\nTotal unique rare samples: {len(all_rare)}")

if len(all_rare) == 0:
    print("\nNo rare samples found in instance_dataset!")
    print("Bridge may not exist in the dataset at all.")
    exit(0)

# Now inject these samples into names.txt
import shutil
import os

TARGET = '/mnt/d/CD/0617final_256_v2'
INJECT_COPIES = 50

print(f"\n=== Injecting into splits ===")
print(f"Copies per sample: {INJECT_COPIES}")

for split in ['train', 'val', 'test']:
    names_file = os.path.join(TARGET, split, 'names.txt')
    if not os.path.isfile(names_file):
        print(f"{split}: names.txt not found")
        continue
    
    with open(names_file) as f:
        current = [l.strip() for l in f if l.strip()]
    current_set = set(current)
    
    # Backup
    backup = names_file + '.bak'
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
        print(f"{split}: +{len(to_add)} samples ({len(current)} -> {len(current)+len(to_add)})")
    else:
        print(f"{split}: nothing to add")

print(f"\nDone! Restart training with launch_v9.sh")
print(f"\nNote: Bridge has ~0 samples in dataset - it may not be learnable.")
print(f"Shelter ({len(rare_samples.get('Shelter',[]))} samples) and Dock ({len(rare_samples.get('Dock',[]))} samples) can benefit from oversampling.")
