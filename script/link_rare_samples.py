"""
Create symlinks from 0617final source to 0617final_256_v2 for rare class samples.
"""
import os
import json
from collections import defaultdict

SOURCE_BASE = '/mnt/d/CD/0617final'
TARGET_BASE = '/mnt/d/CD/0617final_256_v2'

# Rare class samples found in source
RARE_SAMPLES = {
    'Bridge': [
        'Kherson_000000_port_000000000128',
        'Kherson_000000_port_000000000129',
        'Kherson_000000_port_000000000136',
        'Kherson_000000_port_000000000137',
        'Mykolaiv_000000_port_000000000387',
        # ... more from scan
    ],
    'Shelter': [
        'Kyiv_000001_airport_000000000095',
        'Kyiv_000002_airport_000000000054',
        'Kyiv_000002_airport_000000000056',
        'Kyiv_000002_airport_000000000069',
        'Kyiv_000002_airport_000000000084',
        # ... more
    ],
    'Dock': [
        'Kherson_000000_port_000000000128',
        'Kherson_000000_port_000000000129',
        'Kherson_000000_port_000000000130',
        'Kherson_000000_port_000000000131',
        'Kherson_000000_port_000000000132',
        # ... more
    ],
}

print("=== Linking Source Images to v2 Dataset ===")

# First, find all rare samples from source
print("Scanning source for all rare class samples...")

import numpy as np
from osgeo import gdal
gdal.UseExceptions()

BRIDGE_RAW = set(range(18, 23))
SHELTER_RAW = set(range(33, 38))
DOCK_RAW = set(range(48, 53))

def read_label(path):
    ds = gdal.Open(path)
    arr = ds.ReadAsArray()
    ds = None
    return arr

all_rare = set()
sample_to_scene = {}  # sample_name -> (scene, split)

for scene in ['Ports', 'Airports', 'Urban-Rural Areas']:
    scene_path = os.path.join(SOURCE_BASE, scene)
    if not os.path.isdir(scene_path):
        continue
    
    for split in ['train', 'val', 'test']:
        lbl_dir = os.path.join(scene_path, split, 'label')
        img_dir = os.path.join(scene_path, split, 'image')
        
        if not os.path.isdir(lbl_dir):
            continue
        
        for fname in os.listdir(lbl_dir):
            if not fname.endswith('.tif'):
                continue
            
            base = fname.replace('_target.tif', '').replace('.tif', '')
            path = os.path.join(lbl_dir, fname)
            
            try:
                label = read_label(path)
                unique = set(np.unique(label).tolist())
                
                if unique & (BRIDGE_RAW | SHELTER_RAW | DOCK_RAW):
                    all_rare.add(base)
                    sample_to_scene[base] = (scene, split)
            except:
                continue

print(f"Found {len(all_rare)} rare class samples")

# Create symlinks
print(f"\nCreating symlinks...")
linked = 0
errors = 0

for sample_name in sorted(all_rare):
    if sample_name not in sample_to_scene:
        continue
    
    scene, split = sample_to_scene[sample_name]
    
    # Source paths
    src_img_dir = os.path.join(SOURCE_BASE, scene, split, 'image')
    src_lbl_dir = os.path.join(SOURCE_BASE, scene, split, 'label')
    
    # Target paths - put in v2 train directory (will be used by all splits)
    tgt_img_dir = os.path.join(TARGET_BASE, 'train', 'image')
    tgt_lbl_dir = os.path.join(TARGET_BASE, 'train', 'label')
    
    os.makedirs(tgt_img_dir, exist_ok=True)
    os.makedirs(tgt_lbl_dir, exist_ok=True)
    
    # Link image files (pre and post)
    for suffix in ['_pre_war.tif', '_post_war.tif']:
        src = os.path.join(src_img_dir, sample_name + suffix)
        tgt = os.path.join(tgt_img_dir, sample_name + suffix)
        
        if os.path.isfile(src) and not os.path.exists(tgt):
            try:
                os.symlink(src, tgt)
                linked += 1
            except Exception as e:
                errors += 1
    
    # Link label file
    for suffix in ['_target.tif', '_label.tif']:
        src = os.path.join(src_lbl_dir, sample_name + suffix)
        tgt = os.path.join(tgt_lbl_dir, sample_name + suffix)
        
        if os.path.isfile(src) and not os.path.exists(tgt):
            try:
                os.symlink(src, tgt)
                linked += 1
            except Exception as e:
                errors += 1

print(f"\nResults:")
print(f"  Symlinks created: {linked}")
print(f"  Errors: {errors}")
print(f"\nDone! Restart training to use linked samples.")
