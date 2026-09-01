"""
Patch Stitcher - Merge prediction patches back to full image.
"""

import torch
import numpy as np


class PatchStitcher:
    """
    Stitch prediction patches back to full-resolution images.
    
    Usage:
        stitcher = PatchStitcher(img_h=1024, img_w=1024, patch_size=256)
        stitcher.add(patch_pred, patch_y, patch_x, scene, base_name)
        full_pred = stitcher.get(scene, base_name)
    """
    
    def __init__(self, img_h, img_w, patch_size, num_classes):
        self.img_h = img_h
        self.img_w = img_w
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.buffer = {}  # {(scene, base_name): accumulator}
    
    def reset(self):
        self.buffer = {}
    
    def add(self, pred, patch_y, patch_x, scene, base_name):
        """
        Add a prediction patch.
        
        Args:
            pred: [num_classes, ps, ps] logits or [ps, ps] class indices
            patch_y, patch_x: top-left coordinates in original image
        """
        key = (scene, base_name)
        
        if key not in self.buffer:
            self.buffer[key] = {
                'pred_sum': torch.zeros(self.num_classes, self.img_h, self.img_w),
                'count': torch.zeros(self.img_h, self.img_w),
            }
        
        ps = self.patch_size
        buf = self.buffer[key]
        
        if pred.dim() == 3:
            # [C, ps, ps] logits
            buf['pred_sum'][:, patch_y:patch_y+ps, patch_x:patch_x+ps] += pred
        elif pred.dim() == 2:
            # [ps, ps] class indices -> one-hot
            one_hot = torch.zeros(self.num_classes, ps, ps)
            for c in range(self.num_classes):
                one_hot[c] = (pred == c).float()
            buf['pred_sum'][:, patch_y:patch_y+ps, patch_x:patch_x+ps] += one_hot
        
        buf['count'][patch_y:patch_y+ps, patch_x:patch_x+ps] += 1
    
    def get(self, scene, base_name):
        """
        Get stitched prediction for one image.
        
        Returns:
            pred: [img_h, img_w] class indices
        """
        key = (scene, base_name)
        if key not in self.buffer:
            return None
        
        buf = self.buffer[key]
        # Average overlapping predictions
        pred_avg = buf['pred_sum'] / buf['count'].clamp(min=1)
        return pred_avg.argmax(dim=0)  # [img_h, img_w]
    
    def get_all(self):
        """Get all stitched predictions."""
        results = {}
        for key in self.buffer:
            results[key] = self.get(*key)
        return results