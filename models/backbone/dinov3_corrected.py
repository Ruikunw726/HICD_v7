"""Versioned DINO input/feature adapter; the legacy wrapper stays unchanged."""
import torch
from torch.nn import functional as F
from .dinov3 import DINOv3Backbone

INTEGRATION_VERSION = 'hicd.dinov3_input_features.v1'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DINOv3CorrectedBackbone(DINOv3Backbone):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('input_mean', torch.tensor(IMAGENET_MEAN).view(1,3,1,1), persistent=False)
        self.register_buffer('input_std', torch.tensor(IMAGENET_STD).view(1,3,1,1), persistent=False)
        if not hasattr(self.backbone, 'forward_intermediates'):
            raise RuntimeError('This integration requires timm forward_intermediates')

    def normalize_input(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError('Expected RGB B,3,H,W input in dataset [0,1] units')
        return (image - self.input_mean.to(image.dtype)) / self.input_std.to(image.dtype)

    def extract_tokens(self, image):
        # timm Eva handles RoPE, registers and after-block collection itself.
        return self.backbone.forward_intermediates(
            self.normalize_input(image), indices=self.extract_blocks, norm=True,
            output_fmt='NLC', intermediates_only=True)

    def forward(self, image):
        h,w = image.shape[-2:]
        if h % self.patch_size or w % self.patch_size:
            raise ValueError('Input must be divisible by the pretrained patch size')
        hp,wp = h // self.patch_size,w // self.patch_size
        tokens = self.extract_tokens(image)
        features = []
        for head,x,channels in zip(self.proj_heads,tokens,self.out_channels):
            x = head(x).transpose(1,2).reshape(image.shape[0],channels,hp,wp)
            features.append(x)
        return [F.interpolate(features[0],scale_factor=4,mode='bilinear',align_corners=True),
                F.interpolate(features[1],scale_factor=2,mode='bilinear',align_corners=True),
                features[2],F.max_pool2d(features[3],kernel_size=2,stride=2)]
