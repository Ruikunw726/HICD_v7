"""Explicit builder for new runs; never changes legacy build_hicd_v7 defaults."""
import importlib
from unittest.mock import patch
from .backbone.dinov3_corrected import DINOv3CorrectedBackbone, INTEGRATION_VERSION


def build_dino_corrected_hicd(config):
    if config.get('dino_integration_version') != INTEGRATION_VERSION:
        raise ValueError('Corrected DINO runs must explicitly declare the integration version')
    if config['backbone']['type'] != 'dinov3':
        raise ValueError('Corrected adapter only supports DINOv3')
    module = importlib.import_module('models.HICD_v7')
    def build_backbone(cfg):
        b = cfg['backbone']
        return DINOv3CorrectedBackbone(
            variant=b.get('variant','dinov3_base'),pretrained=b.get('pretrained',True),
            pretrained_path=b.get('pretrained_path'),out_channels=b.get('out_channels',[64,128,256,512]),
            freeze_stages=b.get('freeze_stages',2))
    with patch.object(module,'build_backbone',build_backbone):
        return module.HICD_v7(config)
