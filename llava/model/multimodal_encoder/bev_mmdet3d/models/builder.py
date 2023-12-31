# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from mmengine import MODELS
from mmengine.registry import Registry

# MODELS = Registry("models", parent=MODELS)
#
# BACKBONES = MODELS
# DETECTORS = MODELS
# HEADS = MODELS
# LOSSES = MODELS
# NECKS = MODELS
# ROI_EXTRACTORS = MODELS
# SHARED_HEADS = MODELS
# VOXEL_ENCODERS = MODELS
# MIDDLE_ENCODERS = MODELS
# FUSION_LAYERS = MODELS
#
#
def build_vision_encoder(cfg):
    """Build backbone."""
    return MODELS.build(cfg)

def build_backbone(cfg):
    """Build backbone."""
    return MODELS.build(cfg)

def build_neck(cfg):
    """Build neck."""
    return MODELS.build(cfg)


def build_extractor(cfg):
    """Build RoI feature extractor."""
    return MODELS.build(cfg)


def build_shared_head(cfg):
    """Build shared head of detector."""
    return MODELS.build(cfg)


def build_head(cfg):
    """Build head."""
    return MODELS.build(cfg)


def build_loss(cfg):
    """Build loss function."""
    return MODELS.build(cfg)


def build_bev_encoder(cfg, train_cfg=None, test_cfg=None):
    """Build bev vision encoder."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            "train_cfg and test_cfg is deprecated, " "please specify them in model",
            UserWarning,
        )
    assert (
        cfg.get("train_cfg") is None or train_cfg is None
    ), "train_cfg specified in both outer field and model field "
    assert (
        cfg.get("test_cfg") is None or test_cfg is None
    ), "test_cfg specified in both outer field and model field "
    return MODELS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg)
    )

def build_model(cfg, train_cfg=None, test_cfg=None):
    return build_bev_encoder(cfg, train_cfg=train_cfg, test_cfg=test_cfg)
