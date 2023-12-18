# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import copy
from collections import OrderedDict
import torch
from torch import nn

from .. import builder
from mmdet.registry import MODELS
from ..utils.grid_mask import GridMask
import copy


@MODELS.register_module()
class BEVFormerV2(nn.Module):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 fcos3d_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 num_levels=None,
                 num_mono_levels=None,
                 mono_loss_weight=1.0,
                 frames=(0,),
                 ):

        super(BEVFormerV2,self).__init__()
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        assert not self.fp16_enabled  # not support fp16 yet
        # temporal
        self.video_test_mode = video_test_mode
        assert not self.video_test_mode  # not support video_test_mode yet

        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)

        if img_backbone:
            self.img_backbone = builder.build_vision_encoder(img_backbone)
        if img_neck is not None:
            self.with_img_neck = True
            self.img_neck = builder.build_neck(img_neck)

        # levels of features
        self.num_levels = num_levels
        self.num_mono_levels = num_mono_levels
        self.frames = frames

    def extract_img_feat(self, img):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(self, img, img_metas, len_queue=None):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img)
        if 'aug_param' in img_metas[0] and img_metas[0]['aug_param']['CropResizeFlipImage_param'][-1] is True:
            # flip feature
            img_feats = [torch.flip(x, dims=[-1, ]) for x in img_feats]
        return img_feats

    def forward_pts(self, pts_feats, img_metas, prev_bev=None):
        return self.pts_bbox_head(pts_feats, img_metas, prev_bev, only_bev=True)

    def obtain_history_bev(self, img_dict, img_metas_dict):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        # Modify: roll back to previous version for single frame
        is_training = self.training
        self.eval()
        prev_bev = OrderedDict({i: None for i in self.frames})
        with torch.no_grad():
            for t in img_dict.keys():
                img = img_dict[t]
                img_metas = [img_metas_dict[t], ]
                img_feats = self.extract_feat(img=img, img_metas=img_metas)
                if self.num_levels:
                    img_feats = img_feats[:self.num_levels]
                bev = self.pts_bbox_head(
                    img_feats, img_metas, None, only_bev=True)
                prev_bev[t] = bev.detach()
        if is_training:
            self.train()
        return list(prev_bev.values())

    def forward(self, img_metas=None, img=None,):
        img_meta = OrderedDict(sorted(img_metas[0].items()))
        img_dict = {}

        for ind, t in enumerate(img_meta.keys()):
            img_dict[t] = img[:, ind, ...]

        img = img_dict[0]
        img_dict.pop(0)

        prev_img_metas = copy.deepcopy(img_meta)
        prev_img_metas.pop(0)
        prev_bev = self.obtain_history_bev(img_dict, prev_img_metas)

        for item in prev_bev:
            if item is not None:
                print(item.shape)

        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        outputs = self.forward_pts(img_feats if self.num_levels is None else img_feats[:self.num_levels],
                                         img_metas=img_metas, prev_bev=prev_bev)
        return outputs






