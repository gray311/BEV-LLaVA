# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
from torch import nn

from .. import builder
from mmdet.registry import MODELS
from ..utils.grid_mask import GridMask
import copy


@MODELS.register_module()
class BEVFormer(nn.Module):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(
        self,
        use_grid_mask=False,
        img_backbone=None,
        img_neck=None,
        pts_bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
    ):
        super(BEVFormer, self).__init__()

        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
        )
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }

        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            print(pts_bbox_head)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)

        if img_backbone:
            self.img_backbone = builder.build_vision_encoder(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:

            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

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
            if len_queue is not None:
                img_feats_reshaped.append(
                    img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W)
                )
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)

        return img_feats


    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)

            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]["prev_bev_exists"]:
                    prev_bev = None
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True
                )
            self.train()
            return prev_bev


    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img=None,
        proposals=None,
        gt_bboxes_ignore=None,
        img_depth=None,
        img_mask=None,
    ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue - 1] for each in img_metas]
        if not img_metas[0]["prev_bev_exists"]:
            prev_bev = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        outputs = self.pts_bbox_head(pts_feats, img_metas, prev_bev)

        return outputs

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img

        assert len(img_metas) == 1 and len(img) == 1

        if img_metas[0][0]["scene_token"] != self.prev_frame_info["scene_token"]:
            # the first sample of each scene is truncated
            self.prev_frame_info["prev_bev"] = None
        # update idx
        self.prev_frame_info["scene_token"] = img_metas[0][0]["scene_token"]

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        # Get the delta of ego position and angle between two timestamps.
        # tmp_pos = copy.deepcopy(img_metas[0][0]["can_bus"][:3])
        # tmp_angle = copy.deepcopy(img_metas[0][0]["can_bus"][-1])
        # if self.prev_frame_info["prev_bev"] is not None:
        #     img_metas[0][0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
        #     img_metas[0][0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
        # else:
        #     img_metas[0][0]["can_bus"][-1] = 0
        #     img_metas[0][0]["can_bus"][:3] = 0

        new_prev_bev = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info["prev_bev"], **kwargs
        )

        return new_prev_bev

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        return self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentation."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale
        )

        return new_prev_bev
