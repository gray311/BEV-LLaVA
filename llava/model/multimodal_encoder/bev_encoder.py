import argparse
import copy
import mmcv
import os
import time
import numpy as np
import torch
import torch.nn as nn
import warnings
from os import path as osp

from mmengine import Config, DictAction
from bev_mmdet3d.models.builder import build_model
from transformers import AutoImageProcessor

"""
args.bev_config

"""
class BEVVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.args = args
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.load_model()

    def load_model(self):
        cfg = Config.fromfile(self.args.bev_config_file)
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        self.vision_tower = build_model(cfg.model)
        self.vision_tower.requires_grad_(False)
        self.model_config = dict(cfg)['model']
        self._dim_ = dict(cfg)['_dim_']
        self._num_patches_ = dict(cfg)['_num_patches_']

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(img=image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                print(image_forward_out.shape)

                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features


    @property
    def dtype(self):
        return self.vision_tower.pts_bbox_head.positional_encoding.row_embed.weight.dtype

    @property
    def device(self):
        return self.vision_tower.pts_bbox_head.positional_encoding.row_embed.weight.device

    @property
    def config(self):
        return self.model_config

    @property
    def hidden_size(self):
        return self._dim_

    @property
    def num_patches(self):
        return self._num_patches_

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.bev_config_file = "/home/myz/LLaVA/llava/model/multimodal_encoder/bev_mmdet3d/configs/bevformerV2.py"
    args.mm_vision_select_layer = -1
    image_tower = BEVVisionTower("bevformer", args)

    from PIL import Image
    sample_path = "/home/myz/LLaVA/llava/model/multimodal_encoder/samples"

    for image_id in os.listdir(sample_path):
        image_path = os.path.join(sample_path, image_id)
        image_list = [Image.open(os.path.join(image_path, image_name)).convert("RGB") \
                       for image_name in os.listdir(image_path)]
        image_list = [torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                      for image in image_list]

        image_tensor = torch.concat(image_list, axis=0)
        outputs = image_tower(images=image_tensor)
        print(outputs.shape)







"""
n015-2018-07-18-11-50-34+0800__CAM_BACK__1531886157037525


"""