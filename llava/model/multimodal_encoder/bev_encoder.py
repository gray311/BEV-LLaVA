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

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, default_collate
from transformers import AutoImageProcessor

from mmengine import MODELS, Config
from mmengine.model import xavier_init, constant_init
from mmengine.model import BaseModule, ModuleList, Sequential
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.cnn import build_norm_layer, build_conv_layer
from mmdet.models.backbones.resnet import Bottleneck, BasicBlock
import torch.utils.checkpoint as checkpoint

from llava.model.multimodal_encoder.bev_mmdet3d.models.builder import build_model
from llava.model.multimodal_encoder.bev_mmdet3d.datasets.builder import build_dataloader, custom_build_dataset

class ReduceBEVFeatures(nn.Module):
    def __init__(self):
        super(ReduceBEVFeatures, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=3),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv_layers(x)
        x = x.permute(0, 2, 3, 1)
        return x

class BEVVisionTower(nn.Module):
    def __init__(self, vision_tower, args=None, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.args = args
        self.vision_tower_name = vision_tower
        self.load_model()

    def load_model(self, checkpoint_path=None):
        cfg = Config.fromfile(self.vision_tower_name)
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        self.bev_h, self.bev_w = cfg.bev_h_, cfg.bev_w_
        self.aggregation_ops = cfg.aggregation_ops
        self.vision_tower = build_model(cfg.model)
        # self.vision_tower.requires_grad_(False)
        self.model_config = dict(cfg)['model']
        self._dim_ = cfg._dim_ * cfg.dim_scale
        if checkpoint_path is None:
            checkpoint = torch.load(cfg.load_ckpt)
        else:
            checkpoint = torch.load(checkpoint_path)
        self.vision_tower.load_state_dict(checkpoint['state_dict'], strict=False)
        if self.aggregation_ops == "convolution":
            self.fusion_module = ReduceBEVFeatures()

    def bev_feature_aggregate(self, image_forward_outs):
        # bs, bev_h * bev_w, dim
        bs, _, dim = image_forward_outs.shape
        image_forward_outs= image_forward_outs.reshape(bs, self.bev_h, self.bev_w, dim)
        if self.aggregation_ops == "avg_pooling":
            pool_size, stride = 5, 5
            aggregated_bev_features = F.avg_pool2d(image_forward_outs.permute(0, 3, 1, 2), pool_size, stride).permute(0, 2, 3, 1)
        elif self.aggregation_ops == "max_pooling":
            pool_size, stride = 5, 5
            aggregated_bev_features = F.max_pool2d(image_forward_outs.permute(0, 3, 1, 2), pool_size, stride).permute(0, 2, 3, 1)
        elif self.aggregation_ops == "convolution":
            print(self.fusion_module)
            aggregated_bev_features = self.fusion_module(image_forward_outs)
        else:
            raise ValueError(f'Unknown Aggregation Operation: {self.aggregation_ops}')

        return aggregated_bev_features.reshape(bs, -1, dim * 4)

    def forward(self, img_metas, img):
        if type(img) is list:
            assert type(img_metas) is list
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(img_metas=img_metas, img=img.to(device=self.device, dtype=self.dtype))
                image_feature = self.bev_feature_aggregate(image_forward_out).to(self.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(img_metas=img_metas, img=img.to(device=self.device, dtype=self.dtype))
            image_features = self.bev_feature_aggregate(image_forward_outs).to(self.dtype)

        return image_features

    def get_bev_features(self, img_metas, img):
        if type(img) is list:
            assert type(img_metas) is list
            bev_features = []
            for image in images:
                image_forward_out = self.vision_tower(img_metas=img_metas, img=img.to(device=self.device, dtype=self.dtype))
                bev_features.append(image_feature)
        else:
            bev_features = self.vision_tower(img_metas=img_metas, img=img.to(device=self.device, dtype=self.dtype))

        return bev_features

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
    # @property
    # def num_patches(self):
    #     return self._num_patches_

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


def collate(instances):
    batch = dict()

    batch['instruction'] = [instance['instruction'].format(question=instance['question']) for instance in instances]
    batch['answer'] = [instance['answer'] for instance in instances]


    if 'img' in instances[0] and 'img_metas' in instances[0]:
        bev_imgs = [instance['img'] for instance in instances]
        if all(x is not None and x.shape == bev_imgs[0].shape for x in bev_imgs):
            batch['img'] = torch.stack(bev_imgs)
        else:
            batch['img'] = bev_imgs
        batch['img_metas'] = [instance['img_metas'] for instance in instances]

    return batch


if __name__ == "__main__":
    vision_tower = "/home/scratch.chaoweix_nvresearch/visual_instruction/BEV-LLaVA/llava/model/multimodal_encoder/bev_mmdet3d/configs/bevformer.py"
    # image_tower = BEVVisionTower(vision_tower)
    # image_tower.cuda()
    cfg = Config.fromfile(vision_tower)
    dataset = custom_build_dataset(cfg.data.train)
    print(len(dataset))

    train_dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        # sampler=DistributedSampler(dataset, shuffle=True, drop_last=True),
        shuffle=False,
        collate_fn=collate,
    )

    from tqdm import tqdm
    cnt = 0
    for batch in tqdm(train_dataloader):
        cnt += 1



    # from tqdm import tqdm
    # for i in tqdm(range(len(dataset))):
    #     print(dataset[i]['img_metas'].keys())
#
#     outputs = image_tower(img_metas=[dataset[8]['img_metas'], dataset[9]['img_metas']],
#                           img=torch.stack([dataset[8]['img'],
#                                            dataset[9]['img']]))
#     print(outputs.shape)
#










