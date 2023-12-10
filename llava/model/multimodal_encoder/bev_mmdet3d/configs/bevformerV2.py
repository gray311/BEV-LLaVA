from bev_mmdet3d.models.detectors import BEVFormer, BEVFormerV2
from bev_mmdet3d.models.backbone import ResNet
from bev_mmdet3d.models.necks import FPN
from bev_mmdet3d.models.dense_heads import BEVFormerHead, BEVFormerHead_GroupDETR
from bev_mmdet3d.models.modules import (
    PerceptionTransformer,
    PerceptionTransformerV2,
    BEVFormerEncoder,
    BEVFormerLayer,
    TemporalSelfAttention,
    SpatialCrossAttention,
    MSDeformableAttention3D,
    DetrTransformerDecoderLayer,
    DetectionTransformerDecoder,
    CustomMSDeformableAttention,
    GroupMultiheadAttention
)

from torch.nn import ReLU
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention


point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
    'pedestrian', 'traffic_cone', 'trailer', 'truck'
]

# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True
)

img_norm_cfg = dict(mean=[103.53, 116.28, 123.675], std=[1, 1, 1], to_rgb=False)
bev_h_ = 200
bev_w_ = 200
_num_patches_ = 288
frames = (-7,-6,-5,-4,-3,-2,-1,0)
group_detr = 11
voxel_size = [102.4 / bev_h_, 102.4 / bev_w_, 8]

# model
_dim_ = 256
_pos_dim_ = 128
_ffn_dim_ = 512
_num_levels_ = 4
_num_mono_levels_ = 5

model = dict(
    type=BEVFormerV2,
    use_grid_mask=True,
    video_test_mode=False,
    num_levels=_num_levels_,
    num_mono_levels=_num_mono_levels_,
    mono_loss_weight=1.0,
    frames=frames,
    img_backbone=dict(
        type=ResNet,
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN'),
        norm_eval=False,
        style='caffe'),
    img_neck=dict(
        type=FPN,
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_mono_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type=BEVFormerHead_GroupDETR,
        group_detr=group_detr,
        bev_h=bev_h_,
        bev_w=bev_w_,
        pc_range=point_cloud_range,
        embed_dims=_dim_,
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type=PerceptionTransformerV2,
            embed_dims=_dim_,
            frames=frames,
            inter_channels=_dim_*2,
            encoder=dict(
                type=BEVFormerEncoder,
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type=BEVFormerLayer,
                    attn_cfgs=[
                        dict(
                            type=TemporalSelfAttention,
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type=SpatialCrossAttention,
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type=MSDeformableAttention3D,
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=4),
                            embed_dims=_dim_)
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type=DetectionTransformerDecoder,
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type=DetrTransformerDecoderLayer,
                    attn_cfgs=[
                        dict(
                            type=GroupMultiheadAttention,
                            group=group_detr,
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type=CustomMSDeformableAttention,
                            embed_dims=_dim_,
                            num_levels=1)
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')
                )
            )
        )
    )
)


