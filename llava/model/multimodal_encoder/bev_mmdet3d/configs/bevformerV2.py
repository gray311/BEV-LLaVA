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


##### data config

dataset_type = 'CustomNuScenesDatasetV2'
data_root = '/home/scratch.chaoweix_nvresearch/av/AV-GPT/data/nuscenes/'
nuscenes_qa_file = "/home/scratch.chaoweix_nvresearch/visual_instruction/BEV-LLaVA/workspace/data/nuscenes-qa/"


ida_aug_conf = {
    "reisze": [512, 544, 576, 608, 640, 672, 704, 736, 768],  #  (0.8, 1.2)
    "crop": (0, 260, 1600, 900),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}
ida_aug_conf_eval = {
    "reisze": [640, ],
    "crop": (0, 260, 1600, 900),
    "H": 900,
    "W": 1600,
    "rand_flip": False,
}


from bev_mmdet3d.datasets.pipelines import (
    LoadMultiViewImageFromFiles,
    LoadAnnotations3D,
    ObjectRangeFilter,
    ObjectNameFilter,
    MultiScaleFlipAug3D,
    CropResizeFlipImage,
    NormalizeMultiviewImage,
    PadMultiViewImage,
    DefaultFormatBundle3D,
    CustomCollect3D,
    PhotoMetricDistortionMultiViewImage,
    GlobalRotScaleTransImage,
)

train_pipeline = [
    dict(type=LoadMultiViewImageFromFiles, to_float32=True),
    dict(type=PhotoMetricDistortionMultiViewImage),
    dict(type=LoadAnnotations3D, with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type=GlobalRotScaleTransImage,
        rot_range=[-22.5, 22.5],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        reverse_angle=True,
        training=True,
        flip_dx_ratio=0.5,
        flip_dy_ratio=0.5,
        only_gt=True,),
    dict(
        type=ObjectRangeFilter,
        point_cloud_range=point_cloud_range),
    dict(
        type=ObjectNameFilter,
        classes=class_names),
    dict(type=CropResizeFlipImage, data_aug_conf=ida_aug_conf, training=True, debug=False),
    dict(type=NormalizeMultiviewImage, **img_norm_cfg),
    dict(type=PadMultiViewImage, size_divisor=32),
    dict(type=DefaultFormatBundle3D, class_names=class_names),
    dict(
        type=CustomCollect3D,
        keys=['gt_bboxes_3d', 'gt_labels_3d', 'img',
              'ego2global_translation', 'ego2global_rotation', 'lidar2ego_translation', 'lidar2ego_rotation',
              'timestamp', 'mono_input_dict', 'mono_ann_idx', 'aug_param']),
]
eval_pipeline = [
    dict(type=LoadMultiViewImageFromFiles, to_float32=True, ),
    dict(type=CropResizeFlipImage, data_aug_conf=ida_aug_conf_eval, training=False, debug=False),
    dict(type=NormalizeMultiviewImage, **img_norm_cfg),
    dict(type=PadMultiViewImage, size_divisor=32),
    dict(
        type=MultiScaleFlipAug3D,
        img_scale=(1600, 640),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type=DefaultFormatBundle3D,
                class_names=class_names,
                with_label=False),
            dict(type=CustomCollect3D,
                 keys=['img', 'ego2global_translation', 'ego2global_rotation', 'lidar2ego_translation',
                       'lidar2ego_rotation', 'timestamp'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
        type='CustomNuScenesDatasetV2',
        frames=frames,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
        nuscenes_qa_file=nuscenes_qa_file + "NuScenes_train_questions.json",
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR',
        mono_cfg=dict(
            name='nusc_trainval',
            data_root=data_root,
            min_num_lidar_points=3,
            min_box_visibility=0.2)),
    val=dict(
        type='CustomNuScenesDatasetV2',
        frames=frames,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
        nuscenes_qa_file=nuscenes_qa_file + "NuScenes_val_questions.json",
        pipeline=eval_pipeline,
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1),
    test=dict(
        type='CustomNuScenesDatasetV2',
        frames=frames,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
        nuscenes_qa_file=nuscenes_qa_file + "NuScenes_val_questions.json",
        pipeline=eval_pipeline,
        classes=class_names,
        modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
evaluation = dict(interval=4, pipeline=eval_pipeline)

