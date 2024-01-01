from llava.model.multimodal_encoder.bev_mmdet3d.models.detectors import BEVFormer
from llava.model.multimodal_encoder.bev_mmdet3d.models.backbone import ResNet
from llava.model.multimodal_encoder.bev_mmdet3d.models.necks import FPN
from llava.model.multimodal_encoder.bev_mmdet3d.models.dense_heads import BEVFormerHead
from llava.model.multimodal_encoder.bev_mmdet3d.models.modules import (
    PerceptionTransformer,
    BEVFormerEncoder,
    BEVFormerLayer,
    TemporalSelfAttention,
    SpatialCrossAttention,
    MSDeformableAttention3D,
    DetrTransformerDecoderLayer,
    DetectionTransformerDecoder,
    CustomMSDeformableAttention,
)

from torch.nn import ReLU
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]


img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True
)

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
queue_length = 4  # each sequence contains `queue_length` frames.
aggregation_ops = "max_pooling" # avg_pooling, max_pooling, convolution
load_ckpt = "/home/scratch.chaoweix_nvresearch/visual_instruction/BEV-LLaVA/llava/model/multimodal_encoder/bev_mmdet3d/ckpts/bevformer_r101_dcn_24ep.pth"
dim_scale = 4

model = dict(
    type=BEVFormer,
    use_grid_mask=True,
    video_test_mode=True,
    img_backbone=dict(
        type=ResNet,
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True),
    ),
    img_neck=dict(
        type=FPN,
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs="on_output",
        num_outs=4,
        relu_before_extra_convs=True,
    ),
    pts_bbox_head=dict(
        type=BEVFormerHead,
        bev_h=bev_h_,
        bev_w=bev_w_,
        pc_range=point_cloud_range,
        num_query=900,
        num_classes=10,
        code_size=10,
        in_channels=_dim_,
        embed_dims=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type=PerceptionTransformer,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
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
                            type=TemporalSelfAttention, embed_dims=_dim_, num_levels=1
                        ),
                        dict(
                            type=SpatialCrossAttention,
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type=MSDeformableAttention3D,
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_,
                            ),
                            embed_dims=_dim_,
                        ),
                    ],
                    ffn_cfgs=dict(
                        type=FFN,
                        embed_dims=256,
                        feedforward_channels=_ffn_dim_,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type=ReLU, inplace=True),
                    ),
                    # feedforward_channels=_ffn_dim_,
                    # ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
            decoder=dict(
                type=DetectionTransformerDecoder,
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type=DetrTransformerDecoderLayer,
                    attn_cfgs=[
                        dict(
                            type=MultiheadAttention,
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type=CustomMSDeformableAttention,
                            embed_dims=_dim_,
                            num_levels=1,
                        ),
                    ],
                    ffn_cfgs=dict(
                        type=FFN,
                        embed_dims=256,
                        feedforward_channels=_ffn_dim_,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type=ReLU, inplace=True),
                    ),
                    # feedforward_channels=_ffn_dim_,
                    # ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
        ),
    ),
)

from llava.model.multimodal_encoder.bev_mmdet3d.datasets.pipelines import (
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


dataset_type = 'CustomNuScenesDataset'
data_root = '/home/scratch.chaoweix_nvresearch/av/AV-GPT/data/nuscenes/'
nuscenes_qa_file = '/home/scratch.chaoweix_nvresearch/visual_instruction/BEV-LLaVA/workspace/data/nuscenes-qa/'
drivelm_qa_file = '/home/scratch.chaoweix_nvresearch/visual_instruction/BEV-LLaVA/workspace/data/drivelm/'
train_dataset_length = 50000
eval_dataset_length = 500
file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type=LoadMultiViewImageFromFiles, to_float32=True),
    dict(type=PhotoMetricDistortionMultiViewImage),
    # dict(type=LoadAnnotations3D, with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    # dict(type=ObjectRangeFilter, point_cloud_range=point_cloud_range),
    # dict(type=ObjectNameFilter, classes=class_names),
    dict(type=NormalizeMultiviewImage, **img_norm_cfg),
    dict(type=PadMultiViewImage, size_divisor=32),
    dict(type=DefaultFormatBundle3D, class_names=class_names),
    dict(type=CustomCollect3D, keys=['img'])
]

test_pipeline = [
    dict(type=LoadMultiViewImageFromFiles, to_float32=True),
    dict(type=NormalizeMultiviewImage, **img_norm_cfg),
    dict(type=PadMultiViewImage, size_divisor=32),
    dict(type=DefaultFormatBundle3D, class_names=class_names),
    dict(type=CustomCollect3D, keys=['img'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
        # nuscenes_qa_file=nuscenes_qa_file + "NuScenes_train_questions.json",
        drivelm_qa_file=drivelm_qa_file + 'v1_0_train_nus_mini.json',
        # dataset_length=train_dataset_length,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
             # nuscenes_qa_file=nuscenes_qa_file + "NuScenes_val_questions.json",
             drivelm_qa_file=drivelm_qa_file + 'v1_0_train_nus_mini.json',
             dataset_length=eval_dataset_length,
             pipeline=test_pipeline,
             bev_size=(bev_h_, bev_w_),
             queue_length=queue_length,
             classes=class_names,
             modality=input_modality,
             samples_per_gpu=1),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
              # nuscenes_qa_file=nuscenes_qa_file + "NuScenes_val_questions.json",
              drivelm_qa_file=drivelm_qa_file + 'v1_0_train_nus_mini.json',
              dataset_length=eval_dataset_length,
              pipeline=test_pipeline,
              bev_size=(bev_h_, bev_w_),
              queue_length=queue_length,
              classes=class_names,
              modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

evaluation = dict(interval=1, pipeline=test_pipeline)
