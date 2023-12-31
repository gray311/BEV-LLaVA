#!/bin/bash

DISABLE_ADDMM_CUDA_LT=1 CUDA_VISIBLE_DEVICES=0 deepspeed --num_gpus=1 llava/train/train_mem.py \
    --deepspeed ./configs/accelerate_config/zero2.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --bev_tower ./llava/model/multimodal_encoder/bev_mmdet3d/configs/bevformer.py \
    --data_config ./llava/model/multimodal_encoder/bev_mmdet3d/configs/bevformer.py \
    --pretrain_mm_mlp_adapter ./workspace/checkpoints/bev-vicuna-v1.0-7b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_x_start_end False \
    --mm_use_x_patch_token False \
    --fp16 True \
    --output_dir /home/scratch.chaoweix_nvresearch/visual_instruction/BEV-LLaVA/workspace/checkpoints/bev-vicuna-v1.0-7b-pretrain \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --mm_projector_lr 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --lazy_preprocess True \
    --report_to wandb