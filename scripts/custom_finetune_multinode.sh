#!/bin/bash

deepspeed --hostfile ./scripts/hostfile_v128 \
    llava/train/train_xformers.py \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /zecheng/svg/icon-shop/meta_data/svg_to_image/train_llava_image_to_svg.json \
    --image_folder /zecheng/svg/icon-shop/meta_data/svg_to_image/rendered_train_image_nc \
    --vision_tower /zecheng/svg_model_hub/llava_back/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /zecheng/svg_model_hub/llava_back/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 False \
    --fp16 True \
    --output_dir /zecheng/svg_model_hub/custom_llava \
    --num_train_epochs 30 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 100000 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 20 \
    --lazy_preprocess True \
    --report_to tensorboard;
 