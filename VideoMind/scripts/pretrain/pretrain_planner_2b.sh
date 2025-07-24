#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_RT_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH="./:$PYTHONPATH"

ckpt_path="work_dirs/pretrain_planner_2b"

torchrun --nproc_per_node 8 videomind/train/train.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path model_zoo/Qwen2-VL-2B-Instruct \
    --base_model qwen2_vl \
    --conv_type chatml \
    --role planner \
    --lora_enable True \
    --lora_type qkvo \
    --lora_r 64 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --lora_bias none \
    --tuning_modules none \
    --datasets mixed_planning \
    --min_video_len 5 \
    --max_video_len 500 \
    --max_num_words 200 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --output_dir $ckpt_path \
    --save_full_model False \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --fp16 True \
    --report_to tensorboard
