#!/bin/bash
deepspeed --include localhost:3 --master_port 25438 llava/train/train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256\
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /raid_sdd/zzy/model/llava_1_5 \
    --version v1 \
    --data_path /raid_sdd/zzy/data/halle/sharegpt4v_instruct_gpt4-vision_part_coco_sample_9k.json \
    --image_folder /raid_sdd/zzy/data/halle/coco/train2017 \
    --vision_tower /raid_sdd/zzy/model/clip_vit_large_patch14_336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir /raid_sdd/zzy/experiments/halle/train/exp7_llava_lora_sharegpt_9k_1ep_16bz_2e4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True
<<<<<<< HEAD

=======
>>>>>>> fix
