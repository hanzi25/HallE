deepspeed --include localhost:0,3 --master_port 25001 llava/train/train_switch_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /raid_sdd/whz/model/llava_1_5 \
    --version v1 \
    --data_path /raid_sdd/zzy/data/halle/detail_switch_23k.json \
    --image_folder /raid_sdd/zzy/data/halle/coco/train2017 \
    --vision_tower /raid_sdd/whz/model/clip_vit_large_patch14_336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir /raid_sdd/whz/experiments/halle/train/exp4_llava_controller_23k_1ep_8bz_2e5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    # --data_path /raid_sdd/whz/data/halle/detail_switch_minus_1_9093.json \
