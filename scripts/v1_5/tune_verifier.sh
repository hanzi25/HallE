deepspeed --include localhost:3 --master_port 25432 llava/train/train_verifier.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /raid_sdd/zzy/model/llava_1_5 \
    --version v1 \
    --data_path /raid_sdd/zzy/data/halle/detail_switch_minus_1_9093.json \
    --image_folder /raid_sdd/zzy/data/halle/coco/train2017 \
    --vision_tower /raid_sdd/zzy/model/clip_vit_large_patch14_336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --alpha_type scalar \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir /raid_sdd/zzy/experiments/halle/train/exp5_llava_verifier_scalar_init_0.1_minus1_9k_3ep_16bz_3e4 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 3e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \

#CUDA_VISIBLE_DEVICES=1 python3 eval/model_controller.py \
#            --model-path $model_path \
#            --model-version llava_verifier \
#            --bf16 True \
#            --gt_file_path /raid_sdd/zzy/data/halle/coco/coco2014/annotations/instances_val2014.json \
#            --image_path /raid_sdd/zzy/data/halle/coco/coco2014/val2014 \
#            --output_folder $model_path/eval